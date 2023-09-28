import copy

import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn.data_parallel import DataParallel
from tqdm import tqdm

from datasets.steric_clash import get_steric_clash_atom_pairs, get_rec_elements, get_ligand_elements, \
    get_steric_clash_per_flexble_sidechain_atom
from filtering.dataset import ListDataset
from utils import so3, torus
from utils.sampling import randomize_position, sampling
import torch
from utils.diffusion_utils import get_t_schedule, get_inverse_schedule
from datasets.sidechain_conformer_matching import RMSD


def loss_function(tr_pred, rot_pred, tor_pred, sc_tor_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1,sc_tor_weight=1, apply_mean=True, no_torsion=False,flexible_sidechains=False):
    tr_sigma, rot_sigma, tor_sigma, sidechain_tor_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor', 'sc_tor']])

    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score) ** 2 * tr_sigma ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
        tor_loss = ((tor_pred.cpu() - tor_score) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    # sidechain torsion component
    if flexible_sidechains:
        sc_edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.sidechain_tor_sigma_edge for d in data] if device.type == 'cuda' else data.sidechain_tor_sigma_edge))
        sc_tor_score = torch.cat([d.sidechain_tor_score for d in data], dim=0) if device.type == 'cuda' else data.sidechain_tor_score

        sc_tor_score_norm2 = torch.tensor(torus.score_norm(sc_edge_tor_sigma.cpu().numpy())).float()
        sc_tor_loss = ((sc_tor_pred.cpu() - sc_tor_score) ** 2 / sc_tor_score_norm2)
        sc_tor_base_loss = ((sc_tor_score ** 2 / sc_tor_score_norm2)).detach()
        if apply_mean:
            sc_tor_loss, sc_tor_base_loss = sc_tor_loss.mean() * torch.ones(1, dtype=torch.float), sc_tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            if device.type == 'cuda':
                # TODO: this has NOT been checked
                index = torch.cat([torch.ones(len(d['flexResidues'].edge_idx)) * i for i, d in
                               enumerate(data)]).long()
            else:
                # the flexResidues are always indexed from 0...atoms
                # but in a batched-scenario, this needs to be adjusted
                # thus we add the correct offset in the ['atom'] graph
                # We do not use bin counts because here we can be certain that the batches are sorted
                _, atom_bin_counts = data['atom'].batch.unique(sorted=True, return_counts=True)
                bond_offset = atom_bin_counts.cumsum(dim=0)
                # shift it by one so to speak, because the first batch does not have an offset
                bond_offset = (torch.cat((torch.zeros(1, device=bond_offset.device), bond_offset))[:-1]).long()

                # store the bonds of the flexible residues. i.e., we store which atoms are connected
                index = data['atom'].batch[bond_offset[data['flexResidues'].batch]]


            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(sc_tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, sc_tor_loss)
            t_b_l.index_add_(0, index, sc_tor_base_loss)
            sc_tor_loss, sc_tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            sc_tor_loss, sc_tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            sc_tor_loss, sc_tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float) # ?

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + sc_tor_loss * sc_tor_weight
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), sc_tor_loss.detach(), tr_base_loss, rot_base_loss, tor_base_loss, sc_tor_base_loss


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):

        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weigths):
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'sc_tor_loss','tr_base_loss', 'rot_base_loss', 'tor_base_loss','sc_tor_base_loss'])

    for data in tqdm(loader, total=len(loader)):
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            tr_pred, rot_pred, tor_pred, sc_tor_pred = model(data)
        
            loss, tr_loss, rot_loss, tor_loss, sc_tor_loss, tr_base_loss, rot_base_loss, tor_base_loss, sc_tor_base_loss = \
                loss_fn(tr_pred, rot_pred, tor_pred, sc_tor_pred, data=data, t_to_sigma=t_to_sigma, device=device)
            if loss.isnan():
                print("SKIPPING backward pass for batch, loss is nan. This could indicate that the batch has no ligand torsion or sidechain torsions")
            else:
                loss.backward()
                optimizer.step()
                ema_weigths.update(model.parameters())
                meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, sc_tor_loss, tr_base_loss, rot_base_loss, tor_base_loss, sc_tor_base_loss])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    # TODO: meter sc_tor_loss and stuff see train epoch
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'sc_tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'sc_tor_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'sc_tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'sc_tor_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred, sc_tor_pred = model(data)

            loss, tr_loss, rot_loss, tor_loss, sc_tor_loss, tr_base_loss, rot_base_loss, tor_base_loss, sc_tor_base_loss = \
                loss_fn(tr_pred, rot_pred, tor_pred, sc_tor_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, sc_tor_loss, tr_base_loss, rot_base_loss, tor_base_loss, sc_tor_base_loss])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor, complex_t_sc_tor =\
                    [torch.cat([data[i].complex_t[noise_type] for i in range(len(data))]) for noise_type in ['tr', 'rot', 'tor', 'sc_tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                sigma_index_sc_tor = torch.round(complex_t_sc_tor.cpu() * (10 - 1)).long()
                
                if isinstance(model, DataParallel):
                    flex = model.module.flexible_sidechains                    
                else:
                    flex = model.flexible_sidechains
                if not flex:
                    meter_all.add(
                        [loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss],
                        [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_rot,
                        sigma_index_tor])
                else:
                    meter_all.add(
                        [loss.cpu().detach(), tr_loss, rot_loss, tor_loss, sc_tor_loss, tr_base_loss, rot_base_loss, tor_base_loss, sc_tor_base_loss],
                        [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_sc_tor, sigma_index_tr, sigma_index_rot,
                        sigma_index_tor, sigma_index_sc_tor])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch_fix(model, complex_graphs, device, t_to_sigma, args):
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    if args.asyncronous_noise_schedule:
        tr_schedule = get_inverse_schedule(t_schedule, args.sampling_alpha, args.sampling_beta)
        rot_schedule = get_inverse_schedule(t_schedule, args.rot_alpha, args.rot_beta)
        tor_schedule = get_inverse_schedule(t_schedule, args.tor_alpha, args.tor_beta)
        sidechain_tor_schedule = get_inverse_schedule(t_schedule, args.sidechain_tor_alpha, args.sidechain_tor_beta)
    else:
        tr_schedule, rot_schedule, tor_schedule, sidechain_tor_schedule = t_schedule, t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds = []
    rec_lig_steric_clashes = []
    rec_sc_lig_steric_clashes = []
    rec_sc_rec_rest_steric_clashes = []
    rec_sc_rec_sc_steric_clashes = []
    rec_lig_steric_clashes_delta = []
    rec_sc_lig_steric_clashes_delta = []
    rec_sc_rec_rest_steric_clashes_delta = []
    rec_sc_rec_sc_steric_clashes_delta = []
    sc_rmsds = []
    sc_rmsds_improved = []
    sc_rmsds_random = []
    sc_rmsds_from_holo = []
    sc_rmsds_from_holo_ratio = []
    sc_rmsds_improved_from_holo = []
    sc_rmsds_random_from_holo = []
    sc_rmsds_from_holo_max = []

    for orig_complex_graph in tqdm(loader):
        # Number of steric clashes the original complex has
        if args.all_atoms:
            base_rec_lig_steric_clash = get_steric_clash_atom_pairs(orig_complex_graph["atom"].pos[None, :],
                                                                      orig_complex_graph["ligand"].pos[None, :],
                                                                      get_rec_elements(orig_complex_graph),
                                                                      get_ligand_elements(orig_complex_graph)).sum()

        if args.flexible_sidechains:
            flexidx = torch.unique(orig_complex_graph['flexResidues'].subcomponents).cpu().numpy()
            filterSCHs = flexidx[torch.not_equal(orig_complex_graph['atom'].x[flexidx, 0], 0).cpu().numpy()]

            base_rec_sc_lig_steric_clash = get_steric_clash_atom_pairs(orig_complex_graph["atom"].pos[None, :],
                                                                       orig_complex_graph["ligand"].pos[None, :],
                                                                       get_rec_elements(orig_complex_graph),
                                                                       get_ligand_elements(orig_complex_graph), filter1=filterSCHs).sum()

            base_rec_sc_rec_rest_steric_clash = get_steric_clash_per_flexble_sidechain_atom(orig_complex_graph)
            base_rec_sc_rec_sc_steric_clash = get_steric_clash_per_flexble_sidechain_atom(orig_complex_graph, rec_rest=False)

        data_list = [copy.deepcopy(orig_complex_graph)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max,
                           pocket_knowledge=args.inf_pocket_knowledge, pocket_cutoff=args.inf_pocket_cutoff, flexible_sidechains=args.flexible_sidechains)

        if args.flexible_sidechains:
            sc_rmsd_random = RMSD(filterSCHs, data_list[0]['atom'].pos.cpu().numpy(), orig_complex_graph['atom'].pos.cpu().numpy())
            if args.compare_true_protein:
                sc_rmsd_random_from_holo = RMSD(filterSCHs, data_list[0]['atom'].pos.cpu().numpy(), orig_complex_graph['flexResidues'].true_sc_pos.cpu().numpy())

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type == 'cuda' else model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         sidechain_tor_schedule=sidechain_tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args,
                                                         asyncronous_noise_schedule=args.asyncronous_noise_schedule,
                                                         t_schedule=t_schedule)
            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        if failed_convergence_counter > 5: continue
        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph[
                                                         'ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        orig_ligand_pos = np.expand_dims(
            orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
        rmsds.append(rmsd)

        if args.all_atoms:
            rec_lig_steric_clash = get_steric_clash_atom_pairs(predictions_list[0]["atom"].pos[None, :],
                                                                 predictions_list[0]["ligand"].pos[None, :],
                                                                 get_rec_elements(predictions_list[0]),
                                                                 get_ligand_elements(predictions_list[0])).sum()

            rec_lig_steric_clashes.append(rec_lig_steric_clash)
            rec_lig_steric_clashes_delta.append(rec_lig_steric_clash - base_rec_lig_steric_clash)

        if args.flexible_sidechains:
            rec_sc_lig_steric_clash = get_steric_clash_atom_pairs(predictions_list[0]["atom"].pos[None, :],
                                                                  predictions_list[0]["ligand"].pos[None, :],
                                                                  get_rec_elements(predictions_list[0]),
                                                                  get_ligand_elements(predictions_list[0]),
                                                                  filter1=filterSCHs).sum()

            rec_sc_lig_steric_clashes.append(rec_sc_lig_steric_clash)
            rec_sc_lig_steric_clashes_delta.append(rec_sc_lig_steric_clash - base_rec_sc_lig_steric_clash)

            rec_sc_rec_rest_steric_clash = get_steric_clash_per_flexble_sidechain_atom(predictions_list[0])
            rec_sc_rec_sc_steric_clash = get_steric_clash_per_flexble_sidechain_atom(predictions_list[0], rec_rest=False)

            rec_sc_rec_rest_steric_clashes.append(rec_sc_rec_rest_steric_clash)
            rec_sc_rec_rest_steric_clashes_delta.append(rec_sc_rec_rest_steric_clash - base_rec_sc_rec_rest_steric_clash)
            
            rec_sc_rec_sc_steric_clashes.append(rec_sc_rec_sc_steric_clash)
            rec_sc_rec_sc_steric_clashes_delta.append(rec_sc_rec_sc_steric_clash - base_rec_sc_rec_sc_steric_clash)


            sc_rmsd = RMSD(filterSCHs, predictions_list[0]['atom'].pos.cpu().numpy(), orig_complex_graph['atom'].pos.cpu().numpy())
            sc_rmsds.append(sc_rmsd)
            sc_rmsds_random.append(sc_rmsd_random)
            sc_rmsds_improved.append(sc_rmsd_random - sc_rmsd)
            if args.compare_true_protein:
                sc_rmsds_random_from_holo.append(sc_rmsd_random_from_holo)

                sc_rmsd_from_holo = RMSD(filterSCHs, predictions_list[0]['atom'].pos.cpu().numpy(), orig_complex_graph['flexResidues'].true_sc_pos.cpu().numpy())
                sc_rmsd_from_holo_orig = RMSD(filterSCHs, orig_complex_graph['atom'].pos.cpu().numpy(), orig_complex_graph['flexResidues'].true_sc_pos.cpu().numpy())
                sc_rmsd_from_holo_ratio = sc_rmsd_from_holo / sc_rmsd_from_holo_orig
                sc_rmsds_from_holo.append(sc_rmsd_from_holo)
                sc_rmsds_from_holo_ratio.append(sc_rmsd_from_holo_ratio)
                sc_rmsds_improved_from_holo.append(sc_rmsd_random_from_holo - sc_rmsd_from_holo)
                sc_rmsds_from_holo_max.append(abs(sc_rmsd_from_holo-sc_rmsd_from_holo_orig))

    rmsds = np.array(rmsds)

    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))}

    if args.all_atoms:
        losses['rec_lig_steric_clashes'] = np.array(rec_lig_steric_clashes).mean()
        losses['rec_lig_steric_clashes_delta'] = np.array(rec_lig_steric_clashes_delta).mean()
        losses['rec_lig_steric_clash_percentage'] = 100*(np.array(rec_lig_steric_clashes)>0).mean()

    if args.flexible_sidechains:
        losses['rec_sc_lig_steric_clashes'] = np.array(rec_sc_lig_steric_clashes).mean()
        losses['rec_sc_rec_rest_steric_clashes'] = np.array(rec_sc_rec_rest_steric_clashes).mean()
        losses['rec_sc_rec_sc_steric_clashes'] = np.array(rec_sc_rec_sc_steric_clashes).mean()
        losses['rec_sc_lig_steric_clashes_delta'] = np.array(rec_sc_lig_steric_clashes_delta).mean()
        losses['rec_sc_rec_rest_steric_clashes_delta'] = np.array(rec_sc_rec_rest_steric_clashes_delta).mean()
        losses['rec_sc_rec_sc_steric_clashes_delta'] = np.array(rec_sc_rec_sc_steric_clashes_delta).mean()
        losses['rec_sc_lig_steric_clash_percentage'] = 100*(np.array(rec_sc_lig_steric_clashes)>0).mean()
        losses['rec_sc_rec_sc_steric_clash_percentage'] = 100*(np.array(rec_sc_rec_sc_steric_clashes)>0).mean()

        sc_rmsds = np.array(sc_rmsds)
        sc_rmsds_improved = np.array(sc_rmsds_improved)
        sc_rmsds_random = np.array(sc_rmsds_random)

        # TODO: this can be removed, it just serves as a way to find nice values for the confidence model
        losses['rmsds_lt2_and_sc_rmsds_lt01'] = 100 * ((rmsds < 2).squeeze() & (sc_rmsds < 0.1)).sum() / len(sc_rmsds)
        losses['rmsds_lt2_and_sc_rmsds_lt025'] = 100 * ((rmsds < 2).squeeze() & (sc_rmsds < 0.25)).sum() / len(sc_rmsds)
        losses['rmsds_lt2_and_sc_rmsds_lt05'] = 100 * ((rmsds < 2).squeeze() & (sc_rmsds < 0.5)).sum() / len(sc_rmsds)
        losses['rmsds_lt2_and_sc_rmsds_lt1'] = 100 * ((rmsds < 2).squeeze() & (sc_rmsds < 1)).sum() / len(sc_rmsds)
        losses['rmsds_lt2_and_sc_rmsds_lt2'] = 100 * ((rmsds < 2).squeeze() & (sc_rmsds < 2)).sum() / len(sc_rmsds)
        losses['rmsds_lt2_and_sc_rmsds_lt3'] = 100 * ((rmsds < 2).squeeze() & (sc_rmsds < 3)).sum() / len(sc_rmsds)

        losses['rmsds_lt3_and_sc_rmsds_lt025'] = 100 * ((rmsds < 3).squeeze() & (sc_rmsds < 0.25)).sum() / len(sc_rmsds)
        losses['rmsds_lt3_and_sc_rmsds_lt05'] = 100 * ((rmsds < 3).squeeze() & (sc_rmsds < 0.5)).sum() / len(sc_rmsds)

        losses['sc_rmsds_lt1'] = 100 * (sc_rmsds < 1).sum() / len(sc_rmsds)
        losses['sc_rmsds_lt2'] = 100 * (sc_rmsds < 2).sum() / len(sc_rmsds)
        losses['sc_rmsds_lt05'] = 100 * (sc_rmsds < 0.5).sum() / len(sc_rmsds)
        losses['sc_rmsds_lt025'] = 100 * (sc_rmsds < 0.25).sum() / len(sc_rmsds)
        losses['sc_rmsds_lt01'] = 100 * (sc_rmsds < 0.1).sum() / len(sc_rmsds)
        losses['sc_rmsds_avg_improvement'] = 100 * ((np.divide(sc_rmsds_improved[sc_rmsds_improved > 0], sc_rmsds_random[sc_rmsds_improved > 0])).sum() / len(sc_rmsds_improved[sc_rmsds_improved > 0]) if (sc_rmsds_improved > 0).sum() != 0 else 0)
        losses['sc_rmsds_avg_worsening'] = -100 * ((np.divide(sc_rmsds_improved[sc_rmsds_improved < 0], sc_rmsds_random[sc_rmsds_improved < 0])).sum() / len(sc_rmsds_improved[sc_rmsds_improved < 0]) if (sc_rmsds_improved < 0).sum() != 0 else 0)
        if args.compare_true_protein:
            sc_rmsds_from_holo = np.array(sc_rmsds_from_holo)
            sc_rmsds_from_holo_ratio = np.array(sc_rmsds_from_holo_ratio)
            sc_rmsds_improved_from_holo = np.array(sc_rmsds_improved_from_holo)
            sc_rmsds_random_from_holo = np.array(sc_rmsds_random_from_holo)
            sc_rmsds_from_holo_max = np.array(sc_rmsds_from_holo_max)

            losses['sc_rmsds_lt1_from_holo'] = 100 * (sc_rmsds_from_holo < 1).sum() / len(sc_rmsds_from_holo)
            losses['sc_rmsds_lt2_from_holo'] = 100 * (sc_rmsds_from_holo < 2).sum() / len(sc_rmsds_from_holo)
            losses['sc_rmsds_lt05_from_holo'] = 100 * (sc_rmsds_from_holo < 0.5).sum() / len(sc_rmsds_from_holo)
            losses['sc_rmsds_avg_improvement_from_holo'] = 100 * ((np.divide(sc_rmsds_improved_from_holo[sc_rmsds_improved_from_holo > 0], sc_rmsds_random_from_holo[sc_rmsds_improved_from_holo > 0])).sum() / len(sc_rmsds_improved_from_holo[sc_rmsds_improved_from_holo > 0]) if (sc_rmsds_improved_from_holo > 0).sum() != 0 else 0)
            losses['sc_rmsds_avg_worsening_from_holo'] = -100 * ((np.divide(sc_rmsds_improved_from_holo[sc_rmsds_improved_from_holo < 0], sc_rmsds_random_from_holo[sc_rmsds_improved_from_holo < 0])).sum() / len(sc_rmsds_improved_from_holo[sc_rmsds_improved_from_holo < 0]) if (sc_rmsds_improved_from_holo < 0).sum() != 0 else 0)
            losses['sc_rmsds_best_lt1_from_holo'] = 100 * (sc_rmsds_from_holo_max < 1).sum() / len(sc_rmsds_from_holo_max)
            losses['sc_rmsds_best_lt2_from_holo'] = 100 * (sc_rmsds_from_holo_max < 2).sum() / len(sc_rmsds_from_holo_max)
            losses['sc_rmsds_best_lt05_from_holo'] = 100 * (sc_rmsds_from_holo_max < 0.5).sum() / len(sc_rmsds_from_holo_max)

    return losses
