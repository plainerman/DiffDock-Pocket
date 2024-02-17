import copy
import gc
import math
import os

import shutil

from argparse import Namespace, ArgumentParser, FileType

import numpy as np
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    print(f"wandb not installed, cannot use wandb features")

import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm

from filtering.dataset import FilteringDataset
from utils.training import AverageMeter

torch.multiprocessing.set_sharing_strategy('file_system')

import yaml
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model


parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--original_model_dir', type=str, default='workdir', help='Path to folder with trained model and hyperparameters')
parser.add_argument('--restart_dir', type=str, default=None, help='')
parser.add_argument('--use_original_model_cache', action='store_true', default=False, help='If this is true, the same dataset as in the original model will be used. Otherwise, the dataset parameters are used.')
parser.add_argument('--data_dir', type=str, default='data/PDBBIND_atomCorrected/', help='Folder containing original structures')
parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--model_save_frequency', type=int, default=0, help='Frequency with which to save the last model. If 0, then only the early stopping criterion best model is saved and overwritten.')
parser.add_argument('--best_model_save_frequency', type=int, default=0, help='Frequency with which to save the best model. If 0, then only the early stopping criterion best model is saved and overwritten.')
parser.add_argument('--run_name', type=str, default='test_filtering', help='')
parser.add_argument('--project', type=str, default='ligbind_filtering', help='')
parser.add_argument('--split_train', type=str, default='data/splits/timesplit_no_lig_overlap_train', help='Path of file defining the split')
parser.add_argument('--split_val', type=str, default='data/splits/timesplit_no_lig_overlap_val', help='Path of file defining the split')
parser.add_argument('--split_test', type=str, default='data/splits/timesplit_test', help='Path of file defining the split')
parser.add_argument('--dataloader_drop_last', action='store_true', default=False, help='drop_last arg of dataloader')

# Inference parameters for creating the positions and rmsds that the confidence predictor will be trained on.
parser.add_argument('--cache_path', type=str, default='data/cacheNew', help='Folder from where to load/restore cached dataset')
parser.add_argument('--cache_ids_to_combine', nargs='+', type=int, default=None, help='number of caches that will be combined to form the training set of the confidence model.')
parser.add_argument('--cache_creation_id', type=int, default=None, help='number of times that inference is run on the full dataset before concatenating it and coming up with the full filtering dataset')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--inference_steps', type=int, default=2, help='Number of denoising steps')
parser.add_argument('--samples_per_complex', type=int, default=3, help='')
parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='')
parser.add_argument('--inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
parser.add_argument('--inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
parser.add_argument('--different_schedules', action='store_true', default=False, help='')
parser.add_argument('--rot_sigma_schedule', type=str, default='expbeta', help='')
parser.add_argument('--rot_inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
parser.add_argument('--rot_inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
parser.add_argument('--tor_sigma_schedule', type=str, default='expbeta', help='')
parser.add_argument('--tor_inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
parser.add_argument('--tor_inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
parser.add_argument('--balance', action='store_true', default=False, help='If this is true than we do not force the samples seen during training to be the same amount of negatives as positives')
parser.add_argument('--rmsd_prediction', action='store_true', default=False, help='If this is true than we do not force the samples seen during training to be the same amount of negatives as positives')
parser.add_argument('--rmsd_classification_cutoff', nargs='+', type=float, default=2, help='Ligand RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
parser.add_argument('--sc_rmsd_classification_cutoff', nargs='+', type=float, default=1, help='Sidechain RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
parser.add_argument('--affinity_prediction', action='store_true', default=False, help='Whether to jointly predict affinity')
parser.add_argument('--confidence_loss_weigth', type=float, default=1, help='Weight parameter for the confidence loss')


parser.add_argument('--log_dir', type=str, default='workdir', help='')
parser.add_argument('--main_metric', type=str, default='accuracy', help='Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]')
parser.add_argument('--main_metric_goal', type=str, default='max', help='Can be [min, max]')
parser.add_argument('--transfer_weights', action='store_true', default=False, help='')
parser.add_argument('--batch_size', type=int, default=5, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--w_decay', type=float, default=0.0, help='')
parser.add_argument('--scheduler', type=str, default='plateau', help='')
parser.add_argument('--scheduler_patience', type=int, default=20, help='')
parser.add_argument('--n_epochs', type=int, default=5, help='')

# Dataset
parser.add_argument('--use_old_wrong_embedding_order', action='store_true', default=False,
                    help='for backward compatibility to prevent the chain embedding order')
parser.add_argument('--limit_complexes', type=int, default=0, help='')
parser.add_argument('--all_atoms', action='store_true', default=False, help='')
parser.add_argument('--multiplicity', type=int, default=1, help='')
parser.add_argument('--chain_cutoff', type=float, default=10, help='')
parser.add_argument('--receptor_radius', type=float, default=30, help='')
parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='')
parser.add_argument('--atom_max_neighbors', type=int, default=8, help='')
parser.add_argument('--matching_popsize', type=int, default=20, help='')
parser.add_argument('--matching_maxiter', type=int, default=20, help='')
parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms')
parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
parser.add_argument('--num_conformers', type=int, default=1, help='')
parser.add_argument('--lm_embeddings_path', type=str, default=None, help='If this is set then the LM embeddings at that path will be used for the receptor features')
parser.add_argument('--esm_embeddings_path', type=str, default=None,help='If this is set then the LM embeddings at that path will be used for the receptor features')
parser.add_argument('--not_full_dataset', action='store_true', default=False, help='')
parser.add_argument('--not_fixed_knn_radius_graph', action='store_true', default=False, help='')
parser.add_argument('--not_knn_only_graph', action='store_true', default=False, help='')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
parser.add_argument('--pocket_reduction', action='store_true', default=False,
                    help='Remove atoms from receptor that are not in the binding pocket')
parser.add_argument('--pocket_buffer', type=float, default=10,
                    help='Buffer that will be added to the radius of the pocket')

# Diffusion
parser.add_argument('--rot_sigmoid_schedule', action='store_true', default=False, help='')
parser.add_argument('--schedule_k', type=float, default=10, help='')
parser.add_argument('--schedule_m', type=float, default=0.4, help='')
parser.add_argument('--train_sampling', type=str, default='linear', help='')
parser.add_argument('--no_torsion', action='store_true', default=False, help='')
parser.add_argument('--separate_noise_schedule', action='store_true', default=False, help='Use different t for tr, rot, and tor')
parser.add_argument('--asyncronous_noise_schedule', action='store_true', default=False, help='')
parser.add_argument('--correct_torsion_sigmas', action='store_true', default=True, help='We had a bug initially and this parameter is there for backward cmopatibility.')
parser.add_argument('--sampling_alpha', type=float, default=1, help='Alpha parameter of beta distribution for sampling t')
parser.add_argument('--sampling_beta', type=float, default=1, help='Beta parameter of beta distribution for sampling t')
parser.add_argument('--rot_alpha', type=float, default=1,help='Alpha parameter of beta distribution for sampling t')
parser.add_argument('--rot_beta', type=float, default=1,help='Beta parameter of beta distribution for sampling t')
parser.add_argument('--tor_alpha', type=float, default=1,help='Alpha parameter of beta distribution for sampling t')
parser.add_argument('--tor_beta', type=float, default=1,help='Beta parameter of beta distribution for sampling t')

# Model
parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
parser.add_argument('--norm_by_sigma', action='store_true', default=False, help='Whether to normalise the score')
parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
parser.add_argument('--distance_embed_dim', type=int, default=32, help='')
parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='')
parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
parser.add_argument('--cross_max_distance', type=float, default=80, help='')
parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='')
parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
parser.add_argument('--smooth_edges', action='store_true', default=False)
parser.add_argument('--odd_parity', action='store_true', default=False)
parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='')
parser.add_argument('--sigma_embed_dim', type=int, default=32, help='')
parser.add_argument('--embedding_scale', type=int, default=10000, help='')
parser.add_argument('--tp_attention', action='store_true', default=False)
parser.add_argument('--parallel', type=int, default=1, help='')
parser.add_argument('--parallel_aggregators', type=str, default="mean max min std", help='')
parser.add_argument('--sh_lmax', type=int, default=2, help='Size of the embedding of the diffusion time')
parser.add_argument('--adamw', action='store_true', default=False, help='Use AdamW optimizer instead of Adam')

# Confidence Predictor in Model
parser.add_argument('--include_confidence_prediction', action='store_true', default=False,help='Whether to predict an additional confidence metric for each predicted structure')
parser.add_argument('--high_confidence_threshold', type=float, default=5.0,help='If this is 0 then the confidence predictor tries to predict the centroid_distance. Otherwise it is the Ångström below which a prediction is labeled as good for supervising the confidence predictor')
parser.add_argument('--tr_only_confidence', action='store_true', default=True, help='Whether to only supervise the confidence predictor with the translation')
parser.add_argument('--confidence_no_batchnorm', action='store_true', default=False, help='')
parser.add_argument('--confidence_dropout', type=float, default=0.0, help='MLP dropout in confidence readout')
parser.add_argument('--protein_file', type=str, default='protein_processed', help='')
parser.add_argument('--trajectory_training', action='store_true', default=False, help='')

parser.add_argument('--no_train', action='store_true', default=False, help='')
parser.add_argument('--test', action='store_true', default=False, help='')
parser.add_argument('--multiplicity_test', type=int, default=1, help='')

args = parser.parse_args()
if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value
    args.config = args.config.name
assert(args.main_metric_goal == 'max' or args.main_metric_goal == 'min')

if isinstance(args.rmsd_classification_cutoff, list) and len(args.rmsd_classification_cutoff) == 1:
    args.rmsd_classification_cutoff = args.rmsd_classification_cutoff[0]

if isinstance(args.sc_rmsd_classification_cutoff, list) and len(args.sc_rmsd_classification_cutoff) == 1:
    args.sc_rmsd_classification_cutoff = args.sc_rmsd_classification_cutoff[0]


def train_epoch(model, loader, optimizer, rmsd_prediction, affinity_prediction, affinity_loss_weight = 1, confidence_loss_weigth=1, parallel=1):
    model.train()
    meter = AverageMeter(['loss', 'confidence_loss', 'affinity_loss'])

    for data in tqdm(loader, total=len(loader)):
        if device.type == 'cuda' and len(data) % torch.cuda.device_count() == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            pred = model(data)
            affinity_loss = torch.tensor(0.0, dtype=torch.float, device=pred[0].device)
            if affinity_prediction:
                if model.flexible_sidechains:
                    raise NotImplementedError("Affinity prediction not implemented for flexible sidechains")
                if parallel > 1:
                    affinity_pred = pred[1]
                    pred = pred[0]
                    affinity_labels = torch.tensor([graph.affinity for graph in data], device=device) if isinstance(data, list) else data.affinity
                    affinity_loss += F.mse_loss(affinity_labels, affinity_pred)
                else:
                    affinity_pred = pred[:, -1]
                    pred = pred[:,:-1]
                    affinity_labels = torch.tensor([graph.affinity for graph in data], device=device) if isinstance(data,list) else data.affinity
                    rmsds_below_thresh = (torch.tensor([graph.y for graph in data], device=device) if isinstance(data, list) else data.y).bool()
                    affinity_pred = affinity_pred[rmsds_below_thresh]
                    affinity_labels = affinity_labels[rmsds_below_thresh]
                    if torch.any(rmsds_below_thresh): affinity_loss += F.mse_loss(affinity_labels, affinity_pred)
            if rmsd_prediction:
                if model.flexible_sidechains:
                    raise NotImplementedError("RMSD prediction not implemented for flexible sidechains")

                labels = torch.cat([graph.rmsd for graph in data]).to(device) if isinstance(data, list) else data.rmsd
                confidence_loss = F.mse_loss(pred, labels)
            else:
                if isinstance(args.rmsd_classification_cutoff, list):
                    labels = torch.cat([graph.y_binned for graph in data]).to(device) if isinstance(data, list) else data.y_binned
                    confidence_loss = F.cross_entropy(pred, labels)
                else:
                    labels = torch.cat([graph.y for graph in data]).to(device) if isinstance(data, list) else data.y
                    confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
            loss = confidence_loss * confidence_loss_weigth + affinity_loss_weight * affinity_loss
            loss.backward()
            optimizer.step()
            meter.add([loss.cpu().detach(), confidence_loss.cpu().detach(), affinity_loss.cpu().detach()])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

    return meter.summary()

def test_epoch(model, loader, rmsd_prediction, affinity_prediction, confidence_loss_weigth=1, affinity_loss_weight=1,
               parallel=1, trajectory_training=False, multiplicity=1, return_pred_labels=False):
    model.eval()
    meter = AverageMeter(['loss'], unpooled_metrics=True) if rmsd_prediction else AverageMeter(['loss', 'confidence_loss', 'accuracy', 'ROC AUC', 'affinity_loss', 'affinity_rmse'], unpooled_metrics=True)

    if trajectory_training:
        meter_all = AverageMeter(['confidence_loss', 'accuracy'], unpooled_metrics=True, intervals=21)

    all_labels = []
    all_pred = []
    all_affinities = []
    for sample in range(multiplicity):
        loader.dataset.fixed_sample = sample
        for data in tqdm(loader, total=len(loader)):
            try:
                with torch.no_grad():
                    pred = model(data)
                affinity_loss = torch.tensor(0.0, dtype=torch.float, device=pred[0].device)
                accuracy = torch.tensor(0.0, dtype=torch.float, device=pred[0].device)
                if affinity_prediction:
                    if parallel > 1:
                        affinity_pred = pred[1]
                        pred = pred[0]
                        affinity_labels = torch.tensor([graph.affinity for graph in data], device=device) if isinstance(data, list) else data.affinity
                        affinity_loss += F.mse_loss(affinity_labels, affinity_pred)
                    else:
                        affinity_pred = pred[:, 1]
                        pred = pred[:, 0]
                        affinity_labels = torch.tensor([graph.affinity for graph in data], device=device) if isinstance(data,list) else data.affinity
                        rmsds_below_thresh = (torch.tensor([graph.y for graph in data], device=device) if isinstance(data,list) else data.y).bool()
                        affinity_pred = affinity_pred[rmsds_below_thresh]
                        affinity_labels = affinity_labels[rmsds_below_thresh]
                        if torch.any(rmsds_below_thresh): affinity_loss += F.mse_loss(affinity_labels, affinity_pred)
                if rmsd_prediction:
                    labels = torch.cat([graph.rmsd for graph in data]).to(device) if isinstance(data, list) else data.rmsd
                    confidence_loss = F.mse_loss(pred, labels)
                    meter.add([confidence_loss.cpu().detach()])
                else:
                    if isinstance(args.rmsd_classification_cutoff, list):
                        labels = torch.cat([graph.y_binned for graph in data]).to(device) if isinstance(data,list) else data.y_binned
                        confidence_loss = F.cross_entropy(pred, labels)
                    else:
                        labels = torch.cat([graph.y for graph in data]).to(device) if isinstance(data, list) else data.y
                        confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
                        accuracy = torch.mean((labels == (pred > 0).float()).float())
                    try:
                        roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
                    except ValueError as e:
                        if 'Only one class present in y_true. ROC AUC score is not defined in that case.' in str(e):
                            roc_auc = 0
                        else:
                            raise e
                loss = confidence_loss * confidence_loss_weigth + affinity_loss_weight * affinity_loss
                meter.add([loss.cpu().detach(), confidence_loss.cpu().detach(), accuracy.cpu().detach(), torch.tensor(roc_auc), affinity_loss.cpu().detach(), torch.sqrt(affinity_loss.cpu().detach())])

                if trajectory_training:
                    t_index = (torch.cat([d.complex_t['tr'] for d in data]) * 20).long().cpu()
                    assert not rmsd_prediction and not isinstance(args.rmsd_classification_cutoff, list), "not implemented"
                    confidence_loss = F.binary_cross_entropy_with_logits(pred, labels, reduction='none')
                    accuracy = (labels == (pred > 0).float()).float()
                    meter_all.add([confidence_loss.cpu().detach(), accuracy.cpu().detach()], [t_index, t_index])

                all_labels.append(labels)
                all_pred.append(pred.detach().cpu())

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)

    if rmsd_prediction:
        baseline_metric = ((all_labels - all_labels.mean()).abs()).mean()
    else:
        baseline_metric = all_labels.sum() / len(all_labels)
    results = meter.summary()
    results.update({'baseline_metric': baseline_metric})
    if len(all_affinities) == 0:
        results.update({'affinity_mean_mse': torch.tensor(0)})
    else:
        results.update({'affinity_mean_mse': ((all_affinities - all_affinities.mean())**2).mean()})
    out = meter.summary()
    if trajectory_training: out.update(meter_all.summary())
    if return_pred_labels: return out, baseline_metric, all_pred, all_labels
    return out, baseline_metric


def train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir):
    best_val_metric = math.inf if args.main_metric_goal == 'min' else 0
    best_epoch = 0

    print("Starting training...")
    for epoch in range(args.n_epochs):
        logs = {}
        train_metrics = train_epoch(model, train_loader, optimizer, args.rmsd_prediction, args.affinity_prediction, confidence_loss_weigth=args.confidence_loss_weigth, parallel=args.parallel)
        print("Epoch {}: Training loss {:.4f}".format(epoch, train_metrics['loss']))

        val_metrics, baseline_metric = test_epoch(model, val_loader, args.rmsd_prediction, args.affinity_prediction,
                                                  confidence_loss_weigth=args.confidence_loss_weigth, parallel=args.parallel,
                                                  trajectory_training=args.trajectory_training)
        if args.rmsd_prediction:
            print("Epoch {}: Validation loss {:.4f}".format(epoch, val_metrics['loss']))
        else:
            print("Epoch {}: Validation loss {:.4f}  accuracy {:.4f}".format(epoch, val_metrics['loss'], val_metrics['accuracy']))

        if args.wandb:
            logs.update({'valinf_' + k: v for k, v in val_metrics.items()}, step=epoch + 1)
            logs.update({'train_' + k: v for k, v in train_metrics.items()}, step=epoch + 1)
            logs.update({'mean_rmsd' if args.rmsd_prediction else 'fraction_positives': baseline_metric,
                         'current_lr': optimizer.param_groups[0]['lr']})
            wandb.log(logs, step=epoch + 1)

        if scheduler:
            scheduler.step(val_metrics[args.main_metric])

        state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()

        if args.main_metric_goal == 'min' and val_metrics[args.main_metric] < best_val_metric or args.main_metric_goal == 'max' and val_metrics[args.main_metric] > best_val_metric:
            best_val_metric = val_metrics[args.main_metric]
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
        if args.model_save_frequency > 0 and (epoch + 1) % args.model_save_frequency == 0:
            torch.save(state_dict, os.path.join(run_dir, f'model_epoch{epoch+1}.pt'))
        if args.best_model_save_frequency > 0 and (epoch + 1) % args.best_model_save_frequency == 0:
            shutil.copyfile(os.path.join(run_dir, 'best_model.pt'), os.path.join(run_dir, f'best_model_epoch{epoch+1}.pt'))

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation accuracy {} on Epoch {}".format(best_val_metric, best_epoch))


def test(args, model, val_loader, run_dir, multiplicity=1):

    print("Starting testing...")
    losses, accuracies, preds_list, labels_list = [], [], [], []
    for step in range(21):
        val_loader.dataset.fixed_step = step
        val_metrics, baseline_metric, preds, labels = test_epoch(model, val_loader, args.rmsd_prediction, args.affinity_prediction,
                                                  confidence_loss_weigth=args.confidence_loss_weigth, parallel=args.parallel,
                                                  trajectory_training=False, multiplicity=multiplicity, return_pred_labels=True)
        print("Step {}: Validation loss {:.4f}  accuracy {:.4f}".format(step, val_metrics['loss'], val_metrics['accuracy']))

        losses.append(val_metrics['loss'])
        accuracies.append(val_metrics['accuracy'])
        preds_list.append(preds)
        labels_list.append(labels)

        torch.save({
            'loss': val_metrics['loss'],
            'accuracy': val_metrics['accuracy'],
            'predictions': preds,
            'labels': labels
        }, os.path.join(run_dir, f'val_eval_step{step}.pt'))


    losses = torch.from_numpy(np.asarray(losses))
    accuracies = torch.from_numpy(np.asarray(accuracies))
    preds_list = torch.cat([p.unsqueeze(0) for p in preds_list])
    labels_list = torch.cat([p.unsqueeze(0) for p in labels_list])
    torch.save({
        'loss': losses,
        'accuracy': accuracies,
        'predictions': preds_list,
        'labels': labels_list
    }, os.path.join(run_dir, f'val_eval.pt'))


def construct_loader_filtering(args, device):
    common_args = {'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir, 'device': device,
                   'sigma_schedule': args.sigma_schedule, 'inference_steps': args.inference_steps,
                   'inf_sched_alpha': args.inf_sched_alpha, 'inf_sched_beta': args.inf_sched_beta,
                   'rot_inf_sched_alpha': args.rot_inf_sched_alpha, 'rot_inf_sched_beta': args.rot_inf_sched_beta,
                   'tor_inf_sched_alpha': args.tor_inf_sched_alpha, 'tor_inf_sched_beta': args.tor_inf_sched_beta,
                   'samples_per_complex': args.samples_per_complex, 'different_schedules': args.different_schedules,
                   'limit_complexes': args.limit_complexes, 'all_atoms': args.all_atoms, 'balance': args.balance,
                   'rmsd_classification_cutoff': args.rmsd_classification_cutoff,
                   'sc_rmsd_classification_cutoff': args.sc_rmsd_classification_cutoff,
                   'multiplicity': args.multiplicity, 'use_original_model_cache': args.use_original_model_cache,
                   "parallel": args.parallel,
                   'cache_creation_id': args.cache_creation_id, "cache_ids_to_combine": args.cache_ids_to_combine,
                   "model_ckpt": args.ckpt, "trajectory_sampling": args.trajectory_training}
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader

    exception_flag = False
    try:
        train_dataset = FilteringDataset(split="train", args=args, **common_args)
        train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.dataloader_drop_last)
    except Exception as e:
        if 'The generated ligand positions with cache_id do not exist:' in str(e):
            print("HAPPENING | Encountered the following exception when loading the filtering train dataset:")
            print(str(e))
            print("HAPPENING | We are still continuing because we want to try to generate the validation dataset if it has not been created yet:")
            exception_flag = True
        else: raise e

    val_dataset = FilteringDataset(split="val", args=args, **common_args)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=args.dataloader_drop_last)

    if exception_flag: raise Exception('We encountered the exception during train dataset loading: ', e)
    return train_loader, val_loader


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(f'{args.original_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
        if not hasattr(score_model_args, 'separate_noise_schedule'):  # exists for compatibility
            score_model_args.separate_noise_schedule = False
        if not hasattr(score_model_args, 'lm_embeddings_path'):  # exists for compatibility
            score_model_args.lm_embeddings_path = None
        if not hasattr(score_model_args, 'all_atoms'):  # exists for compatibility
            score_model_args.all_atoms = False
        if not hasattr(score_model_args,'tr_only_confidence'):  # exists for compatibility
            score_model_args.tr_only_confidence = True
        if not hasattr(score_model_args,'high_confidence_threshold'):  # exists for compatibility
            score_model_args.high_confidence_threshold = 0.0
        if not hasattr(score_model_args, 'include_confidence_prediction'):  # exists for compatibility
            score_model_args.include_confidence_prediction = False
        if not hasattr(score_model_args, 'esm_embeddings_path'):  # exists for compatibility
            score_model_args.esm_embeddings_path = None

    # Sidechain configuration is specified by the original model
    args.flexible_sidechains = score_model_args.flexible_sidechains
    args.flexdist = score_model_args.flexdist
    args.flexdist_distance_metric = score_model_args.flexdist_distance_metric

    # construct loader
    train_loader, val_loader = construct_loader_filtering(args, device)
    model = get_model(score_model_args if args.transfer_weights else args, device, t_to_sigma=None, confidence_mode=True)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.main_metric_goal)

    if args.transfer_weights:
        print("HAPPENING | Transferring weights from original_model_dir to the new model after using original_model_dir's arguments to construct the new model.")
        checkpoint = torch.load(os.path.join(args.original_model_dir,args.ckpt), map_location=device)
        model_state_dict = model.state_dict()
        transfer_weights_dict = {k: v for k, v in checkpoint.items() if k in list(model_state_dict.keys())}
        model_state_dict.update(transfer_weights_dict)  # update the layers with the pretrained weights
        model.load_state_dict(model_state_dict)

    elif args.restart_dir:
        dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
        model.module.load_state_dict(dict['model'], strict=True)
        optimizer.load_state_dict(dict['optimizer'])
        print("Restarting from epoch", dict['epoch'])

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')
    run_dir = os.path.join(args.log_dir, args.run_name)

    if not args.no_train:
        if args.wandb:
            wandb.init(
                entity='coarse-graining-mit',
                settings=wandb.Settings(start_method="fork"),
                project=args.project,
                name=args.run_name,
                config=args
            )
            wandb.log({'numel': numel})

        # record parameters
        yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
        save_yaml_file(yaml_file_name, args.__dict__)
        args.device = device

        train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir)

    if args.test:
        test(args, model, val_loader, run_dir, multiplicity=args.multiplicity_test)
