import copy

import numpy as np
import torch
import os

from torch_geometric.loader import DataLoader
from Bio.PDB import PDBIO,PDBParser,Select
from copy import deepcopy
import concurrent.futures

from utils.diffusion_utils import modify_conformer, set_time, modify_sidechains
from utils.torsion import modify_conformer_torsion_angles, get_dihedrals, get_torsion_angles_svgd, get_rigid_svgd
from scipy.spatial.transform import Rotation as R

def randomize_position(data_list, no_torsion, no_random, tr_sigma_max, pocket_knowledge=False, pocket_cutoff=7, flexible_sidechains=False):
    # in place modification of the list
    center_pocket = 0
    if pocket_knowledge:
        complex = data_list[0]
        d = torch.cdist(complex['receptor'].pos, torch.from_numpy(complex['ligand'].orig_pos[0]).float() - complex.original_center)
        label = torch.any(d < pocket_cutoff, dim=1)

        if torch.any(label):
            center_pocket = complex['receptor'].pos[label].mean(dim=0)
        else:
            print("No pocket residue below minimum distance ", pocket_cutoff, "taking closest at", torch.min(d))
            center_pocket = complex['receptor'].pos[torch.argmin(torch.min(d, dim=1)[0])]

    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                    complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0], torsion_updates)
    
    if flexible_sidechains:
        for complex_graph in data_list:
            sidechain_torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=len(complex_graph['flexResidues'].edge_idx))
            modify_sidechains(complex_graph, sidechain_torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T + center_pocket
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update


def is_iterable(arr):
    try:
        some_object_iterator = iter(arr)
        return True
    except TypeError as te:
        return False

def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, sidechain_tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, sidechain_visualization_list=None, confidence_model=None, filtering_data_list=None, filtering_model_args=None,
             asyncronous_noise_schedule=False, t_schedule=None, batch_size=32, no_final_step_noise=False, pivot=None, return_full_trajectory=False,
             svgd_weight=0.0, svgd_repulsive_weight=1.0, svgd_only=False, svgd_rot_rel_weight=1.0, svgd_tor_rel_weight=1.0, svgd_sidechain_tor_rel_weight = 1.0,
             temp_sampling=1.0, temp_psi=0.0, temp_sigma_data=0.5, flexible_sidechains=None):

    flexible_sidechains = model_args.flexible_sidechains if flexible_sidechains is None else flexible_sidechains

    if flexible_sidechains:
        # If in the whole batch there are no flexible residues, we have to delete the sidechain information, so that the loader does not break
        no_sidechains_in_batch = sum([len(c["flexResidues"].subcomponents) for c in data_list]) == 0
        if no_sidechains_in_batch:
            data_list = copy.deepcopy(data_list)
            for c in data_list:
                del c["flexResidues"]

    N = len(data_list)
    trajectory = []
    sidechain_trajectory = []

    if svgd_weight > 0:
        dihedrals = get_dihedrals(data_list)

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor, t_sidechain_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx], sidechain_tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]
        dt_sidechain_tor = sidechain_tor_schedule[t_idx] - sidechain_tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else sidechain_tor_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        
        if return_full_trajectory:
            trajectory.append(np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in data_list]))
            if no_sidechains_in_batch:
                sidechain_trajectory.append(np.asarray([]))
            else:
                sidechain_trajectory.append(np.asarray([complex_graph['atom'].pos.cpu().numpy()[complex_graph['flexResidues'].subcomponents.unique().cpu().numpy()] for complex_graph in data_list]))

        tr_score_list, rot_score_list, tor_score_list, sidechain_tor_score_list = [], [], [], []
        tr_sigma, rot_sigma, tor_sigma, sidechain_tor_sigma = t_to_sigma(t_tr, t_rot, t_tor, t_sidechain_tor)

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            set_time(complex_graph_batch, t_schedule[t_idx] if t_schedule is not None else None, t_tr, t_rot, t_tor, t_sidechain_tor, b,
                     'all_atoms' in model_args and model_args.all_atoms, asyncronous_noise_schedule, device, include_miscellaneous_atoms=hasattr(model_args, 'include_miscellaneous_atoms') and model_args.include_miscellaneous_atoms)
            
            with torch.no_grad():
                tr_score, rot_score, tor_score, sidechain_tor_score = model(complex_graph_batch)

            tr_score_list.append(tr_score.cpu())
            rot_score_list.append(rot_score.cpu())
            tor_score_list.append(tor_score.cpu())
            sidechain_tor_score_list.append(sidechain_tor_score.cpu())

        tr_score, rot_score, tor_score, sidechain_tor_score = torch.cat(tr_score_list, dim=0), torch.cat(rot_score_list, dim=0), torch.cat(tor_score_list, dim=0), torch.cat(sidechain_tor_score_list, dim=0)

        tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
        rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

        if ode:
            tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
            rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
        else:
            tr_z = torch.zeros((N, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                else torch.normal(mean=0, std=1, size=(N, 3))
            tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

            rot_z = torch.zeros((N, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                else torch.normal(mean=0, std=1, size=(N, 3))
            rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

        if not model_args.no_torsion:
            tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
            if ode:
                tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
            else:
                tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=tor_score.shape)
                tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
            torsions_per_molecule = tor_perturb.shape[0] // N
        else:
            tor_perturb = None

        if flexible_sidechains:
            sidechain_tor_g = sidechain_tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.sidechain_tor_sigma_max / model_args.sidechain_tor_sigma_min)))
            if ode:
                sidechain_tor_perturb = (0.5 * sidechain_tor_g ** 2 * dt_sidechain_tor * sidechain_tor_score.cpu()).numpy()
            else:
                sidechain_tor_z = torch.zeros(sidechain_tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=sidechain_tor_score.shape)
                sidechain_tor_perturb = (sidechain_tor_g ** 2 * dt_sidechain_tor * sidechain_tor_score.cpu() + sidechain_tor_g * np.sqrt(dt_sidechain_tor) * sidechain_tor_z).numpy()
            sidechain_torsions_per_molecule = sidechain_tor_perturb.shape[0] // N
        else:
            sidechain_tor_perturb = None

        if not is_iterable(temp_sampling):
            temp_sampling = [temp_sampling] * 4
        if not is_iterable(temp_psi):
            temp_psi = [temp_psi] * 4

        assert len(temp_sampling) == 4
        assert len(temp_psi) == 4
        assert len(temp_psi) == len(temp_sampling)

        if temp_sampling[0] != 1.0:
            tr_sigma_data = np.exp(temp_sigma_data * np.log(model_args.tr_sigma_max) + (1-temp_sigma_data) * np.log(model_args.tr_sigma_min))
            lambda_tr = (tr_sigma_data + tr_sigma) / (tr_sigma_data + tr_sigma/temp_sampling[0])
            tr_perturb = (tr_g ** 2 * dt_tr * (lambda_tr + temp_sampling[0] * temp_psi[0] / 2) * tr_score.cpu() + tr_g * np.sqrt(dt_tr * (1 + temp_psi[0])) * tr_z).cpu()

        if temp_sampling[1] != 1.0:
            rot_sigma_data = np.exp(temp_sigma_data * np.log(model_args.rot_sigma_max) + (1-temp_sigma_data) * np.log(model_args.rot_sigma_min))
            lambda_rot = (rot_sigma_data + rot_sigma) / (rot_sigma_data + rot_sigma/temp_sampling[1])
            rot_perturb = (rot_g ** 2 * dt_rot * (lambda_rot + temp_sampling[1] * temp_psi[1] / 2) * rot_score.cpu() + rot_g * np.sqrt(dt_rot * (1 + temp_psi[1])) * rot_z).cpu()

        if temp_sampling[2] != 1.0:
            tor_sigma_data = np.exp(temp_sigma_data * np.log(model_args.tor_sigma_max) + (1-temp_sigma_data) * np.log(model_args.tor_sigma_min))
            lambda_tor = (tor_sigma_data + tor_sigma) / (tor_sigma_data + tor_sigma/temp_sampling[2])
            tor_perturb = (tor_g ** 2 * dt_tor * (lambda_tor + temp_sampling[2] * temp_psi[2] / 2) * tor_score.cpu() + tor_g * np.sqrt(dt_tor * (1 + temp_psi[2])) * tor_z).cpu().numpy()

        if temp_sampling[3] != 1.0:
            sidechain_tor_sigma_data = np.exp(temp_sigma_data * np.log(model_args.sidechain_tor_sigma_max) + (1-temp_sigma_data) * np.log(model_args.sidechain_tor_sigma_min))
            lambda_sidechain_tor = (sidechain_tor_sigma_data + sidechain_tor_sigma) / (sidechain_tor_sigma_data + sidechain_tor_sigma/temp_sampling[3])
            sidechain_tor_perturb = (sidechain_tor_g ** 2 * dt_sidechain_tor * (lambda_sidechain_tor + temp_sampling[3] * temp_psi[3] / 2) * sidechain_tor_score.cpu() + sidechain_tor_g * np.sqrt(dt_sidechain_tor * (1 + temp_psi[3])) * sidechain_tor_z).cpu().numpy()

        if svgd_weight > 0:
            batch_pos = torch.cat([complex_graph['ligand'].pos.cpu().unsqueeze(0) for complex_graph in data_list])

            tor_matrix = 0
            if data_list[0]['ligand'].edge_mask.sum() > 0: # there are some torsion angles
                tor_matrix, tor_diff = get_torsion_angles_svgd(dihedrals, batch_pos)
            tr_matrix, rot_matrix, tr_diff, rot_diff = get_rigid_svgd(batch_pos)
            total_matrix = tr_matrix + svgd_rot_rel_weight * rot_matrix + svgd_tor_rel_weight * tor_matrix

            med2 = torch.median(total_matrix, dim=1, keepdim=True)[0]
            h = svgd_repulsive_weight * med2 / max(np.log(N), 1)
            k = torch.exp(-1 / h * total_matrix)

            if flexible_sidechains:
                raise NotImplementedError("Not implemented for sidechains")

            tr_repulsive = torch.sum(2 / h * tr_diff * k, dim=1).cpu()
            tr_attractive = torch.sum(k.cpu() * tr_score.reshape(1, N, -1), dim=1)
            tr_total_svgd = (tr_g ** 2 * dt_tr * (tr_attractive + tr_repulsive) / N)

            rot_repulsive = torch.sum(2 / h * svgd_rot_rel_weight * rot_diff * k, dim=1).cpu()
            rot_attractive = torch.sum(k.cpu() * rot_score.reshape(1, N, -1), dim=1)
            rot_total_svgd = (rot_g ** 2 * dt_rot * (rot_attractive + rot_repulsive) / N)

            tor_total_svgd = torch.zeros(tor_score.shape)
            if data_list[0]['ligand'].edge_mask.sum() > 0:
                tor_repulsive = torch.sum(2 / h * svgd_tor_rel_weight * tor_diff * k, dim=1).cpu()
                tor_attractive = torch.sum(k.cpu() * tor_score.reshape(1, N, -1), dim=1)
                tor_total_svgd = (tor_g ** 2 * dt_tor * (tor_attractive + tor_repulsive) / N).reshape(-1)

            sidechain_tor_total_svgd = torch.zeros(sidechain_tor_score.shape)
            if flexible_sidechains:
                sidechain_tor_repulsive = torch.sum(2 / h * svgd_sidechain_tor_rel_weight * sidechain_tor_diff * k, dim=1).cpu()
                sidechain_tor_attractive = torch.sum(k.cpu() * sidechain_tor_score.reshape(1, N, -1), dim=1)
                sidechain_tor_total_svgd = (sidechain_tor_g ** 2 * dt_sidechain_tor * (sidechain_tor_attractive + sidechain_tor_repulsive) / N).reshape(-1)

            if svgd_only:
                tr_perturb = svgd_weight * tr_total_svgd
                rot_perturb = svgd_weight * rot_total_svgd
                tor_perturb = (svgd_weight * tor_total_svgd).numpy()
                sidechain_tor_perturb = (svgd_weight * sidechain_tor_total_svgd).numpy
            else:
                tr_perturb += svgd_weight * tr_total_svgd
                rot_perturb += svgd_weight * rot_total_svgd
                tor_perturb += (svgd_weight * tor_total_svgd).numpy()
                sidechain_tor_perturb += (svgd_weight * sidechain_tor_total_svgd).numpy()

        # Apply noise
        if flexible_sidechains:
            for i, complex_graph in enumerate(data_list):
                modify_sidechains(complex_graph, sidechain_tor_perturb[i*sidechain_torsions_per_molecule:(i+1)*sidechain_torsions_per_molecule])

        data_list = [modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                      tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None, pivot=pivot)
                     for i, complex_graph in enumerate(data_list)]

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)
                
        if sidechain_visualization_list is not None:
            for idx, visualization in enumerate(sidechain_visualization_list):
                # append all subcomponents (i.e., sidechain data)
                visualization.append(data_list[idx][0]["atom"].pos
                                     + data_list[idx][0]["original_center"])

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            filtering_loader = iter(DataLoader(filtering_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if filtering_data_list is not None:
                    filtering_complex_graph_batch = next(filtering_loader).to(device)
                    filtering_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(filtering_complex_graph_batch, 0, 0, 0, 0, 0, N, filtering_model_args.all_atoms,
                             asyncronous_noise_schedule, device, include_miscellaneous_atoms=hasattr(filtering_model_args, 'include_miscellaneous_atoms') and filtering_model_args.include_miscellaneous_atoms)
                    confidence.append(confidence_model(filtering_complex_graph_batch))
                else:
                    set_time(complex_graph_batch, 0, 0, 0, 0, 0, N, filtering_model_args.all_atoms,
                             asyncronous_noise_schedule, device, include_miscellaneous_atoms=hasattr(filtering_model_args, 'include_miscellaneous_atoms') and filtering_model_args.include_miscellaneous_atoms)

                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None
    if return_full_trajectory:
        return data_list, confidence, trajectory, sidechain_trajectory
    return data_list, confidence


def compute_affinity(data_list, affinity_model, affinity_data_list, device, parallel, all_atoms, include_miscellaneous_atoms):

    with torch.no_grad():
        if affinity_model is not None:
            assert parallel <= len(data_list)
            loader = DataLoader(data_list, batch_size=parallel)
            complex_graph_batch = next(iter(loader)).to(device)
            positions = complex_graph_batch['ligand'].pos

            assert affinity_data_list is not None
            complex_graph = affinity_data_list[0]
            N = complex_graph['ligand'].num_nodes
            complex_graph['ligand'].x = complex_graph['ligand'].x.repeat(parallel, 1)
            complex_graph['ligand'].edge_mask = complex_graph['ligand'].edge_mask.repeat(parallel)
            complex_graph['ligand', 'ligand'].edge_index = torch.cat(
                [N * i + complex_graph['ligand', 'ligand'].edge_index for i in range(parallel)], dim=1)
            complex_graph['ligand', 'ligand'].edge_attr = complex_graph['ligand', 'ligand'].edge_attr.repeat(parallel, 1)
            complex_graph['ligand'].pos = positions

            affinity_loader = DataLoader([complex_graph], batch_size=1)
            affinity_batch = next(iter(affinity_loader)).to(device)
            set_time(affinity_batch, 0, 0, 0, 0, 1, all_atoms, True, device, include_miscellaneous_atoms=include_miscellaneous_atoms)
            _, affinity = affinity_model(affinity_batch)
        else:
            affinity = None

    return affinity


