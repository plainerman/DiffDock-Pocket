import functools
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles,modify_sidechain_torsion_angle


def sigmoid(t):
    return 1 / (1 + np.e**(-t))


def sigmoid_schedule(t, k=10, m=0.5):
    s = lambda t: sigmoid(k*(t-m))
    return (s(t)-s(0))/(s(1)-s(0))


def t_to_sigma_individual(t, schedule_type, sigma_min, sigma_max, schedule_k=10, schedule_m=0.4):
    if schedule_type == "exponential":
        return sigma_min ** (1 - t) * sigma_max ** t
    elif schedule_type == 'sigmoid':
        return sigmoid_schedule(t, k=schedule_k, m=schedule_m) * (sigma_max - sigma_min) + sigma_min


def t_to_sigma(t_tr, t_rot, t_tor, t_sc_tor, args):
    tr_sigma = t_to_sigma_individual(t_tr, 'exponential', args.tr_sigma_min, args.tr_sigma_max)
    rot_sigma = t_to_sigma_individual(t_rot, 'exponential', args.rot_sigma_min, args.rot_sigma_max)
    tor_sigma = t_to_sigma_individual(t_tor, 'exponential', args.tor_sigma_min, args.tor_sigma_max)
    sc_tor_sigma = t_to_sigma_individual(t_sc_tor, 'exponential', args.sidechain_tor_sigma_min, args.sidechain_tor_sigma_max)
    return tr_sigma, rot_sigma, tor_sigma, sc_tor_sigma


def modify_conformer(data, tr_update, rot_update, torsion_updates, pivot=None):
    orig_device = data['ligand'].pos.device
    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True).to(orig_device)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze()).to(orig_device)
    rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T + tr_update.to(orig_device) + lig_center.to(orig_device)

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                           data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) else data['ligand'].mask_rotate[0],
                                                           torsion_updates).to(rigid_new_pos.device)
        if pivot is None:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        else:
            R1, t1 = rigid_transform_Kabsch_3D_torch(pivot.T, rigid_new_pos.T)
            R2, t2 = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, pivot.T)

            aligned_flexible_pos = (flexible_new_pos @ R2.T + t2.T) @ R1.T + t1.T

        data['ligand'].pos = aligned_flexible_pos.to(orig_device)
    else:
        data['ligand'].pos = rigid_new_pos.to(orig_device)
    return data


def modify_conformer_coordinates(pos, tr_update, rot_update, torsion_updates, edge_mask, mask_rotate, edge_index):
    # Made this function which does the same as modify_conformer because passing a graph would require
    # creating a new heterograph for reach graph when unbatching a batch of graphs
    lig_center = torch.mean(pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (pos - lig_center) @ rot_mat.T + tr_update + lig_center

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,edge_index.T[edge_mask],mask_rotate \
            if isinstance(mask_rotate, np.ndarray) else mask_rotate[0], torsion_updates).to(rigid_new_pos.device)

        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        return aligned_flexible_pos
    else:
        return rigid_new_pos


def modify_sidechains(data, torsion_updates):
    # iterate over all torsion updates and modify the corresponding atoms 
    for i, torsion_update in enumerate(torsion_updates):
        data['atom'].pos = modify_sidechain_torsion_angle(data['atom'].pos,
                                                          data['flexResidues'].edge_idx[i],
                                                          data['flexResidues'].subcomponentsMapping[i],
                                                          data['flexResidues'].subcomponents,
                                                          torsion_update)


def sinusoidal_embedding(timesteps, dim, scale=1.0, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    timesteps *= scale
    half_dim = dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, dim, scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = functools.partial(sinusoidal_embedding, dim=dim, scale=scale)
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(size=dim, scale=scale)
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(sigma_schedule, inference_steps, inf_sched_alpha=1, inf_sched_beta=1, t_max=1):
    if sigma_schedule == 'expbeta':
        lin_max = beta.cdf(t_max, a=inf_sched_alpha, b=inf_sched_beta)
        c = np.linspace(lin_max, 0, inference_steps + 1)[:-1]
        return beta.ppf(c, a=inf_sched_alpha, b=inf_sched_beta)
    raise Exception()


def get_inverse_schedule(t, sched_alpha=1, sched_beta=1):
    return beta.ppf(t, a=sched_alpha, b=sched_beta)


def set_time(complex_graphs, t, t_tr, t_rot, t_tor, t_sidechain_tor, batchsize, all_atoms, asyncronous_noise_schedule, device, include_miscellaneous_atoms=False):
    complex_graphs['ligand'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'sc_tor': t_sidechain_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
    }
    complex_graphs['receptor'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'sc_tor': t_sidechain_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
    }
    complex_graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),
                               'rot': t_rot * torch.ones(batchsize).to(device),
                               'tor': t_tor * torch.ones(batchsize).to(device),
                               'sc_tor': t_sidechain_tor * torch.ones(batchsize).to(device)
    }

    if all_atoms:
        complex_graphs['atom'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'sc_tor': t_sidechain_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device),
        }

    # TODO: asynchronous noise schedule for sidechain torsions ? 
    if include_miscellaneous_atoms and not all_atoms:
        complex_graphs['misc_atom'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['misc_atom'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['misc_atom'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['misc_atom'].num_nodes).to(device),
            'sc_tor': t_sidechain_tor * torch.ones(complex_graphs['misc_atom'].num_nodes).to(device)}
    if asyncronous_noise_schedule:
        complex_graphs['ligand'].node_t['t'] = t * torch.ones(complex_graphs['ligand'].num_nodes).to(device)
        complex_graphs['receptor'].node_t['t'] = t * torch.ones(complex_graphs['receptor'].num_nodes).to(device)
        complex_graphs.complex_t['t'] = t * torch.ones(batchsize).to(device)
        if all_atoms:
            complex_graphs['atom'].node_t['t'] = t * torch.ones(complex_graphs['atom'].num_nodes).to(device)
        if include_miscellaneous_atoms:
            complex_graphs['misc_atom'].node_t['t'] = t * torch.ones(complex_graphs['misc_atom'].num_nodes).to(device)
