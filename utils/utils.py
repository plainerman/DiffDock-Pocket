import logging
import math
import os
import subprocess
import warnings
from datetime import datetime
from contextlib import contextmanager
import signal
from functools import wraps
from typing import List, Tuple

import numpy
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import RemoveHs, MolToPDBFile
from torch import nn, Tensor
from torch_geometric.nn.data_parallel import DataParallel
from torch_geometric.utils import degree

from models.all_atom_score_model import TensorProductScoreModel as AAScoreModel
from models.score_model import TensorProductScoreModel as CGScoreModel
from utils.diffusion_utils import get_timestep_embedding
from spyrmsd import rmsd, molecule


def get_obrmsd(mol1_path, mol2_path, cache_name=None):
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f') if cache_name is None else cache_name
    os.makedirs(".openbabel_cache", exist_ok=True)
    if not isinstance(mol1_path, str):
        MolToPDBFile(mol1_path, '.openbabel_cache/obrmsd_mol1_cache.pdb')
        mol1_path = '.openbabel_cache/obrmsd_mol1_cache.pdb'
    if not isinstance(mol2_path, str):
        MolToPDBFile(mol2_path, '.openbabel_cache/obrmsd_mol2_cache.pdb')
        mol2_path = '.openbabel_cache/obrmsd_mol2_cache.pdb'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return_code = subprocess.run(f"obrms {mol1_path} {mol2_path} > .openbabel_cache/obrmsd_{cache_name}.rmsd",
                                     shell=True)
        print(return_code)
    obrms_output = read_strings_from_txt(f".openbabel_cache/obrmsd_{cache_name}.rmsd")
    rmsds = [line.split(" ")[-1] for line in obrms_output]
    return np.array(rmsds, dtype=np.float)


def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def unbatch(src, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    if isinstance(src, numpy.ndarray):
        return np.split(src, np.array(sizes).cumsum()[:-1], axis=dim)
    else:
        return src.split(sizes, dim)


def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def unbatch_edge_attributes(edge_attributes, edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    edge_batch = batch[edge_index[0]]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_attributes.split(sizes, dim=0)


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def get_optimizer_and_scheduler(args, model, scheduler_mode='min'):
    optimizer = torch.optim.AdamW if args.adamw == 'adamw' else torch.optim.Adam
    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=float(args.lr),
                                 weight_decay=args.w_decay)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=float(args.lr) / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False):
    if 'all_atoms' in args and args.all_atoms:
        model_class = AAScoreModel
    else:
        model_class = CGScoreModel

    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type if 'embedding_type' in args else 'sinusoidal',
        dim=args.sigma_embed_dim,
        scale=args.embedding_scale if 'embedding_scale' in args else 10000)

    # lm_embedding_type = None
    lm_embedding_type = 'esm'

    model = model_class(t_to_sigma=t_to_sigma,
                        device=device,
                        no_torsion=args.no_torsion,
                        timestep_emb_func=timestep_emb_func,
                        num_conv_layers=args.num_conv_layers,
                        lig_max_radius=args.max_radius,
                        scale_by_sigma=args.scale_by_sigma,
                        sh_lmax=args.sh_lmax,
                        sigma_embed_dim=args.sigma_embed_dim,
                        norm_by_sigma='norm_by_sigma' in args and args.norm_by_sigma,
                        ns=args.ns, nv=args.nv,
                        distance_embed_dim=args.distance_embed_dim,
                        cross_distance_embed_dim=args.cross_distance_embed_dim,
                        batch_norm=not args.no_batch_norm,
                        dropout=args.dropout,
                        use_second_order_repr=args.use_second_order_repr,
                        cross_max_distance=args.cross_max_distance,
                        dynamic_max_cross=args.dynamic_max_cross,
                        separate_noise_schedule=args.separate_noise_schedule,
                        smooth_edges=args.smooth_edges if "smooth_edges" in args else False,
                        odd_parity=args.odd_parity if "odd_parity" in args else False,
                        lm_embedding_type=lm_embedding_type,
                        confidence_mode=confidence_mode,
                        asyncronous_noise_schedule=args.asyncronous_noise_schedule if "asyncronous_noise_schedule" in args else False,
                        affinity_prediction=args.affinity_prediction if 'affinity_prediction' in args else False,
                        parallel=args.parallel if "parallel" in args else 1,
                        num_confidence_outputs=len(
                            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                            args.rmsd_classification_cutoff, list) else 1,
                        parallel_aggregators=args.parallel_aggregators if "parallel_aggregators" in args else "",
                        fixed_center_conv=not args.not_fixed_center_conv if "not_fixed_center_conv" in args else False,
                        no_aminoacid_identities=args.no_aminoacid_identities if "no_aminoacid_identities" in args else False,
                        atom_max_neighbors=args.atom_max_neighbors,
                        flexible_sidechains=args.flexible_sidechains,
                        include_miscellaneous_atoms=args.include_miscellaneous_atoms if hasattr(args, 'include_miscellaneous_atoms') else False,
                        use_old_atom_encoder=args.use_old_atom_encoder if hasattr(args, 'use_old_atom_encoder') else True)

    if device.type == 'cuda' and not no_parallel:
        model = DataParallel(model)
    model.to(device)
    return model


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
        )
        return RMSD


def to_none(x):
    if isinstance(x, list):
        return [to_none(y) for y in x]
    if isinstance(x, (int, float, complex, np.number)):
        return None if (math.isnan(x) or pd.isna(x)) else x
    elif not x:
        return None
    else:
        return x


def center_to_torch(x, y, z):
    if any([to_none(i) is None for i in (x, y, z)]):
        return None
    return torch.tensor([x, y, z], dtype=torch.float32)


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # Not all operations implemented in MPS yet
        use_mps = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
        if use_mps:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')


def get_available_devices(max_devices=None):
    device = get_default_device()
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        if max_devices is not None:
            num_gpus = min(num_gpus, max_devices)
        gpu_list = [get_device(i) for i in range(num_gpus)]
        return gpu_list
    else:
        num_devices = 1
        if max_devices is None and device.type == "cpu":
            num_devices = torch.multiprocessing.cpu_count()
        return [device]*num_devices


def get_device(gpu_id: int):
    if gpu_id is not None and torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu_id}')
    else:
        return None


def ensure_device(func):
    """
    Decorator to ensure that the device kwarg is set to the default device if it is None.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        device = kwargs.get("device", None)
        # We specifically only do this stuff if device is one of the kwargs, but is None
        needs_device = "device" in kwargs and kwargs["device"] is None
        if needs_device:
            device = get_default_device()
        kwargs["device"] = device
        if device is not None and device.type == 'cuda':
            # Set the default device to the device we are using
            # This will ensure any new tensors created are on the correct device
            with torch.cuda.device(device):
                result = func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)

        return result

    return wrapper


def count_open_files():
    """Count the number of open files in /dev/shm/torch. Checking for memory leaks with torch multiprocessing"""
    import subprocess
    args = ['lsof', '-p', str(os.getpid())]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    lines = stdout.decode().splitlines()

    # Filter for lines containing "dev/shm/torch"
    filtered_lines = [line for line in lines if "dev/shm/torch" in line]
    return len(filtered_lines)

