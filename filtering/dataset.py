import itertools
import math
import os
import pickle
import random
from argparse import Namespace
from functools import partial
import binascii
import copy

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.pdbbind import PDBBind
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_cache_path(args, split):
    cache_path = args.cache_path
    split_path = args.split_train if split == 'train' else args.split_val
    matching = True
    full_dataset = True
    cross_docking = False
    use_original_protein_file = False
    keep_local_structures = False
    protein_path_list = None
    ligand_descriptions = None
    fixed_knn_radius_graph = False if not hasattr(args, 'not_fixed_knn_radius_graph') else not args.not_fixed_knn_radius_graph
    knn_only_graph = False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph
    include_miscellaneous_atoms = False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms
    use_old_wrong_embedding_order = True if not hasattr(args, 'use_old_wrong_embedding_order') else args.use_old_wrong_embedding_order
    match_protein_file = 'protein_processed_fix' if not hasattr(args, 'match_protein_file') else args.match_protein_file
    use_original_conformer_fallback = False if not hasattr(args, 'use_original_conformer_fallback') else args.use_original_conformer_fallback
    conformer_match_sidechains = False if not hasattr(args, 'conformer_match_sidechains') else args.conformer_match_sidechains
    match_max_rmsd = 2.0 if not hasattr(args, 'match_max_rmsd') else args.match_max_rmsd
    use_original_conformer = False if not hasattr(args, 'use_original_conformer') else args.use_original_conformer
    conformer_match_score = 'dist' if not hasattr(args, 'conformer_match_score') else args.conformer_match_score
    compare_true_protein = False if not hasattr(args, 'compare_true_protein') else args.compare_true_protein

    if split == 'train':
        compare_true_protein = False
    else:
        match_max_rmsd = None
        use_original_conformer = False
        use_original_conformer_fallback = False

    if matching or protein_path_list is not None and ligand_descriptions is not None:
        cache_path += '_torsion'
    if args.all_atoms:
        cache_path += '_allatoms'

    cache_path = os.path.join(cache_path, f'limit{args.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(split_path))[0]}'
                                                        f'_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}'
                                                        f'_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                                        f'_chainCutoff{args.chain_cutoff}'
                                            + ('' if not matching or args.num_conformers == 1 else f'_confs{args.num_conformers}')
                                            + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + ('' if not full_dataset else f'_full')
                                            + ('' if not cross_docking else f'_cross')
                                            + ('' if not use_original_protein_file else f'_origProt')
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode())))
                                            + ('' if args.protein_file == "protein_processed" else '_' + args.protein_file)
                                            + ('' if match_protein_file == "protein_processed_fix" else '_' + match_protein_file)
                                            + ('' if not use_original_conformer_fallback else '_fallback')
                                            + ('' if not conformer_match_sidechains else '_match' + ((('' if match_max_rmsd is None else str(match_max_rmsd))) + ('' if not use_original_conformer else 'with_orig')))
                                            + ('' if not conformer_match_sidechains else ('' if conformer_match_score == 'dist' else '_score' + conformer_match_score))
                                            + ('' if not compare_true_protein else '_compare')
                                            + ('' if not args.pocket_reduction else '_reduced' + str(args.pocket_buffer))
                                            + ('' if not args.pocket_reduction else ('' if args.pocket_reduction_mode == 'center-dist' else '_' + str(args.pocket_reduction_mode)))
                                            + ('' if not args.pocket_reduction else ('' if args.pocket_cutoff == 5.0 else '_' + str(args.pocket_cutoff)))
                                            + ('' if not args.pocket_reduction else ('skip' if args.skip_no_pocket_atoms else ''))
                                            + ('' if not args.flexible_sidechains else '_flexible' + str(args.flexdist) + args.flexdist_distance_metric)
                                            + ('' if not fixed_knn_radius_graph else (f'_fixedKNN' if not knn_only_graph else '_fixedKNNonly'))
                                            + ('' if not include_miscellaneous_atoms else '_miscAtoms')
                                            + ('' if use_old_wrong_embedding_order else '_chainOrd'))
    return cache_path

def get_args_and_cache_path(original_model_dir, split, protein_file=None):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
        if not hasattr(model_args, 'separate_noise_schedule'):  # exists for compatibility
            model_args.separate_noise_schedule = False
        if not hasattr(model_args, 'lm_embeddings_path'):  # exists for compatibility
            model_args.lm_embeddings_path = None
        if not hasattr(model_args, 'all_atoms'):  # exists for compatibility
            model_args.all_atoms = False
        if not hasattr(model_args,'tr_only_confidence'):  # exists for compatibility
            model_args.tr_only_confidence = True
        if not hasattr(model_args,'high_confidence_threshold'):  # exists for compatibility
            model_args.high_confidence_threshold = 0.0
        if not hasattr(model_args, 'include_confidence_prediction'):  # exists for compatibility
            model_args.include_confidence_prediction = False
        if not hasattr(model_args, 'confidence_dropout'):
            model_args.confidence_dropout = model_args.dropout
        if not hasattr(model_args, 'confidence_no_batchnorm'):
            model_args.confidence_no_batchnorm = False
        if not hasattr(model_args, 'confidence_weight'):
            model_args.confidence_weight = 1
        if not hasattr(model_args, 'asyncronous_noise_schedule'):
            model_args.asyncronous_noise_schedule = False
        if not hasattr(model_args, 'correct_torsion_sigmas'):
            model_args.correct_torsion_sigmas = False
        if not hasattr(model_args, 'not_full_dataset'):
            model_args.not_full_dataset = True
        if not hasattr(model_args, 'esm_embeddings_path'):
            model_args.esm_embeddings_path = None
        if protein_file is not None:
            model_args.protein_file = protein_file

    return model_args, get_cache_path(model_args, split)


class FilteringDataset(Dataset):
    def __init__(self, cache_path, original_model_dir, split, device, sigma_schedule, limit_complexes,
                 inference_steps, inf_sched_alpha, inf_sched_beta, rot_inf_sched_alpha, rot_inf_sched_beta,
                 tor_inf_sched_alpha, tor_inf_sched_beta, samples_per_complex, different_schedules, all_atoms,
                 args, model_ckpt, balance=False, multiplicity=1,use_original_model_cache=True,
                 rmsd_classification_cutoff=2, sc_rmsd_classification_cutoff=1,
                 parallel=1, cache_ids_to_combine= None, cache_creation_id=None, trajectory_sampling=False, include_miscellaneous_atoms=False):

        super(FilteringDataset, self).__init__()

        self.device, self.sigma_schedule = device, sigma_schedule
        self.inference_steps = inference_steps
        self.inf_sched_alpha, self.inf_sched_beta = inf_sched_alpha, inf_sched_beta
        self.rot_inf_sched_alpha, self.rot_inf_sched_beta = rot_inf_sched_alpha, rot_inf_sched_beta
        self.tor_inf_sched_alpha, self.tor_inf_sched_beta = tor_inf_sched_alpha, tor_inf_sched_beta
        self.different_schedules, self.limit_complexes = different_schedules, limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.multiplicity = multiplicity
        self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.sc_rmsd_classification_cutoff = sc_rmsd_classification_cutoff
        self.parallel = parallel
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt
        self.args, self.split = args, split
        self.trajectory_sampling = trajectory_sampling
        self.fixed_step = None
        self.fixed_sample = None
        self.include_miscellaneous_atoms = include_miscellaneous_atoms

        self.original_model_args, original_model_cache = get_args_and_cache_path(original_model_dir, split, protein_file=args.protein_file)
        self.full_cache_path = get_cache_path(args, split)

        if (not os.path.exists(os.path.join(self.full_cache_path, "ligand_positions.pkl")) and self.cache_creation_id is None) or \
                (not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{self.cache_creation_id}.pkl")) and self.cache_creation_id is not None):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing(original_model_cache)

        if self.original_model_args.flexible_sidechains:
            if not self.use_original_model_cache:
                raise ValueError('Flexible sidechains are not supported when not using the original model cache.'
                                 'Please use the original model cache of a model with flexible sidechains.')

        self.complex_graphs_cache = original_model_cache if self.use_original_model_cache else self.full_cache_path
        print('Using the cached complex graphs of the original model args' if self.use_original_model_cache else 'Not using the cached complex graphs of the original model args. Instead the complex graphs are used that are at the location given by the dataset parameters given to filtering_train.py')
        print(self.complex_graphs_cache)
        if not os.path.exists(os.path.join(self.complex_graphs_cache, "heterographs.pkl")):
            print(f'HAPPENING | Complex graphs path does not exist yet: {os.path.join(self.complex_graphs_cache, "heterographs.pkl")}. For that reason, we are now creating the dataset.')
            PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes, multiplicity=args.multiplicity,
                    chain_cutoff=args.chain_cutoff,
                    receptor_radius=args.receptor_radius,
                    cache_path=args.cache_path, split_path=args.split_val if split == 'val' else args.split_train,
                    remove_hs=args.remove_hs, max_lig_size=None,
                    c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                    matching=not args.no_torsion, keep_original=True,
                    popsize=args.matching_popsize,
                    maxiter=args.matching_maxiter,
                    all_atoms=args.all_atoms,
                    esm_embeddings_path=args.esm_embeddings_path,
                    require_ligand=True,
                    num_workers=args.num_workers,
                    protein_file=args.protein_file,
                    pocket_reduction=args.pocket_reduction, pocket_buffer=args.pocket_buffer,
                    pocket_cutoff=args.pocket_cutoff, skip_no_pocket_atoms=args.skip_no_pocket_atoms,
                    pocket_reduction_mode=args.pocket_reduction_mode,
                    flexible_sidechains=args.flexible_sidechains, flexdist=args.flexdist, flexdist_distance_metric=args.flexdist_distance_metric,
                    fixed_knn_radius_graph=False if not hasattr(args, 'not_fixed_knn_radius_graph') else not args.not_fixed_knn_radius_graph,
                    knn_only_graph=False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                    include_miscellaneous_atoms=False if not hasattr(args,'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                    use_old_wrong_embedding_order=True if not hasattr(args, 'use_old_wrong_embedding_order') else args.use_old_wrong_embedding_order
                    )

        print(f'HAPPENING | Loading complex graphs from: {os.path.join(self.complex_graphs_cache, "heterographs.pkl")}')
        with open(os.path.join(self.complex_graphs_cache, "heterographs.pkl"), 'rb') as f:
            complex_graphs = pickle.load(f)
        self.complex_graph_dict = {d.name: d for d in complex_graphs}

        if self.cache_ids_to_combine is None:
            print(f'HAPPENING | Loading positions and rmsds from: {os.path.join(self.full_cache_path, "ligand_positions.pkl")}')
            if trajectory_sampling:
                with open(os.path.join(self.full_cache_path, f"trajectories.pkl"), 'rb') as f:
                    self.full_ligand_positions, self.rmsds = pickle.load(f)
                if self.original_model_args.flexible_sidechains:
                    with open(os.path.join(self.full_cache_path, f"trajectories.pkl"), 'rb') as f:
                        self.full_sc_positions, self.sc_rmsds = pickle.load(f)
            else:
                with open(os.path.join(self.full_cache_path, "ligand_positions.pkl"), 'rb') as f:
                    self.full_ligand_positions, self.rmsds = pickle.load(f)

                if self.original_model_args.flexible_sidechains:
                    with open(os.path.join(self.full_cache_path, "sc_positions.pkl"), 'rb') as f:
                        self.full_sc_positions, self.sc_rmsds = pickle.load(f)
            if os.path.exists(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl")):
                with open(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"), 'rb') as f:
                    generated_rmsd_complex_names = pickle.load(f)
            else:
                print('HAPPENING | The path, ', os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"),
                      ' does not exist. \n => We assume that means that we are using a ligand_positions.pkl where the '
                      'code was not saving the complex names for them yet. We now instead use the complex names of '
                      'the dataset that the original model used to create the ligand positions and RMSDs.')
                with open(os.path.join(original_model_cache, "heterographs.pkl"), 'rb') as f:
                    original_model_complex_graphs = pickle.load(f)
                    generated_rmsd_complex_names = [d.name for d in original_model_complex_graphs]
        else:
            all_rmsds_unsorted, all_sc_rmsds_unsorted, all_full_ligand_positions_unsorted, all_full_sc_positions_unsorted, all_names_unsorted = [], [], [], [], []
            for idx, cache_id in enumerate(self.cache_ids_to_combine):
                print(f'HAPPENING | Loading positions and rmsds from cache_id {cache_id} from the path: {os.path.join(self.full_cache_path, ("trajectories_" if self.trajectory_sampling else "ligand_positions_") + str(cache_id) + ".pkl")}')
                if not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl")): raise Exception(f'The generated ligand positions with cache_id do not exist: {cache_id}') # be careful with changing this error message since it is sometimes caught in a try catch
                if trajectory_sampling:
                    with open(os.path.join(self.full_cache_path, f"trajectories_id{cache_id}.pkl"), 'rb') as f:
                        full_ligand_positions, rmsds = pickle.load(f)
                    if self.original_model_args.flexible_sidechains:
                        with open(os.path.join(self.full_cache_path, f"sidechain_trajectories_id{cache_id}.pkl"), 'rb') as f:
                            full_sc_positions, sc_rmsds = pickle.load(f)
                else:
                    with open(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl"), 'rb') as f:
                        full_ligand_positions, rmsds = pickle.load(f)
                    if self.original_model_args.flexible_sidechains:
                        with open(os.path.join(self.full_cache_path, f"sc_positions_id{cache_id}.pkl"), 'rb') as f:
                            full_sc_positions, sc_rmsds = pickle.load(f)
                with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                    names_unsorted = pickle.load(f)
                all_names_unsorted.append(names_unsorted)
                all_rmsds_unsorted.append(rmsds)
                all_full_ligand_positions_unsorted.append(full_ligand_positions)

                if self.original_model_args.flexible_sidechains:
                    all_sc_rmsds_unsorted.append(sc_rmsds)
                    all_full_sc_positions_unsorted.append(full_sc_positions)
                else:
                    all_sc_rmsds_unsorted.append(None)
                    all_full_sc_positions_unsorted.append(None)

            names_order = list(set.intersection(*map(set, all_names_unsorted)))
            all_rmsds, all_sc_rmsds, all_full_ligand_positions, all_full_sc_positions, all_names = [], [], [], [], []
            for idx, (rmsds_unsorted, sc_rmsds_unsorted, full_ligand_positions_unsorted, full_sc_positions_unsorted, names_unsorted) in enumerate(zip(all_rmsds_unsorted, all_sc_rmsds_unsorted, all_full_ligand_positions_unsorted, all_full_sc_positions_unsorted, all_names_unsorted)):
                name_to_pos_dict = {name: (pos, rmsd, sc_pos, sc_rmsd) for name, pos, rmsd, sc_pos, sc_rmsd in zip(names_unsorted, full_ligand_positions_unsorted, rmsds_unsorted, full_sc_positions_unsorted, sc_rmsds_unsorted) }
                all_full_ligand_positions.append([name_to_pos_dict[name][0] for name in names_order])
                all_rmsds.append([name_to_pos_dict[name][1] for name in names_order])
                all_full_sc_positions.append([name_to_pos_dict[name][2] for name in names_order])
                all_sc_rmsds.append([name_to_pos_dict[name][3] for name in names_order])
            self.full_ligand_positions, self.rmsds = [], []

            if self.original_model_args.flexible_sidechains:
                self.full_sc_positions, self.sc_rmsds = [], []

            for positions_tuple in list(zip(*all_full_ligand_positions)):
                self.full_ligand_positions.append(np.concatenate(positions_tuple, axis=(1 if trajectory_sampling else 0)))
            for positions_tuple in list(zip(*all_rmsds)):
                self.rmsds.append(np.concatenate(positions_tuple, axis=0))
            if self.original_model_args.flexible_sidechains:
                for positions_tuple in list(zip(*all_full_sc_positions)):
                    self.full_sc_positions.append(
                        np.concatenate(positions_tuple, axis=(1 if trajectory_sampling else 0)))
                for positions_tuple in list(zip(*all_sc_rmsds)):
                    self.sc_rmsds.append(np.concatenate(positions_tuple, axis=0))
            generated_rmsd_complex_names = names_order

        assert (len(self.rmsds) == len(generated_rmsd_complex_names))
        if self.original_model_args.flexible_sidechains:
            assert (len(self.sc_rmsds) == len(generated_rmsd_complex_names))

        print('Number of complex graphs:', len(self.complex_graph_dict))
        print('Number of overall samples:', len(np.array(self.rmsds).flatten()))
        print('Number of RMSDs and positions for the complex graphs:', len(self.full_ligand_positions))
        print('1st position shape:', self.full_ligand_positions[0].shape)

        if self.original_model_args.flexible_sidechains:
            print('Number of sidechain RMSDs and positions for the complex graphs:', len(self.full_sc_positions))
            print('1st sidechain position shape:', self.full_sc_positions[0].shape)

        self.all_samples_per_complex = samples_per_complex * (1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine))

        if self.original_model_args.flexible_sidechains:
            self.positions_rmsds_dict = {name: (pos, rmsd, sc_pos, sc_rmsd) for name, pos, rmsd, sc_pos, sc_rmsd in
                                         zip(generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds,
                                             self.full_sc_positions, self.sc_rmsds)}
        else:
            self.positions_rmsds_dict = {name: (pos, rmsd, None, None) for name, pos, rmsd in
                                         zip(generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds)}

        self.dataset_names = list(set(self.positions_rmsds_dict.keys()) & set(self.complex_graph_dict.keys()))
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]

        if not isinstance(self.rmsd_classification_cutoff, list):
            successful_docking = np.array(self.rmsds).flatten() < self.rmsd_classification_cutoff
            print(f'{(successful_docking.mean() * 100).round(2)}% of the complexes have RMSD less than {self.rmsd_classification_cutoff}')
            if self.original_model_args.flexible_sidechains:
                if not isinstance(self.sc_rmsd_classification_cutoff, list):
                    successful_sidechains = np.array(self.sc_rmsds).flatten() < self.sc_rmsd_classification_cutoff
                    print(f'{(successful_sidechains.mean() * 100).round(2)}% of the complexes have sidechain RMSD less than {self.sc_rmsd_classification_cutoff}')
                    print(f'{((successful_docking & successful_sidechains).mean() * 100).round(2)}% of the complexes have both RMSD and sidechain RMSD within range')

        # for affinity prediction
        df = pd.read_csv('data/INDEX_general_PL_data.2020', sep="  |//|=", comment='#', header=None,
                         names=['PDB code', 'resolution', 'release year', '-logKd/Ki', 'Kd/Ki', 'Kd/Ki value',
                                'reference ligand name', 'refef', 'ef', 'ee', 'asd'])
        self.affinities = df.set_index('PDB code').to_dict()['-logKd/Ki']

    def len(self):
        return len(self.dataset_names) * self.multiplicity

    def get(self, idx):
        if self.multiplicity > 1: idx = idx % len(self.dataset_names)

        complex_graph = copy.deepcopy(self.complex_graph_dict[self.dataset_names[idx]])
        positions, rmsds, sc_positions, sc_rmsds = self.positions_rmsds_dict[self.dataset_names[idx]]
        t = 0

        if self.parallel > 1:
            if self.parallel == len(rmsds):
                idxs = np.arange(self.parallel)
            elif self.parallel < len(rmsds):
                idxs = np.random.choice(len(rmsds), size=self.parallel, replace=False)
            else:
                raise Exception("parallel size larger than sample size")

            N = complex_graph['ligand'].num_nodes
            complex_graph['ligand'].x = complex_graph['ligand'].x.repeat(self.parallel, 1)
            complex_graph['ligand'].edge_mask = complex_graph['ligand'].edge_mask.repeat(self.parallel)
            complex_graph['ligand', 'ligand'].edge_index = torch.cat([N*i+complex_graph['ligand', 'ligand'].edge_index for i in range(self.parallel)], dim=1)
            complex_graph['ligand', 'ligand'].edge_attr = complex_graph['ligand', 'ligand'].edge_attr.repeat(self.parallel, 1)
            complex_graph['ligand'].pos = torch.from_numpy(positions[idxs].reshape(-1, 3))
            complex_graph.rmsd = torch.from_numpy(rmsds[idxs]).unsqueeze(0)
            complex_graph.y = torch.from_numpy(rmsds[idxs]<2).unsqueeze(0).float()

            if self.original_model_args.flexible_sidechains:
                raise NotImplementedError("parallel not implemented for flexible sidechains")
        else:
            if self.trajectory_sampling:
                step = random.randint(0, len(positions)-1) if self.fixed_step is None else self.fixed_step
                t = step/(len(positions)-1)
                positions = positions[len(positions)-step-1]
            if self.balance:
                if isinstance(self.rmsd_classification_cutoff, list): raise ValueError("a list for --rmsd_classification_cutoff can only be used without --balance")
                label = random.randint(0, 1)
                success = rmsds < self.rmsd_classification_cutoff
                n_success = np.count_nonzero(success)
                if (label == 0 and n_success != self.all_samples_per_complex) or (n_success == 0 and self.trajectory_sampling):
                    # sample negative complex
                    sample = random.randint(0, self.all_samples_per_complex - n_success - 1)
                    lig_pos = positions[~success][sample]
                    complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
                else:
                    # sample positive complex
                    if n_success > 0: # if no successfull sample returns the matched complex
                        sample = random.randint(0, n_success - 1)
                        lig_pos = positions[success][sample]
                        complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
                complex_graph.y = torch.tensor(label).float()

                if self.original_model_args.flexible_sidechains:
                    raise NotImplementedError("parallel not implemented for flexible sidechains")
            else:
                sample = random.randint(0, self.all_samples_per_complex - 1) if self.fixed_sample is None else self.fixed_sample

                complex_graph['ligand'].pos = torch.from_numpy(positions[sample])
                complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
                if isinstance(self.rmsd_classification_cutoff, list):
                    complex_graph.y_binned = torch.tensor(np.logical_and(rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],rmsds[sample] >= [0] + self.rmsd_classification_cutoff), dtype=torch.float).unsqueeze(0)
                    complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
                complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()

                if self.original_model_args.flexible_sidechains:
                    flex_ids = complex_graph["flexResidues"].subcomponents.unique().cpu().numpy()
                    if len(flex_ids) > 0:
                        complex_graph['atom'].pos[flex_ids] = torch.from_numpy(sc_positions[sample])
                        # We multiply the target outcome because we want our confidence model to learn good predictions
                        # Meaning that the RMSD of the ligand < threshold AND the sidechain RMSD < threshold
                        try:
                            complex_graph.y *= torch.tensor(sc_rmsds[sample] < self.sc_rmsd_classification_cutoff).float().unsqueeze(0)
                        except Exception as e:
                            print(e)
                            print(complex_graph.name)
                            print(complex_graph.y)
                            print(sc_rmsds[sample].shape)
                            print(sc_rmsds[sample] < self.sc_rmsd_classification_cutoff)
                            raise e

                        assert isinstance(self.sc_rmsd_classification_cutoff, list) == \
                               isinstance(self.rmsd_classification_cutoff, list),\
                            "sc_rmsd_classification_cutoff and rmsd_classification_cutoff must be both lists or both not lists"

                        if isinstance(self.sc_rmsd_classification_cutoff, list):
                            # TODO: implement this
                            raise NotImplementedError("sc_rmsd_classification_cutoff as list not implemented")

                    complex_graph.sc_rmsd = torch.tensor(sc_rmsds[sample]).unsqueeze(0).float()

        complex_graph['ligand'].node_t = {'tr': t * torch.ones(complex_graph['ligand'].num_nodes),
                                          'rot': t * torch.ones(complex_graph['ligand'].num_nodes),
                                          'tor': t * torch.ones(complex_graph['ligand'].num_nodes),
                                          'sc_tor': t * torch.ones(complex_graph['atom'].num_nodes)}
        complex_graph['receptor'].node_t = {'tr': t * torch.ones(complex_graph['receptor'].num_nodes),
                                            'rot': t * torch.ones(complex_graph['receptor'].num_nodes),
                                            'tor': t * torch.ones(complex_graph['receptor'].num_nodes),
                                            'sc_tor': t * torch.ones(complex_graph['atom'].num_nodes)}
        if self.all_atoms:
            complex_graph['atom'].node_t = {'tr': t * torch.ones(complex_graph['atom'].num_nodes),
                                            'rot': t * torch.ones(complex_graph['atom'].num_nodes),
                                            'tor': t * torch.ones(complex_graph['atom'].num_nodes),
                                            'sc_tor': t * torch.ones(complex_graph['atom'].num_nodes)}
        if self.include_miscellaneous_atoms:
            complex_graph['misc_atom'].node_t = {'tr': t * torch.ones(complex_graph['misc_atom'].num_nodes),
                                                 'rot': t * torch.ones(complex_graph['misc_atom'].num_nodes),
                                                 'tor': t * torch.ones(complex_graph['misc_atom'].num_nodes),
                                                 'sc_tor': t * torch.ones(complex_graph['misc_atom'].num_nodes)}
        complex_graph.complex_t = {'tr': t * torch.ones(1), 'rot': t * torch.ones(1),
                                   'tor': t * torch.ones(1), 'sc_tor': t * torch.ones(1)}
        complex_graph.affinity = torch.tensor(self.affinities[complex_graph.name]).float()
        return complex_graph

    def preprocessing(self, original_model_cache):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)

        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{self.original_model_dir}/{self.model_ckpt}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()

        tr_schedule = get_t_schedule(sigma_schedule=self.sigma_schedule, inference_steps=self.inference_steps,
                                     inf_sched_alpha=self.inf_sched_alpha, inf_sched_beta=self.inf_sched_beta)
        if self.different_schedules:
            rot_schedule = get_t_schedule(sigma_schedule=self.rot_sigma_schedule, inference_steps=self.inference_steps,
                                          inf_sched_alpha=self.rot_inf_sched_alpha,
                                          inf_sched_beta=self.rot_inf_sched_beta)
            tor_schedule = get_t_schedule(sigma_schedule=self.tor_sigma_schedule, inference_steps=self.inference_steps,
                                          inf_sched_alpha=self.tor_inf_sched_alpha,
                                          inf_sched_beta=self.tor_inf_sched_beta)

            sidechain_tor_schedule = get_t_schedule(sigma_schedule=self.args.sidechain_tor_sigma_schedule,
                                                    inference_steps=self.args.inference_steps,
                                                    inf_sched_alpha=self.args.sidechain_tor_inf_sched_alpha,
                                                    inf_sched_beta=self.args.sidechain_tor_inf_sched_beta)

            print('tr schedule', tr_schedule)
            print('rot schedule', rot_schedule)
            print('tor schedule', tor_schedule)
            print('sidechain_tor_schedule', sidechain_tor_schedule)

        else:
            rot_schedule = tr_schedule
            tor_schedule = tr_schedule
            sidechain_tor_schedule = tr_schedule
            print('common t schedule', tr_schedule)

        print('HAPPENING | loading cached complexes of the original model to create the filtering dataset RMSDs and '
              'predicted positions. Doing that from: ', os.path.join(original_model_cache, "heterographs.pkl"))
        if not os.path.exists(os.path.join(original_model_cache, "heterographs.pkl")):
            print(f'HAPPENING | Complex graphs path does not exist yet: {os.path.join(original_model_cache, "heterographs.pkl")}. For that reason, we are now creating the dataset.')
            dataset = PDBBind(transform=None, root=self.args.data_dir, limit_complexes=self.args.limit_complexes,
                                chain_cutoff=self.args.chain_cutoff,
                                receptor_radius=self.original_model_args.receptor_radius,
                                cache_path=self.args.cache_path, split_path=self.args.split_val if self.split == 'val' else self.args.split_train,
                                remove_hs=self.original_model_args.remove_hs, max_lig_size=None,
                                c_alpha_max_neighbors=self.original_model_args.c_alpha_max_neighbors,
                                matching=not self.original_model_args.no_torsion, keep_original=True,
                                popsize=self.original_model_args.matching_popsize,
                                maxiter=self.original_model_args.matching_maxiter,
                                all_atoms=self.original_model_args.all_atoms,
                                esm_embeddings_path=self.args.esm_embeddings_path,
                                require_ligand=True,
                                num_workers=self.args.num_workers,
                                protein_file=self.args.protein_file,match_protein_file=self.args.match_protein_file,
                                conformer_match_sidechains=self.args.conformer_match_sidechains,
                                conformer_match_score=self.args.conformer_match_score,
                                compare_true_protein=self.args.compare_true_protein,
                                match_max_rmsd=self.args.match_max_rmsd,
                                use_original_conformer=self.args.use_original_conformer,
                                use_original_conformer_fallback=self.args.use_original_conformer_fallback,
                                pocket_reduction=self.args.pocket_reduction, pocket_buffer=self.args.pocket_buffer,
                                pocket_cutoff=self.args.pocket_cutoff, skip_no_pocket_atoms=self.args.skip_no_pocket_atoms,
                                pocket_reduction_mode=self.args.pocket_reduction_mode,
                                flexible_sidechains=self.args.flexible_sidechains, flexdist=self.args.flexdist, flexdist_distance_metric=self.args.flexdist_distance_metric,
                              fixed_knn_radius_graph=False if not hasattr(self.args, 'not_fixed_knn_radius_graph') else not self.args.not_fixed_knn_radius_graph,
                              knn_only_graph=False if not hasattr(self.args, 'not_knn_only_graph') else not self.args.not_knn_only_graph,
                              include_miscellaneous_atoms= False if not hasattr(self.args,'include_miscellaneous_atoms') else self.args.include_miscellaneous_atoms,
                              use_old_wrong_embedding_order = True if not hasattr(self.args,'use_old_wrong_embedding_order') else self.args.use_old_wrong_embedding_order
            )
            complex_graphs = dataset.complex_graphs
        else:
            with open(os.path.join(original_model_cache, "heterographs.pkl"), 'rb') as f:
                complex_graphs = pickle.load(f)
        dataset = ListDataset(complex_graphs)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        rmsds, sc_rmsds, full_ligand_positions, full_sc_positions, names, trajectories, sidechain_trajectories = [], [], [], [], [], [], []
        with tqdm(total=len(complex_graphs)) as pbar:
            for idx, orig_complex_graph in tqdm(enumerate(loader), total=len(complex_graphs)):
                # TODO try to get the molecule directly from file without and check same results to avoid any kind of leak
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
                randomize_position(data_list, self.original_model_args.no_torsion, False, self.original_model_args.tr_sigma_max, flexible_sidechains=self.original_model_args.flexible_sidechains)

                predictions_list = None
                failed_convergence_counter = 0
                while predictions_list == None:
                    try:
                        predictions_list, confidences, trajectory, sidechain_trajectory = sampling(data_list=data_list, model=model, inference_steps=self.inference_steps,
                                     tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule, sidechain_tor_schedule=sidechain_tor_schedule,
                                     device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args, return_full_trajectory=True)
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
                if self.original_model_args.no_torsion:
                    orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

                filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

                if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                    orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

                ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
                orig_ligand_pos = np.expand_dims(orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

                rmsds.append(rmsd)
                final_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in predictions_list])
                full_ligand_positions.append(final_pos)
                names.append(orig_complex_graph.name[0])

                trajectory.append(final_pos)
                trajectories.append(np.asarray(trajectory))

                # Check if there are sidechains in the current batch
                if sum([len(c["flexResidues"].subcomponents) for c in data_list]) > 0:
                    # All sidechains have the same indices because they are the same complex graph replicated
                    flex_ids = data_list[0]["flexResidues"].subcomponents.unique().cpu().numpy()

                    target_sidechain_pos = orig_complex_graph['atom'].pos.cpu().numpy()[flex_ids]
                    final_sidechain_pos = np.asarray(
                        [complex_graph['atom'].pos.cpu().numpy()[flex_ids] for complex_graph in predictions_list]
                    )

                    sc_rmsd = np.sqrt(((final_sidechain_pos - target_sidechain_pos) ** 2).sum(axis=2).mean(axis=1))

                    full_sc_positions.append(final_sidechain_pos)
                    sc_rmsds.append(sc_rmsd)
                    sidechain_trajectory.append(final_sidechain_pos)

                elif model.flexible_sidechains:
                    # Sidechains are flexible but there are none in the current batch.
                    # We still add an entry so that the dimensions are consistent with the other trajectories.
                    full_sc_positions.append(np.array([]))
                    sc_rmsds.append(np.zeros_like(rmsd))  # Append a zero rmsd for each sample
                    sidechain_trajectory.append(np.array([]))

                sidechain_trajectories.append(np.asarray(sidechain_trajectory))
                assert(len(orig_complex_graph.name) == 1) # I just put this assert here because of the above line where I assumed that the list is always only lenght 1. Just in case it isn't maybe check what the names in there are.

                if not isinstance(self.rmsd_classification_cutoff, list):
                    successful_docking = np.array(rmsds).flatten() < self.rmsd_classification_cutoff
                    rmsd_lt = (successful_docking.mean() * 100).round(2)
                    desc = f'rmsd: {rmsd_lt}%'

                    if self.original_model_args.flexible_sidechains:
                        if not isinstance(self.sc_rmsd_classification_cutoff, list):
                            successful_sidechains = np.array(sc_rmsds).flatten() < self.sc_rmsd_classification_cutoff
                            sc_rmsd_lt = (successful_sidechains.mean() * 100).round(2)
                            both_lt = ((successful_docking & successful_sidechains).mean() * 100).round(2)

                            desc = f'rmsd: {rmsd_lt}% | sc_rmsd: {sc_rmsd_lt}% | y: {both_lt}%'

                pbar.set_description(desc)
                pbar.update()


        with open(os.path.join(self.full_cache_path, f"ligand_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((full_ligand_positions, rmsds), f)
        with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((names), f)
        with open(os.path.join(self.full_cache_path, f"trajectories{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((trajectories, rmsds), f)
        if model.flexible_sidechains:
            with open(os.path.join(self.full_cache_path, f"sc_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
                pickle.dump((full_sc_positions, sc_rmsds), f)
            with open(os.path.join(self.full_cache_path, f"sidechain_trajectories{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
                pickle.dump((sidechain_trajectories), f)


