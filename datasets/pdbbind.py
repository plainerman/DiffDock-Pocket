import functools
import logging
import operator
import sys
import time
import traceback
import binascii
import os
import pickle
import random
import copy
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import Bio
from rdkit.Chem import MolFromSmiles, AddHs, Mol
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from datasets.process_mols import read_molecule, get_rec_graph, \
    get_lig_graph_with_matching, extract_receptor_structure, parse_receptor_structure, parse_pdb_from_path, \
    generate_conformer, get_sidechain_rotation_masks, set_sidechain_rotation_masks, count_pdb_warnings
from datasets.sidechain_conformer_matching import optimize_rotatable_bonds, RMSD
from utils.diffusion_utils import modify_conformer, get_inverse_schedule, set_time, modify_sidechains
from utils.utils import read_strings_from_txt, get_available_devices, \
    to_none, center_to_torch, ensure_device
from utils import so3, torus
import utils.esm as esm_utils

from datasets.steric_clash import get_steric_clash_atom_pairs, get_rec_elements, get_ligand_elements, \
    get_steric_clash_per_flexble_sidechain_atom


class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom, alpha=1,beta=1, rot_alpha=1, rot_beta=1, tor_alpha=1, tor_beta=1, sidechain_tor_alpha=1,sidechain_tor_beta=1,flexible_sidechains=False,separate_noise_schedule=False, asyncronous_noise_schedule=False,  include_miscellaneous_atoms=False):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom
        self.flexible_sidechains = flexible_sidechains
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.separate_noise_schedule = separate_noise_schedule
        self.asyncronous_noise_schedule = asyncronous_noise_schedule
        self.alpha = alpha
        self.rot_alpha = rot_alpha
        self.tor_alpha = tor_alpha
        self.beta = beta
        self.rot_beta = rot_beta
        self.tor_beta = tor_beta
        self.sidechain_tor_alpha = sidechain_tor_alpha
        self.sidechain_tor_beta = sidechain_tor_beta

    def __call__(self, data):
        t_tr, t_rot, t_tor, t_sidechain_tor, t = self.get_time()
        complex_graph = self.apply_noise(data["complex_graph"], t_tr, t_rot, t_tor, t_sidechain_tor, t)
        return complex_graph

    def get_time(self):
        if self.separate_noise_schedule:
            t = None
            t_tr = np.random.beta(self.alpha, self.beta)
            t_rot = np.random.beta(self.rot_alpha, self.rot_beta)
            t_tor = np.random.beta(self.tor_alpha, self.tor_beta)
            t_sidechain_tor = np.random.beta(self.sidechain_tor_alpha, self.sidechain_tor_beta)
        elif self.asyncronous_noise_schedule:
            t = np.random.uniform(0, 1)
            t_tr = get_inverse_schedule(t, self.alpha, self.beta)
            t_rot = get_inverse_schedule(t, self.rot_alpha, self.rot_beta)
            t_tor = get_inverse_schedule(t, self.tor_alpha, self.tor_beta)
            t_sidechain_tor = get_inverse_schedule(t, self.sidechain_tor_alpha, self.sidechain_tor_beta)
        else:
            t = np.random.beta(self.alpha, self.beta)
            t_tr, t_rot, t_tor, t_sidechain_tor = t, t, t, t
        return t_tr, t_rot, t_tor, t_sidechain_tor, t

    def apply_noise(self, data, t_tr, t_rot, t_tor, t_sidechain_tor, t, tr_update = None, rot_update=None, torsion_updates=None,sidechain_torsion_updates=None):
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        tr_sigma, rot_sigma, tor_sigma, sidechain_tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor, t_sidechain_tor)
        set_time(data, t, t_tr, t_rot, t_tor, t_sidechain_tor, 1, self.all_atom, self.asyncronous_noise_schedule, device=None, include_miscellaneous_atoms=self.include_miscellaneous_atoms)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
        torsion_updates = None if self.no_torsion else torsion_updates
        modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)

        # apply sidechain torsion noise if flexible sidechains exist
        if self.flexible_sidechains:
            sidechain_torsion_updates = np.random.normal(loc=0.0, scale=sidechain_tor_sigma, size=len(data['flexResidues'].edge_idx)) if sidechain_torsion_updates is None else sidechain_torsion_updates
            modify_sidechains(data, sidechain_torsion_updates)
        else:
            sidechain_torsion_updates = None 

        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma
        data.sidechain_tor_score = None if not self.flexible_sidechains else torch.from_numpy(torus.score(sidechain_torsion_updates, sidechain_tor_sigma)).float() 
        data.sidechain_tor_sigma_edge = None if not self.flexible_sidechains else np.ones(len(sidechain_torsion_updates)) * sidechain_tor_sigma
        return data


SORTING_DICT = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],  
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],  
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],  
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],  
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"], 
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}


class PDBBind(Dataset):
    def __init__(self, protein_ligand_df=None,
                 transform=None, cache_path='data/cache', split_path=None, limit_complexes=0, multiplicity=1, chain_cutoff=10,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 require_ligand=False,
                 full_dataset=True, cross_docking=False, use_full_size_protein_file=False, use_original_protein_file = False,
                 keep_local_structures=True, protein_file="protein_processed",
                 match_protein_file="protein_processed_fix", conformer_match_sidechains=False, conformer_match_score="dist", compare_true_protein=False,
                 match_max_rmsd=None, use_original_conformer=False, use_original_conformer_fallback=False,
                 pocket_reduction=False, pocket_buffer=10,
                 pocket_cutoff=5, skip_no_pocket_atoms=False,
                 pocket_reduction_mode='center-dist',
                 flexible_sidechains=False, flexdist=3.5,flexdist_distance_metric = "L2",
                 knn_only_graph=False, fixed_knn_radius_graph=False, include_miscellaneous_atoms=False, use_old_wrong_embedding_order=False,
                 cross_docking_testset = False):

        super(PDBBind, self).__init__(root=None, transform=transform)

        #############################
        # These are the arrays of data; protein path, ligand info, esm_embeddings.
        # If we want to keep data in sync, need to keep track of these.
        # self.ligand_descriptions = ligand_descriptions
        # self.protein_path_list = protein_path_list
        # Note: Should decide whether this takes a path or a list, and name accordingly
        # self.esm_embeddings_path = esm_embeddings_path
        assert protein_ligand_df is not None, "Must provide a Protein-Ligand dataframe"
        self.protein_ligand_df: pd.DataFrame = protein_ligand_df
        #############################

        # self.pdbbind_dir = ""
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.multiplicity = multiplicity
        self.chain_cutoff = chain_cutoff
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs

        self.use_old_wrong_embedding_order = use_old_wrong_embedding_order
        self.use_original_protein_file = use_original_protein_file
        self.require_ligand = require_ligand
        self.pocket_reduction = pocket_reduction
        self.pocket_buffer = pocket_buffer
        self.pocket_cutoff = pocket_cutoff
        self.skip_no_pocket_atoms = skip_no_pocket_atoms
        self.pocket_reduction_mode = pocket_reduction_mode

        self.cross_docking = cross_docking
        self.use_full_size_protein_file = use_full_size_protein_file

        self.keep_local_structures = keep_local_structures
        self.protein_file = protein_file
        self.match_protein_file = match_protein_file
        self.conformer_match_sidechains = conformer_match_sidechains
        self.conformer_match_score = conformer_match_score
        self.compare_true_protein = compare_true_protein
        self.match_max_rmsd = match_max_rmsd
        self.use_original_conformer = use_original_conformer
        self.use_original_conformer_fallback = use_original_conformer_fallback
        self.fixed_knn_radius_graph = fixed_knn_radius_graph
        self.knn_only_graph = knn_only_graph
        self.flexible_sidechains = flexible_sidechains
        self.flexdist = flexdist 
        self.flexdist_distance_metric = flexdist_distance_metric
        self.all_atoms = all_atoms
        self.cross_docking_testset = cross_docking_testset

        if self.compare_true_protein and self.use_original_conformer_fallback:
            raise NotImplementedError("compare_true_protein and use_original_conformer_fallback cannot be used together")

        # if self.use_original_conformer_fallback and not self.conformer_match_sidechains:
        #    raise NotImplementedError("use_original_conformer_fallback requires conformer_match_sidechains")

        def _get_protein_ligand_hash():
            return str(binascii.crc32(''.join(str(self.ligand_path_list + self.protein_path_list)).encode()))

        if matching:
            cache_path += '_torsion'
        if all_atoms:
            cache_path += '_allatoms'

        _split_tag = os.path.splitext(os.path.basename(self.split_path))[0] if self.split_path else 'full_dataset'

        self.full_cache_dir = os.path.join(cache_path, f'limit{self.limit_complexes}'
                                                        f'_INDEX{_split_tag}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                                        f'_chainCutoff{self.chain_cutoff}'
                                           + ('' if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                           + ('' if not full_dataset else f'_full')
                                           + ('' if not cross_docking else f'_cross')
                                           + ('' if not use_original_protein_file else f'_origProt')
                                           + ('' if not keep_local_structures else f'_keptLocalStruct')
                                           + (_get_protein_ligand_hash())
                                           + ('' if protein_file == "protein_processed" else '_' + protein_file)
                                           + ('' if match_protein_file == "protein_processed_fix" else '_' + match_protein_file)
                                           + ('' if not use_original_conformer_fallback else '_fallback')
                                           + ('' if not conformer_match_sidechains else '_match' + ((('' if match_max_rmsd is None else str(match_max_rmsd))) + ('' if not use_original_conformer else 'with_orig')))
                                           + ('' if not conformer_match_sidechains else ('' if conformer_match_score == 'dist' else '_score' + conformer_match_score))
                                           + ('' if not compare_true_protein else '_compare')
                                           + ('' if not pocket_reduction else '_reduced' + str(pocket_buffer))
                                           + ('' if not pocket_reduction else ('' if pocket_reduction_mode == 'center-dist' else '_' + str(pocket_reduction_mode)))
                                           + ('' if not pocket_reduction else ('' if pocket_cutoff == 5.0 else '_' + str(pocket_cutoff)))
                                           + ('' if not pocket_reduction else ('skip' if skip_no_pocket_atoms else ''))
                                           + ('' if not flexible_sidechains else '_flexible' + str(flexdist) + flexdist_distance_metric)
                                           + ('' if not self.fixed_knn_radius_graph else (f'_fixedKNN' if not self.knn_only_graph else '_fixedKNNonly'))
                                           + ('' if not self.include_miscellaneous_atoms else '_miscAtoms')
                                           + ('' if self.use_old_wrong_embedding_order else '_chainOrd')
                                           + ('' if not self.cross_docking_testset else "_crossTest"))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers

        if not os.path.exists(self.full_cache_file_path):
            logging.info(f"Cache file not found, reprocessing data. Saving to {self.full_cache_file_path}")
            os.makedirs(self.full_cache_dir, exist_ok=True)

            # Assume that "split_path" is a path to a file containing a list of complex names (eg PDB IDs)
            # Format of the file is one complex name per line
            # We use these to filter the protein_ligand_df, column "complex_name"
            if self.split_path:
                complex_names = read_strings_from_txt(self.split_path)
                self.protein_ligand_df = protein_ligand_df[self.protein_ligand_df["complex_name"].isin(complex_names)].copy(deep=False)
                self.protein_ligand_df.reset_index(inplace=True, drop=True)
                logging.info(f"Kept {self.protein_ligand_df.shape[0]} rows after filtering by {self.split_path}")

            self.preprocessing()

        logging.info('Loading data from memory: ', self.full_cache_file_path)
        with open(self.full_cache_file_path, 'rb') as f:
            self.protein_ligand_df = pickle.load(f)

        self._indices = None

        if require_ligand:
            missing_count = 0
            for ind, row in self.protein_ligand_df.iterrows():
                ligs = row["rdkit_ligand"]
                if not ligs:
                    print(f"{ind} Ligand not found at {row['experimental_protein']}, {row['ligand_path']}")
                    missing_count += 1
            if missing_count > 0:
                raise ValueError(f"Missing {missing_count} ligands")

        print_statistics(self.protein_ligand_df["complex_graph"].to_list())

    def len(self):
        return len(self.protein_ligand_df)

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        # We want to return a PDBBind object,
        # but with only the selected region of the data.
        obj = super().__getitem__(idx)

        # If `idx` is a single value, the transform will be applied,
        # and the result will be a HeteroData object.
        # If `idx` has multiple values, we are selecting a slice of the dataset,
        # and we want to return a PDBBind object containing that slice.
        if hasattr(obj, "protein_ligand_df"):
            obj.protein_ligand_df = self.protein_ligand_df.iloc[idx, :].copy(deep=False)
            obj.protein_ligand_df.reset_index(inplace=True, drop=True)
            obj._indices = obj.protein_ligand_df.index.values.tolist()

        return obj

    def get(self, idx):
        item = self.protein_ligand_df.iloc[idx, :]
        # complex_graph = item["complex_graph"]
        return item

    def have_esm_embeddings(self):
        return ("esm_embeddings" in self.protein_ligand_df.columns and
                self.protein_ligand_df["esm_embeddings"].iloc[0])

    @property
    def full_cache_file_path(self):
        return os.path.join(self.full_cache_dir, "protein_ligand_df.pkl")

    def complex_cache_file_path(self, ind, complex_name):
        return os.path.join(self.full_cache_dir, f"{ind}_{complex_name}.pkl")

    @staticmethod
    def _calculate_binding_pocket(receptor, ligand, buffer, pocket_cutoff, skip_no_pocket_atoms=False) -> (torch.tensor, float):
        d = torch.cdist(receptor, ligand)
        label = torch.any(d < pocket_cutoff, axis=1)
        if label.any():
            center_pocket = receptor[label].mean(axis=0)
        else:
            if skip_no_pocket_atoms:
                raise NoAtomCloseToLigandException(pocket_cutoff)

            print("No pocket residue below minimum distance ", pocket_cutoff, "taking closest at", d.min())
            center_pocket = receptor[d.min(axis=1)[0].argmin()]

        # TODO: train a model that uses a sphere around the ligand, and not the distance to the pocket? maybe better testset performance
        radius_pocket = torch.linalg.norm(ligand - center_pocket[None, :], axis=1)

        return center_pocket, radius_pocket.max() + buffer  # add a buffer around the binding pocket

    @staticmethod
    def _get_flexdist_cutoff_func(rec, ligand, flexdist, mode, pocket_cutoff):  # mode can be either "L2 or prism"
        if mode == "L2":
            # compute distance cutoff with l2 distance 
            pocket_center, pocket_radius_buffered = PDBBind._calculate_binding_pocket(rec, ligand, flexdist, pocket_cutoff)
            def L2_distance_metric(atom:Bio.PDB.Atom.Atom):
                return torch.linalg.vector_norm(torch.tensor(atom.coord)-pocket_center.squeeze()) <= pocket_radius_buffered

            return L2_distance_metric
        
        elif mode == "prism":
            xMin, yMin, zMin = torch.min(ligand, dim=0).values - flexdist
            xMax, yMax, zMax = torch.max(ligand, dim=0).values + flexdist
            def prism_distance_metric(atom:Bio.PDB.Atom.Atom):
                atom_coord = torch.tensor(atom.coord)
                if (xMin <= atom_coord[0] <= xMax) * (yMin <= atom_coord[1] <= yMax) * (zMin <= atom_coord[2] <= zMax):
                    # check distance to ligand atoms akin to gnina, valid as hydrogens are removed during graph construction
                    return torch.any(torch.linalg.vector_norm(ligand - atom_coord, ord=2, dim=1) < flexdist)
                else: 
                    return False 
            return prism_distance_metric
        else:
            raise NotImplementedError(f"The distancec metric {mode} is not implemented.")

    @staticmethod
    def order_atoms_in_residue(res, atom):
        """
        An order function that sorts atoms of a residue.
        Atoms N, CA, C, O always come first, thereafter the rest of the atoms are sorted according
        to how they appear in the chemical components. Hydrogens come last and are not sorted.
        """

        if atom.name == "OXT":
            return 999
        elif atom.element == "H":
            return 1000

        if res.resname in SORTING_DICT:
            if atom.name in SORTING_DICT[res.resname]:
                return SORTING_DICT[res.resname].index(atom.name)
        else:
            raise Exception("Unknown residue", res.resname)
        raise Exception(f"Could not find atom {atom.name} in {res.resname}")

    @property
    def protein_path_list(self):
        return self.protein_ligand_df["experimental_protein"].to_list()

    @property
    def ligand_path_list(self):
        return self.protein_ligand_df["ligand_path"].to_list()

    def _get_complex_from_row(self, ind, row, lm_embedding_chains):
        cur_graph, cur_ligand = self.get_complex(row["experimental_protein"], lm_embedding_chains,
                                                 row["mol"], row["ligand_path"],
                                                 lig_center=row["pocket_center"],
                                                 predefined_flexible_sidechains=row.get("flexible_sidechains", None),
                                                 comp_protein_path=row.get("computational_protein", None))
        return {"ind": ind, "complex_graph": cur_graph, "rdkit_ligand": cur_ligand}

    def process_dataframe_piece(self, df, device=None) -> List[str]:
        """
        Process a dataframe (whole or part).
        We save the results for each row to disk, and return a list of paths to the saved files.
        This is for more granular caching and to avoid too many IPC file handles.
        """
        print(f"Computing ESM embeddings for {len(df)} proteins...")
        esm_embeddings = esm_utils.esm_embeddings_from_complexes(df["complex_name"],
                                                                 df["experimental_protein"],
                                                                 device=device)
        complex_paths = []
        rn = 0
        with tqdm(total=len(df), desc='Loading complexes') as pbar:
            for ind, row in df.iterrows():
                complex_path = self.complex_cache_file_path(ind, row["complex_name"])
                if not os.path.exists(complex_path):
                    result = self._get_complex_from_row(ind, row, esm_embeddings[rn])
                    with open(complex_path, 'wb') as f:
                        pickle.dump(result, f)

                complex_paths.append(complex_path)
                rn += 1
                pbar.update()
        return complex_paths

    def process_dataframe_in_parallel(self, df, devices=(None,)):
        processes = len(devices)
        df_pieces = np.array_split(df, processes)
        args = [(df_piece, device) for df_piece, device in zip(df_pieces, devices)]
        with torch.multiprocessing.Pool(processes=processes) as pool:
            complex_paths = pool.starmap(self.process_dataframe_piece, args)

        complex_paths = functools.reduce(operator.iconcat, complex_paths, [])
        return complex_paths

    def preprocessing(self):
        print('Data Preprocessing. Reading molecules and generating local structures with RDKit')
        self.protein_ligand_df["mol"] = None
        for ind, row in tqdm(self.protein_ligand_df.iterrows()):
            ligand_path = row["ligand_path"]
            mol = read_molecule(ligand_path, remove_hs=False)
            if not self.keep_local_structures:
                # Note: This fails sometimes.
                mol.RemoveAllConformers()
                mol = AddHs(mol)
                generate_conformer(mol)

            self.protein_ligand_df.at[ind, "mol"] = mol

        print('Generating graphs for ligands and proteins')
        self.protein_ligand_df["complex_graph"] = None
        self.protein_ligand_df["rdkit_ligand"] = None

        counter = 0
        if self.num_workers > 1:
            devices = get_available_devices(self.num_workers)
            print(f"Processing inputs with {len(devices)} workers")
            complex_paths = self.process_dataframe_in_parallel(self.protein_ligand_df, devices)
        else:
            complex_paths = self.process_dataframe_piece(self.protein_ligand_df)

        for complex_path in complex_paths:
            result = pickle.load(open(complex_path, 'rb'))
            ind, cur_graph, cur_ligand = result["ind"], result["complex_graph"], result["rdkit_ligand"]
            if cur_graph and cur_ligand:
                cur_graph.mol = cur_ligand
                counter += 1

                self.protein_ligand_df.at[ind, "complex_graph"] = cur_graph
                self.protein_ligand_df.at[ind, "rdkit_ligand"] = cur_ligand

        num_expected = len(self)
        keep_rows = self.protein_ligand_df["complex_graph"].notna() & self.protein_ligand_df["rdkit_ligand"].notna()
        self.protein_ligand_df = self.protein_ligand_df[keep_rows.values]
        print(f"Saving {self.protein_ligand_df.shape[0]} / {num_expected} processed complexes & ligands to cache")

        with open(self.full_cache_file_path, 'wb') as f:
            pickle.dump(self.protein_ligand_df, f)

    def get_complex(self, exp_protein_path: str, lm_embedding_chains: List, ligand: Mol, ligand_path: str,
                    lig_center: Optional[torch.tensor] = None, predefined_flexible_sidechains: bool = False,
                    comp_protein_path: str = None):
        if not os.path.exists(exp_protein_path):
            raise ValueError(f"File {exp_protein_path} does not exist")

        experimental_receptor = None   # The HOLO / PDB structure     (aka "rec_model_match")
        computational_receptor = None  # The APO / ESM-fold structure (aka "rec_model")
        # If a computational model is available, we align it to the experimental model via conformer matching,
        # and that is used as the protein structure.
        # If not (and self.use_original_conformer_fallback), we use the experimental model as the protein structure.
        true_protein_matching = None

        # rec_parser = parse_cross_receptor if self.cross_docking else parse_crosstest_receptor if self.cross_docking_testset else parse_receptor
        try:
            experimental_receptor = parse_pdb_from_path(exp_protein_path)
            computational_receptor = parse_pdb_from_path(comp_protein_path) if comp_protein_path else None
            complex_name = f'{os.path.basename(exp_protein_path)}___{os.path.basename(ligand_path)}'
            lig, lig_center, ligs_predefined_flexible_sidechains = ligand, lig_center, predefined_flexible_sidechains

            if computational_receptor is None:
                msg = f"Computational structure not found for {complex_name}"
                if self.use_original_conformer_fallback:
                    # This is expected for inference
                    logging.debug(f"Warning: {msg}")
                elif self.conformer_match_sidechains:
                    raise ValueError(f"{msg}\n"
                                     f"Cannot perform conformer_match_sidechains without a computational structure.\n"
                                     f"Run with --use_original_conformer_fallback to use the experimental structure only.\n")

            # IMPORTANT: The indices between experimental_receptor and computational_receptor are not a 1:1 mapping
            # So we sort the atoms by element name, such that they are equal

            def _sort_atoms_by_element(_protein):
                for res in _protein.get_residues():
                    res.child_list.sort(key=lambda atom: PDBBind.order_atoms_in_residue(res, atom))

            def _remove_hs(_protein):
                for res in _protein.get_residues():
                    atoms_to_remove = []
                    for atom in res:
                        if atom.element == 'H':
                            atoms_to_remove.append(atom)
                    for atom in atoms_to_remove:
                        res.detach_child(atom.id)

            remove_hs_and_sort = self.conformer_match_sidechains or self.compare_true_protein or self.flexible_sidechains
            if remove_hs_and_sort:
                _remove_hs(experimental_receptor)
                _sort_atoms_by_element(experimental_receptor)

            if computational_receptor is not None and remove_hs_and_sort:
                # In the case that we are flexible (or conformer matching) we sort the atoms,
                # so that we can compare them in the inference script
                _remove_hs(computational_receptor)
                _sort_atoms_by_element(computational_receptor)

            if computational_receptor is not None and experimental_receptor is not None:
                len_comp = len(list(computational_receptor.get_atoms()))
                len_exp = len(list(experimental_receptor.get_atoms()))
                assert len_comp == len_exp, \
                    (f"The length of the experimental structure ({exp_protein_path}, {len_exp}) "
                     f"does not match the length of the computational structure ({comp_protein_path}, {len_comp})")

                # Check if we have 100% atom identity (hydrogens were ignored in loading already)
                assert [a.name for a in computational_receptor.get_atoms()] == [a.name for a in experimental_receptor.get_atoms()], \
                    "The proteins do not have 100% sequence identity (excluding hydrogens)"

            # Create a mapping that tells us for each computational-model residue
            # the corresponding experimental-model residue
            if self.compare_true_protein:
                true_protein_res = list(experimental_receptor.get_residues())
                true_protein_matching = {(res.parent.id, res.id[1]): true_protein_res[i] for i, res in
                                         enumerate(computational_receptor.get_residues())}

        except Exception as e:
            print(f'Skipping {exp_protein_path} because of the error:')
            print(e)
            print(traceback.format_exc())
            return None, None

        if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
            print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. '
                  f'Not including {exp_protein_path} in preprocessed data.')
            return None, None

        try:
            complex_graph = HeteroData()
            complex_graph['name'] = complex_name
            get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching,
                                        self.keep_original, self.num_conformers, remove_hs=self.remove_hs)

            # use the c-alpha atoms to define the pocket
            if self.conformer_match_sidechains or computational_receptor is None:
                # use the holo structure to define the pocket
                pocket_model = experimental_receptor
            else:
                pocket_model = computational_receptor

            assert pocket_model is not None, "No pocket model found"
            rec_atoms_for_pocket = torch.tensor(
                np.array([a.coord for a in pocket_model.get_atoms() if a.name == 'CA']),
                dtype=complex_graph['ligand'].pos.dtype)

            selector = None
            if self.pocket_reduction or self.conformer_match_sidechains:
                if lig_center is not None:  # change to predefined pocket if any
                    pocket_center = lig_center
                    molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0)
                    pocket_radius = torch.max(
                        torch.linalg.vector_norm(complex_graph['ligand'].pos - molecule_center.unsqueeze(0), dim=1))
                else:
                    pocket_center, pocket_radius = PDBBind._calculate_binding_pocket(rec_atoms_for_pocket,
                                                                                     complex_graph['ligand'].pos, 0,
                                                                                     pocket_cutoff=self.pocket_cutoff,
                                                                                     skip_no_pocket_atoms=self.skip_no_pocket_atoms)

                pocket_radius_buffered = pocket_radius + self.pocket_buffer

                if self.pocket_reduction_mode == 'center-dist':
                    selector = PocketSelector()
                    selector.pocket = pocket_center.cpu().detach().numpy()
                    selector.radius = pocket_radius_buffered.item()
                    selector.all_atoms = self.all_atoms
                elif self.pocket_reduction_mode == 'ligand-dist':
                    selector = AnyHeavyAtomCloseToAnyLigandAtomSelector()
                    selector.ligand = complex_graph['ligand'].pos.cpu()
                    selector.radius = 12
                else:
                    raise NotImplementedError(f'Pocket mode {self.pocket_reduction_mode} is not implemented.')

            conformer_matched = False
            if computational_receptor is not None:
                if self.conformer_match_sidechains:
                    # 1) check whether the two specified structures are similar enough
                    # we use the holo structure to define the pocket
                    # then we keep the atoms based on the holo structure
                    # we keep identical atoms in the apo structure
                    idxs = np.array([selector.accept_residue(a.parent) for a in computational_receptor.get_atoms()])
                    comp_atoms = np.array([a.coord for a in computational_receptor.get_atoms()])
                    exp_atoms = np.array([a.coord for a in experimental_receptor.get_atoms()])
                    rmsd = RMSD(idxs, comp_atoms, exp_atoms)
                    complex_graph.match_rmsd = rmsd
                    if self.match_max_rmsd is not None and rmsd > self.match_max_rmsd:
                        if self.use_original_conformer:
                            receptor = copy.deepcopy(experimental_receptor)
                        else:
                            raise RMSDTooLarge(rmsd, self.match_max_rmsd)
                    else:
                        # 2) calculate the sidechains
                        print(f"Determining residues that will be conformer matched for {exp_protein_path}:")

                        distance_cutoff_func = PDBBind._get_flexdist_cutoff_func(rec_atoms_for_pocket, complex_graph['ligand'].pos,
                                                                                 self.flexdist, self.flexdist_distance_metric, self.pocket_cutoff)

                        subcomponents, subcomponentsMapping, edge_idx, residueNBondsMapping, pdbIds, true_coords =\
                            get_sidechain_rotation_masks(computational_receptor, distance_cutoff_func, remove_hs=self.remove_hs)

                        # 3) conformer match computational receptor with experimental receptor (ground truth)
                        # use copy.deepcopy to avoid changing the original computational_receptor
                        receptor, sc_conformer_match_rotations, sc_conformer_match_improvements = (
                            optimize_rotatable_bonds(copy.deepcopy(computational_receptor), experimental_receptor, subcomponents, subcomponentsMapping, edge_idx, residueNBondsMapping, complex_graph['ligand'].pos, score=self.conformer_match_score))
                        conformer_matched = True
                else:
                    receptor = copy.deepcopy(computational_receptor)
            elif self.use_original_conformer_fallback:
                receptor = copy.deepcopy(experimental_receptor)
            else:
                raise Exception("No computational receptor found. Use --use_original_conformer_fallback to use the experimental structure.")

            receptor, rec_coords, c_alpha_coords, n_coords, c_coords, misc_coords, misc_features, lm_embeddings = extract_receptor_structure(
                receptor, lig, cutoff=self.chain_cutoff,
                lm_embedding_chains=lm_embedding_chains,
                include_miscellaneous_atoms=self.include_miscellaneous_atoms,
                all_atom=self.all_atoms,
                selector=selector if self.pocket_reduction else None)

            if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                raise ValueError(f'LM embeddings for complex {exp_protein_path} did not have the right length for the protein.')

            if not self.knn_only_graph or not self.fixed_knn_radius_graph:
                raise NotImplementedError('Backwards compatibility has been dropped. We only support knn_only_graph=True and fixed_knn_radius_graph=True.')

            get_rec_graph(receptor, rec_coords, c_alpha_coords, n_coords, c_coords, misc_coords, misc_features,
                          complex_graph,
                          rec_radius=self.receptor_radius,
                          c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                          remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)

            if self.conformer_match_sidechains:
                if conformer_matched:
                    complex_graph.sc_conformer_match_rotations = sc_conformer_match_rotations
                    complex_graph.sc_conformer_match_improvements = sc_conformer_match_improvements
                else:
                    complex_graph.sc_conformer_match_rotations = []
                    complex_graph.sc_conformer_match_improvements = 0

            # select flexible sidechains in receptor
            if self.flexible_sidechains:
                logging.debug(f"Computing flexible residues within radius {self.flexdist} of binding pocket using {self.flexdist_distance_metric} distance metric")

                if self.conformer_match_sidechains and conformer_matched:
                    # we are using the same sidechains we have been using for conformer matching before
                    # use pdbIds to determine the previous flexible sidechains
                    accept_atom_function = lambda atom: (atom.parent.get_full_id()[1], atom.parent.get_full_id()[2], atom.parent.get_full_id()[3][1]) in pdbIds
                elif predefined_flexible_sidechains is not None:
                    predefined_flexible_sidechains = predefined_flexible_sidechains.split('-')
                    accept_atom_function = lambda atom: f"{atom.parent.get_full_id()[2]}:{atom.parent.get_full_id()[3][1]}" in predefined_flexible_sidechains
                else:
                    accept_atom_function = PDBBind._get_flexdist_cutoff_func(rec_atoms_for_pocket, complex_graph['ligand'].pos,
                                                                             self.flexdist, self.flexdist_distance_metric,
                                                                             self.pocket_cutoff)
                if self.compare_true_protein:
                    complex_graph = set_sidechain_rotation_masks(complex_graph, receptor, accept_atom_function, remove_hs=self.remove_hs,
                                                                 true_model=experimental_receptor, true_model_matching=true_protein_matching)
                else:
                    complex_graph = set_sidechain_rotation_masks(complex_graph, receptor, accept_atom_function, remove_hs=self.remove_hs)

        except Exception as e:
            print(f'Skipping {exp_protein_path} because of the error: {e}')
            if not isinstance(e, ProcessingException):
                print(traceback.format_exc())
            return None, None

        if self.pocket_reduction:
            protein_center = pocket_center[None, :]
        else:
            # Center the protein around the mean C-alpha position
            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)

        complex_graph = self.center_complex(complex_graph, protein_center)

        return complex_graph, lig

    def center_complex(self, complex_graph, protein_center):
        # Center the protein around the specified pos
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        if self.compare_true_protein:
            complex_graph["flexResidues"].true_sc_pos -= protein_center

        if (not self.matching) or self.num_conformers == 1:
            complex_graph['ligand'].pos -= protein_center
        else:
            for p in complex_graph['ligand'].pos:
                p -= protein_center

        complex_graph.original_center = protein_center

        return complex_graph

    def save_receptor_pdb(self, complex_graph, out):
        # TODO: is this custom structure parser really necessary? Or can we create a new structure?
        rec_structure = parse_receptor_structure(complex_graph["name"], self.pdbbind_dir, self.use_full_size_protein_file, self.use_original_protein_file, self.protein_file)

        # In case the pocket has been reduced this will be approximately 0
        pocket, radius = PDBBind._calculate_binding_pocket(complex_graph['receptor'].pos, complex_graph['ligand'].pos,
                                                           self.pocket_buffer, self.pocket_cutoff)

        from Bio.PDB import PDBIO
        io = PDBIO()
        io.set_structure(rec_structure)

        if self.pocket_reduction_mode == 'center-dist':
            selector = PocketSelector()
            selector.pocket = (pocket + complex_graph.original_center).cpu().numpy()
            selector.radius = radius.item()
            selector.all_atoms = self.all_atoms
        else:
            raise NotImplementedError(f'Pocket mode {self.pocket_reduction_mode} is not implemented.')

        if self.pocket_reduction:
            io.save(out, select=selector)
        else:
            io.save(out)


class ProcessingException(Exception):
    def __init__(self, message):
        super().__init__(message)


class RMSDTooLarge(ProcessingException):
    def __init__(self, rmsd, max_rmsd):
        super().__init__(f"RMSD between the two specified structures is too large. rmsd = {rmsd} > {max_rmsd} = max rmsd")


class NoAtomCloseToLigandException(ProcessingException):
    def __init__(self, pocket_cutoff):
        super().__init__(f"No pocket residue below minimum distance {pocket_cutoff}. "
                         f"Skipping complex because skip_no_pocket_atoms is set to True")


class PocketSelector(Bio.PDB.Select):
    def __int__(self, pocket, radius, all_atoms):
        self.pocket = pocket
        self.radius = radius
        self.all_atoms = all_atoms

    def accept_residue(self, residue):
        if self.all_atoms:
            return (np.linalg.norm(np.array([a.coord for a in residue.child_list]) - self.pocket, axis=1) < self.radius).any()
        return np.linalg.norm(residue.child_dict["CA"].coord - self.pocket) < self.radius


class AnyHeavyAtomCloseToAnyLigandAtomSelector(Bio.PDB.Select):
    def accept_residue(self, residue):
        return (torch.cdist(torch.tensor(np.array([a.coord for a in residue.child_list if a.element != 'H']), device=self.ligand.device), self.ligand) < self.radius).any()


def print_statistics(complex_graphs):
    statistics = ([], [], [], [], [], [], [], [], [], [], [])
    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching',
            'sidechain conformer match rotations', 'sidechain_conformer_match_improvements',
            'rec_lig_steric_clashes', 'rec_sc_lig_steric_clashes', 'rec_sc_rec_rest_steric_clashes', 'rec_sc_rec_sc_steric_clashes',
            'match_rmsd']

    complex_names = []
    fail_count = 0
    for ind, complex_graph in tqdm(enumerate(complex_graphs[:])):
        # print(f"Complex graph: {complex_graph}")
        try:
            if 'flexResidues' in complex_graph.node_types and not hasattr(complex_graph['flexResidues'],'edge_idx'):
                complex_graphs.remove(complex_graph)
                continue
            lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
            if (torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1).nelement()==0):
                complex_graphs.remove(complex_graph)
                continue
            radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
            molecule_center = torch.mean(lig_pos, dim=0)
            radius_molecule = torch.max(
                torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
            distance_center = torch.linalg.vector_norm(molecule_center)
            statistics[0].append(radius_protein)
            statistics[1].append(radius_molecule)
            statistics[2].append(distance_center)
            if "rmsd_matching" in complex_graph:
                statistics[3].append(complex_graph.rmsd_matching)
            else:
                statistics[3].append(0)

            if "sc_conformer_match_rotations" in complex_graph:
                for m in complex_graph.sc_conformer_match_rotations:
                    statistics[4].extend(m.tolist())
                del complex_graph["sc_conformer_match_rotations"]
            else:
                statistics[4].append(0)

            if "sc_conformer_match_improvements" in complex_graph:
                statistics[5].append(complex_graph.sc_conformer_match_improvements)
            else:
                statistics[5].append(0)

            if "atom" in complex_graph:  # check whether we have all atoms
                statistics[6].append(get_steric_clash_atom_pairs(complex_graph["atom"].pos[None, :],
                                                                 complex_graph["ligand"].pos[None, :],
                                                                 get_rec_elements(complex_graph),
                                                                 get_ligand_elements(complex_graph)).sum())
            else:
                statistics[6].append(0)

            if 'flexResidues' in complex_graph.node_types:
                flexidx = torch.unique(complex_graph['flexResidues'].subcomponents).cpu().numpy()
                filterSCHs = flexidx[torch.not_equal(complex_graph['atom'].x[flexidx, 0], 0).cpu().numpy()]

                statistics[7].append(get_steric_clash_atom_pairs(complex_graph["atom"].pos[None, :],
                                                                 complex_graph["ligand"].pos[None, :],
                                                                 get_rec_elements(complex_graph),
                                                                 get_ligand_elements(complex_graph),
                                                                 filter1=filterSCHs).sum())

                statistics[8].append(get_steric_clash_per_flexble_sidechain_atom(complex_graph))
                statistics[9].append(get_steric_clash_per_flexble_sidechain_atom(complex_graph, rec_rest = False))

                if 'match_rmsd' in complex_graph:
                    statistics[10].append(complex_graph.match_rmsd)

                complex_names.append(complex_graph['name'])
            else:
                statistics[7].append(0)
                statistics[8].append(0)
                statistics[9].append(0)
        except Exception as e:
            if complex_graph is not None:
                print('Failed for complex', complex_graph['name'])
            else:
                print(f'Complex graph {ind} is None')

            print(e)
            fail_count += 1
            continue

    # store optimal.npy to see potential how good our model could be
    # used in paper_figures.ipynb
    # np.save('optimal.npy', np.asarray(statistics[5]))

    # store match_rmsd.npy to see the pocket rmsd between protein_file and match_file
    # used in paper_figures.ipynb
    # np.save('match_rmsd.npy', np.asarray(statistics[10]))

    # store the complex names so that we can assign optimal and match_rmsd
    # np.save('complex_names.npy', np.asarray(complex_names))

    print('Number of complexes: ', len(complex_graphs), 'of which', fail_count, 'failed')
    for cur_tat, cur_name in zip(statistics, name):
        array = np.asarray(cur_tat)
        if len(array) == 0:
            print(f"{cur_name}: no data")
        else:
            line = f"{cur_name}: mean {np.mean(array)}, std {np.std(array)}, min {np.min(array)}, max {np.max(array)}, sum {np.sum(array)}"
            if "steric_clash" in cur_name:
                line+=f", percentage: {100*(array>0).sum()/len(array)}"
            print(line)


def construct_loader(args, t_to_sigma):
    transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,flexible_sidechains=args.flexible_sidechains,
                               all_atom=args.all_atoms, alpha=args.sampling_alpha, beta=args.sampling_beta,
                               rot_alpha=args.rot_alpha, rot_beta=args.rot_beta, tor_alpha=args.tor_alpha,
                               tor_beta=args.tor_beta, sidechain_tor_alpha=args.sidechain_tor_alpha,sidechain_tor_beta=args.sidechain_tor_beta,
                               separate_noise_schedule=args.separate_noise_schedule,
                               asyncronous_noise_schedule=args.asyncronous_noise_schedule,
                               include_miscellaneous_atoms=False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms)

    protein_ligand_df = load_protein_ligand_df(args.protein_ligand_csv)

    common_args = {'transform': transform, 'protein_ligand_df': protein_ligand_df,
                   'limit_complexes': args.limit_complexes, 'multiplicity': args.multiplicity,
                   'chain_cutoff': args.chain_cutoff, 'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'full_dataset': not args.not_full_dataset,
                   'keep_local_structures': True,
                   'use_full_size_protein_file': False if not hasattr(args, 'use_full_size_protein_file') else args.use_full_size_protein_file,
                   'protein_file': args.protein_file, 'fixed_knn_radius_graph': False if not hasattr(args, 'not_fixed_knn_radius_graph') else not args.not_fixed_knn_radius_graph,
                   'match_protein_file': args.match_protein_file, 'conformer_match_sidechains': args.conformer_match_sidechains, 'conformer_match_score': args.conformer_match_score,
                   'pocket_reduction': args.pocket_reduction, 'pocket_buffer': args.pocket_buffer,
                   'pocket_cutoff': args.pocket_cutoff, 'skip_no_pocket_atoms': args.skip_no_pocket_atoms,
                   'pocket_reduction_mode': args.pocket_reduction_mode,
                   'flexible_sidechains': args.flexible_sidechains, 'flexdist': args.flexdist, 'flexdist_distance_metric':args.flexdist_distance_metric,
                   'knn_only_graph': False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                   'include_miscellaneous_atoms': False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                   'use_old_wrong_embedding_order': True if not hasattr(args, 'use_old_wrong_embedding_order') else args.use_old_wrong_embedding_order}

    train_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                            num_conformers=args.num_conformers, match_max_rmsd=args.match_max_rmsd,
                            use_original_conformer=args.use_original_conformer, use_original_conformer_fallback=args.use_original_conformer_fallback,
                            **common_args)
    val_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_val, keep_original=True, compare_true_protein=args.compare_true_protein,
                          **common_args)

    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)

    return train_loader, val_loader


def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.sdf'), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return [(lig, None) for lig in ligs]


@count_pdb_warnings
def load_protein_ligand_df(protein_ligand_csv: str, strict: bool = False, max_protein_length: int = 1024,
                           df=None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(protein_ligand_csv)

    keep_inds = []
    missing_or_bad = 0
    if strict:
        protein_path_cols = ["experimental_protein"]
        # We don't need a computational protein column, especially for inference.
        # Only validate the column if it exists.
        if "computational_protein" in df.columns:
            protein_path_cols.append("computational_protein")

        for ind, row in df.iterrows():
            ligand_path = row["ligand"]
            keep_row = True
            if not os.path.exists(ligand_path):
                logging.warning(f"Missing ligand path {ligand_path}. Skipping.")
                keep_row = False
                continue

            for protein_path_col in protein_path_cols:
                protein_path = row[protein_path_col]
                if not os.path.exists(protein_path):
                    logging.warning(f"Missing protein {protein_path}. Skipping.")
                    keep_row = False
                    # Can break out of inner loop
                    break

                cur_protein = esm_utils.get_sequence_simple(protein_path)
                if len(cur_protein) >= max_protein_length:
                    logging.warning(f"Protein {protein_path} is too long ({len(cur_protein)} >= {max_protein_length}). Skipping.")
                    keep_row = False
                    break

            if keep_row:
                keep_inds.append(ind)

        keep_inds = np.array(keep_inds)
        missing_or_bad = len(df) - len(keep_inds)

    logging.info(f"Loaded {len(keep_inds)} / {len(df)} total complexes. Skipped/couldn't find {missing_or_bad}.")

    if len(keep_inds) > 0:
        df = df.iloc[keep_inds, :]

    df = df.rename(columns={"ligand": "ligand_path"})
    cols_to_sanitize = ["pocket_center_x", "pocket_center_y", "pocket_center_z", "flexible_sidechains"]
    for col in cols_to_sanitize:
        if col in df.columns:
            df[col] = to_none(df[col].tolist())

    protein_ligand_df = df.reset_index(drop=True)
    protein_ligand_df["pocket_center"] = None
    if "pocket_center_x" in protein_ligand_df.columns:
        protein_ligand_df.apply(
            lambda row: center_to_torch(row["pocket_center_x"], row["pocket_center_y"], row["pocket_center_z"]), axis=1)

    return protein_ligand_df
