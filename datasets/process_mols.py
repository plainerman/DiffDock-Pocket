import copy
import logging
import os
import warnings
from functools import wraps

import Bio
import numpy as np
import scipy.spatial as spa
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs 
from rdkit.Geometry import Point3D
from scipy import spatial
from scipy.special import softmax
from scipy.sparse import dok_matrix
from sklearn.neighbors import radius_neighbors_graph
from torch import cdist
from torch_cluster import radius_graph, knn_graph

import torch.nn.functional as F

from datasets.conformer_matching import get_torsion_angles, optimize_rotatable_bonds
from utils.torsion import get_transformation_mask,get_sidechain_rotation_mask


biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
    'possible_flexible_sidechains': {'ARG','HIS','LYS','ASP','GLU','SER','THR','ASN','GLN','CYS','SEC','GLY','PRO','ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP'}
}

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 0)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 0)


# build linking for cross docking
#with open('data/PDBBind_PDBid_UniprotId.txt', 'r') as f:
#    id_lists = [s.split('  ') for s in f.readlines()]

pdb_to_uniprod = {}
uniprod_to_pdb_train = {}
#for l in id_lists:
#    pdb_to_uniprod[l[0]] = l[2]
#    if l[2] != '------' and l[1] != '2019':
#        if l[2] in uniprod_to_pdb_train:
#            uniprod_to_pdb_train[l[2]].append(l[0])
#        else:
#            uniprod_to_pdb_train[l[2]] = [l[0]]


def lig_atom_featurizer(mol):
    #ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        #g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            #g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])
    return torch.tensor(atom_features_list)


#sr = ShrakeRupley(probe_radius=1.4,  # in A. Default is 1.40 roughly the radius of a water molecule.
#                  n_points=100)  # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.


def rec_residue_featurizer(rec):
    feature_list = []
    #sr.compute(rec, level="R")
    for residue in rec.get_residues():
        """sasa = residue.sasa
        for atom in residue:
            if atom.name == 'CA':
                bfactor = atom.bfactor
        assert not np.isinf(bfactor)
        assert not np.isnan(bfactor)
        assert not np.isinf(sasa)
        assert not np.isnan(sasa)"""
        feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname()),
                             #sasa, bfactor
                             ])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def parse_cross_receptor(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, conformer_match_sidechains, match_protein_file):
    if conformer_match_sidechains:
        raise NotImplementedError("conformer_match_sidechains not implemented for cross docking")

    uniprodid = pdb_to_uniprod[pdbid]
    to_choose = set(uniprod_to_pdb_train[uniprodid] if uniprodid in uniprod_to_pdb_train else []) - {pdbid}
    if len(to_choose) == 0:
        return None

    orig_model = parsePDB(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file)
    print("original model", pdbid, uniprodid)
    for chain in orig_model:
        for res in chain:
            print(res.resname, end=' ')
        print()

    for cross_pdbid in to_choose: #cross_pdbid = random.choice(list(to_choose))

        cross_model = parsePDB(cross_pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file)

        print("cross model", cross_pdbid, uniprodid)
        for chain in cross_model:
            for res in chain:
                print(res.resname, end=' ')
            print()


    return None, None
    ref_atoms = []
    alt_atoms = []

    for (ref_chain, alt_chain) in zip(orig_model, cross_model):
        for ref_res, alt_res in zip(ref_chain, alt_chain):
            print(ref_res.resname, alt_res.resname)
            assert ref_res.resname == alt_res.resname
            assert ref_res.id == alt_res.id
            ref_atoms.append(ref_res['CA'])
            alt_atoms.append(alt_res['CA'])

    # Align these paired atom lists:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, alt_atoms)
    super_imposer.apply(cross_model.get_atoms())

    return cross_model, None


def parse_receptor(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, protein_file, conformer_match_sidechains, match_protein_file, compare_true_protein):
    try:
        rec = parsePDB(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, protein_file)
    except FileNotFoundError as e:
        # In this case, we allow the protein to be missing, because we will handle it in the dataset
        print(e)
        rec = None
    rec_match = None

    if conformer_match_sidechains or compare_true_protein:
        rec_match = parsePDB(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, match_protein_file)
        # We must remove the hydrogens, otherwise the atoms will not align perfectly
        for chain in rec_match:
            for res in chain:
                atoms_to_remove = []
                for atom in res:
                    if atom.element == 'H':
                        atoms_to_remove.append(atom)
                for atom in atoms_to_remove:
                    res.detach_child(atom.id)

        if rec is not None:
            assert len(list(rec.get_residues())) == len(list(rec_match.get_residues())), \
                            "The residue length of the proteins does not match"

    return rec, rec_match


def parse_crosstest_receptor(line, pdbbind_dir, use_full_size_file, use_original_protein_file, protein_file, conformer_match_sidechains, match_protein_file, compare_true_protein):
    path = line.split(",")
    rec_path = os.path.join(pdbbind_dir, path[0], path[1].split(".")[0] + protein_file)
    print(rec_path)
    return parse_pdb_from_path(rec_path), None


def parse_receptor_structure(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, protein_file):
    rec = parsePDBStructure(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, protein_file)
    return rec


def _get_pdb_rec_path(pdbid, pdbbind_dir,use_full_size_file, use_original_protein_file, protein_file):
    rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_{protein_file}.pdb')
    if protein_file == "protein_processed" and (
            not os.path.exists(rec_path) or use_full_size_file or use_original_protein_file):
        rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein_obabel_reduce.pdb')
        if not os.path.exists(rec_path) or use_original_protein_file:
            rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein.pdb')
    return rec_path


def parsePDB(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, protein_file="protein_processed"):
    rec_path = _get_pdb_rec_path(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, protein_file)
    return parse_pdb_from_path(rec_path)


def parse_pdb_from_path(path):
    return parse_pdb_structure_from_path(path)[0]


def parse_pdb_structure_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure(os.path.basename(path), path)
        return structure


def parsePDBStructure(pdbid, pdbbind_dir,use_full_size_file, use_original_protein_file, protein_file="protein_processed"):
    rec_path = _get_pdb_rec_path(pdbid, pdbbind_dir, use_full_size_file, use_original_protein_file, protein_file)
    return parse_pdb_structure_from_path(rec_path)


def extract_receptor_structure(rec, lig, cutoff, lm_embedding_chains=None, include_miscellaneous_atoms=False, all_atom=False, selector=None):
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    min_distances = []
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    misc_coords = []
    misc_features = []
    valid_chain_ids = []
    all_chain_ids = []
    lengths = []
    discarded_res_ids = {}
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        chain_misc_coords = []
        chain_misc_features = []
        valid_count = 0
        atom_idx = 0
        invalid_res_ids = []
        discarded_res_ids[chain] = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue

            residue_coords = []
            residue_features = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))
                if include_miscellaneous_atoms:
                    residue_features.append(np.array(get_rec_misc_atom_feat(atom)))

            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex

                if selector is None or selector.accept_residue(residue):
                    chain_c_alpha_coords.append(c_alpha)
                    chain_n_coords.append(n)
                    chain_c_coords.append(c)
                    chain_coords.append(np.array(residue_coords))
                    valid_count += 1
                else:
                    discarded_res_ids[chain].append((atom_idx, residue.get_id()))

                atom_idx += 1
            else:
                if include_miscellaneous_atoms:
                    chain_misc_coords.append(np.array(residue_coords))
                    chain_misc_features.append(np.array(residue_features))
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        for _, res_id in discarded_res_ids[chain]:
            chain.detach_child(res_id)

        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf

        # this removes chains if they are not close enough to the ligand
        min_distances.append(min_distance)
        lengths.append(valid_count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        all_chain_ids.append(chain.get_id())
        if min_distance < cutoff:
            valid_chain_ids.append(chain.get_id())
        misc_coords.append(chain_misc_coords)
        misc_features.append(chain_misc_features)
    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(all_chain_ids[np.argmin(min_distances)])
    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    raise ValueError(' Encountered valid chain id that was not present in the LM embeddings')

                mask = torch.ones(len(lm_embedding_chains[i]), dtype=torch.bool, device=lm_embedding_chains[i].device)
                mask[[d[0] for d in discarded_res_ids[chain]]] = 0  # do not use the discarded elements
                valid_lm_embeddings.append(lm_embedding_chains[i][mask])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)
    if include_miscellaneous_atoms:
        flattened_coords = [item for sublist in misc_coords for item in sublist]
        flattened_features = [item for sublist in misc_features for item in sublist]
        if not all_atom: # hack to not have an empty array if we are not using all atoms
            flattened_coords.append(c_coords.mean(axis=0)[None,:])
            flattened_features.append(np.array(get_rec_misc_atom_feat(get_misc_features=True))[None,:])
        if len(flattened_coords) == 0:
            misc_coords_array = np.array([])
            misc_features_array = np.array([])
        else:
            misc_coords_array = np.concatenate(flattened_coords)
            misc_features_array = np.concatenate(flattened_features)
            misc_all_dist = spatial.distance.cdist(misc_coords_array, c_alpha_coords)
            misc_coords_array = misc_coords_array[np.min(misc_all_dist, axis=1) < 10]
            misc_features_array = misc_features_array[np.min(misc_all_dist, axis=1) < 10]
    else:
        misc_coords_array = None
        misc_features_array = None
    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    return rec, coords, c_alpha_coords, n_coords, c_coords, misc_coords_array, misc_features_array, lm_embeddings


def get_lig_graph(mol, complex_graph):
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    atom_feats = lig_atom_featurizer(mol)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    complex_graph['ligand'].x = atom_feats
    complex_graph['ligand'].pos = lig_coords
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = edge_index
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = edge_attr
    return

def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # else:
    #    AllChem.MMFFOptimizeMolecule(mol_rdkit, confId=0)

def get_lig_graph_with_matching(mol_, complex_graph, popsize, maxiter, matching, keep_original, num_conformers, remove_hs):
    if matching:
        mol_maybe_noh = copy.deepcopy(mol_)
        if remove_hs:
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True)
        if keep_original:
            complex_graph['ligand'].orig_pos = mol_maybe_noh.GetConformer().GetPositions()

        rotable_bonds = get_torsion_angles(mol_maybe_noh)
        if not rotable_bonds: print("no_rotable_bonds but still using it")

        for i in range(num_conformers):
            mol_rdkit = copy.deepcopy(mol_)

            mol_rdkit.RemoveAllConformers()
            mol_rdkit = AllChem.AddHs(mol_rdkit)
            generate_conformer(mol_rdkit)
            if remove_hs:
                mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
            mol = copy.deepcopy(mol_maybe_noh)
            if rotable_bonds:
                optimize_rotatable_bonds(mol_rdkit, mol, rotable_bonds, popsize=popsize, maxiter=maxiter)
            mol.AddConformer(mol_rdkit.GetConformer())
            rms_list = []
            AllChem.AlignMolConformers(mol, RMSlist=rms_list)
            mol_rdkit.RemoveAllConformers()
            mol_rdkit.AddConformer(mol.GetConformers()[1])
            
            if i == 0:
                complex_graph.rmsd_matching = rms_list[0]
                get_lig_graph(mol_rdkit, complex_graph)
            else:
                if torch.is_tensor(complex_graph['ligand'].pos):
                    complex_graph['ligand'].pos = [complex_graph['ligand'].pos]
                complex_graph['ligand'].pos.append(torch.from_numpy(mol_rdkit.GetConformer().GetPositions()).float())

    else:  # no matching
        complex_graph.rmsd_matching = 0
        if remove_hs: mol_ = RemoveHs(mol_)
        get_lig_graph(mol_, complex_graph)

    edge_mask, mask_rotate = get_transformation_mask(complex_graph)
    complex_graph['ligand'].edge_mask = torch.tensor(edge_mask)
    complex_graph['ligand'].mask_rotate = mask_rotate

    return




def get_rec_atom_feat(bio_atom=None, atom_name=None, element=None, get_misc_features=False):
    if get_misc_features:
        return [safe_index(allowable_features['possible_amino_acids'], 'misc'),
                 safe_index(allowable_features['possible_atomic_num_list'], 'misc'),
                 safe_index(allowable_features['possible_atom_type_2'], 'misc'),
                 safe_index(allowable_features['possible_atom_type_3'], 'misc')]
    if atom_name is not None:
        atom_name = atom_name
    else:
        atom_name = bio_atom.name
    if element is not None:
        element = element
    else:
        element = bio_atom.element
    if element == 'CD':
        element = 'C'
    assert not element == ''
    try:
        atomic_num = periodic_table.GetAtomicNumber(element)
    except:
        atomic_num = -1

    atom_feat = [safe_index(allowable_features['possible_amino_acids'], bio_atom.get_parent().get_resname()),
                 safe_index(allowable_features['possible_atomic_num_list'], atomic_num),
                 safe_index(allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                 safe_index(allowable_features['possible_atom_type_3'], atom_name)]
    return atom_feat

def get_rec_misc_atom_feat(bio_atom=None, atom_name=None, element=None, get_misc_features=False):
    if get_misc_features:
        return [safe_index(allowable_features['possible_amino_acids'], 'misc'),
                 safe_index(allowable_features['possible_atomic_num_list'], 'misc'),
                 safe_index(allowable_features['possible_atom_type_2'], 'misc'),
                 safe_index(allowable_features['possible_atom_type_3'], 'misc')]
    if atom_name is not None:
        atom_name = atom_name
    else:
        atom_name = bio_atom.name
    if element is not None:
        element = element
    else:
        element = bio_atom.element
    if element == 'CD':
        element = 'C'
    assert not element == ''
    try:
        atomic_num = periodic_table.GetAtomicNumber(element.lower().capitalize())
    except:
        atomic_num = -1

    atom_feat = [safe_index(allowable_features['possible_amino_acids'], bio_atom.get_parent().get_resname()),
                 safe_index(allowable_features['possible_atomic_num_list'], atomic_num),
                 safe_index(allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                 safe_index(allowable_features['possible_atom_type_3'], atom_name)]
    return atom_feat

def rec_atom_featurizer(rec):
    atom_feats = []
    for i, atom in enumerate(rec.get_atoms()):
        atom_feats.append(get_rec_atom_feat(atom))
    return atom_feats


def get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, misc_coords,misc_features, complex_graph, rec_radius, c_alpha_max_neighbors=None, all_atoms=False, remove_hs=False, lm_embeddings=None):
    if all_atoms:
        return get_fullrec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, misc_coords, misc_features, complex_graph,
                                 c_alpha_cutoff=rec_radius, c_alpha_max_neighbors=c_alpha_max_neighbors,
                                 remove_hs=remove_hs,lm_embeddings=lm_embeddings)
    else:
        return get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, misc_coords,misc_features, complex_graph, rec_radius, c_alpha_max_neighbors, lm_embeddings=lm_embeddings)

def get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, misc_coords,misc_features,complex_graph, cutoff=20, max_neighbor=None, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None):
    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec)
    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))

    complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1) if lm_embeddings is not None else node_feat
    complex_graph['receptor'].pos = torch.from_numpy(c_alpha_coords).float()
    complex_graph['receptor'].mu_r_norm = mu_r_norm
    complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))
    if misc_coords is not None:
        if remove_hs:
            not_hs = (misc_features[:, 1] != 0)
            misc_features = misc_features[not_hs]
            misc_coords = misc_coords[not_hs]
        misc_atoms_edge_index = knn_graph(torch.from_numpy(misc_coords),
                                          k=atom_max_neighbors if atom_max_neighbors else 1000)
        misc_calpha_dist = spatial.distance.cdist(misc_coords, c_alpha_coords)
        misc_atom_res_index = torch.from_numpy(np.stack([np.arange(len(misc_coords)), np.argmin(misc_calpha_dist, axis=1)]))
        complex_graph["misc_atom"].x = torch.from_numpy(misc_features)
        complex_graph["misc_atom"].pos = torch.from_numpy(misc_coords).float()
        complex_graph["misc_atom", 'misc_atom_contact', 'misc_atom'].edge_index = misc_atoms_edge_index
        complex_graph["misc_atom", 'misc_atom_rec_contact', 'receptor'].edge_index = misc_atom_res_index
    return

def get_fullrec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, misc_coords,misc_features, complex_graph, c_alpha_cutoff=20,
                      c_alpha_max_neighbors=None, remove_hs=False, lm_embeddings=None):
    # builds the receptor graph with both residues and atoms

    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph of residues
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < c_alpha_cutoff)[0])
        dst.remove(i)
        if c_alpha_max_neighbors != None and len(dst) > c_alpha_max_neighbors:
            dst = list(np.argsort(distances[i, :]))[1: c_alpha_max_neighbors + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'The c_alpha_cutoff {c_alpha_cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert 1 - 1e-2 < weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec)
    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))

    complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1) if lm_embeddings is not None else node_feat
    complex_graph['receptor'].pos = torch.from_numpy(c_alpha_coords).float()
    complex_graph['receptor'].mu_r_norm = mu_r_norm
    complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    src_c_alpha_idx = np.concatenate([np.asarray([i] * len(l)) for i, l in enumerate(rec_coords)])
    atom_feat = torch.from_numpy(np.asarray(rec_atom_featurizer(rec)))
    atom_coords = torch.from_numpy(np.concatenate(rec_coords, axis=0)).float()
    if misc_coords is not None and len(misc_coords > 0):
        misc_calpha_dist = spatial.distance.cdist(misc_coords, c_alpha_coords)
        closest_c_alpha_idx = np.argmin(misc_calpha_dist, axis=1)
        src_c_alpha_idx = np.concatenate([src_c_alpha_idx, closest_c_alpha_idx])
        atom_coords = torch.cat([atom_coords, torch.from_numpy(misc_coords).float()])
        atom_feat = torch.cat([atom_feat, torch.from_numpy(misc_features)])

    if remove_hs:
        not_hs = (atom_feat[:, 1] != 0)
        atom_feat = atom_feat[not_hs]
        atom_coords = atom_coords[not_hs]
        src_c_alpha_idx = src_c_alpha_idx[not_hs]

    # The knn-graph of atoms will be built in the forward pass
    # atoms_edge_index = knn_graph(atom_coords, k=atom_max_neighbors if atom_max_neighbors else 1000)
    np_array = np.stack([np.arange(len(atom_feat)), src_c_alpha_idx])
    atom_res_edge_index = torch.from_numpy(np_array).long()
    complex_graph['atom'].x = atom_feat
    complex_graph['atom'].pos = atom_coords
    complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index
    return

def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    w.write(mol)
    w.close()

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol


def get_sidechain_rotation_masks(rec, accept_atom_function, remove_hs=False, true_model=None, true_model_matching=None):
    """
    find all sidechains within a specified flexdist of the pocket and compute an ordered rotation mask
    such that the sidechain torsions are applied from the Ca downwards
    """

    flexResIds = {}
    resNames = []  # maps a flexRes idx to the corresponding atom offset of the atom graph, e.g. flexRes 1 -> starts at atom 4719
    resIds = []  # maps each flexible residue to the id from the pdb file

    flexResIdx = 0
    for res in rec.get_residues():
        resname = res.get_resname()
        if resname in {"ALA","GLY","PRO",} or resname not in allowable_features['possible_flexible_sidechains']:  # ignore sidechains without torsional degrees of freedom and pdb special sidechains, that do not correspond to amino acids
            continue 
        
        for atom in res.get_atoms():
            # skip hydrogens and non-sidechain atoms for distance check:
            if atom.element == "H" or atom.name in {"CA", "N", "C", "O", "OXT"}:
                continue
            # compute if sidechain atom is within specified distance (either l2 dist or rectangluar prism akin to gnina)
            if accept_atom_function(atom):
                # hashKey determined by chainId + seqId, as seqIds are not unique
                fullId = res.get_full_id()
                resIds.append((fullId[1], fullId[2], fullId[3][1]))
                resNames.append(fullId[2] + ":" + str(fullId[3][1]))
                hashKey = fullId[2] + str(fullId[3][1])
                if hashKey not in flexResIds:
                    flexResIds[hashKey] = flexResIdx
                    flexResIdx += 1 
                break 
    flexResidues = [{} for _ in range(flexResIdx)]
    flexResIdx = 0

    for i, atom in enumerate(rec.get_atoms()):
        try:
            fullId = atom.get_parent().get_full_id()
            hashKey = fullId[2] + str(fullId[3][1])

            if hashKey in flexResIds:
                true_res = None
                if true_model is not None:
                    cur_residue = atom.get_parent()
                    true_res = true_model_matching[(cur_residue.parent.id, cur_residue.id[1])]

                    assert cur_residue.get_resname() == true_res.get_resname(), "There is a mismatch in the residues"

                # get rotatable bonds and their rotation mask
                flexResidues[flexResIdx] = get_sidechain_rotation_mask(atom.get_parent(), i, true_res)
                flexResIdx += 1
                del flexResIds[hashKey]
        except Exception as e:
            print(f"Skipping residue {atom.get_parent()} because of the error:", e)

    logging.debug(f"Flexible residues: {'-'.join(resNames)}")
    if len(resNames) == 0:
        logging.warning(f"Found {len(resNames)} flexible residues in receptor {rec.get_full_id()}. "
                        f"This is likely to lead to errors downstream. "
                        f"It may be a sign of misaligned coordinates between the receptor and ligand. ")

    # move into tensors for now 
    subcomponentsMerged = []
    edge_idx_merged = []
    subcomponentsMapping = []
    res_n_bonds = []
    
    rows=[]
    cols=[]
    coords=[]

    i = 0
    for res in flexResidues:
        subcomponents,edge_idxs = res["subcomponents"],res["edge_idx"]
        # save how many, i.e, which bonds and edges beling to which residue
        res_n_bonds.append(len(subcomponents))
        for component,edge_idx in zip(subcomponents,edge_idxs):
            subcomponentsMerged += component
            edge_idx_merged.append(edge_idx)
            subcomponentsMapping.append([i,i+len(component)])
            i += len(component)
        if true_model is not None:
            for j, row in enumerate(res["rows"]):
                rows.append([row, row, row])
                cols.append([0, 1, 2])
                coords.append([res["coords"][j][0], res["coords"][j][1], res["coords"][j][2]])

    true_coords = None
    if true_model is not None:
        true_coords = dok_matrix((len(list(rec.get_atoms())), 3), dtype=np.float32)
        for row, col, coord in zip(rows, cols, coords):
            true_coords[row, col] = coord
        true_coords = true_coords.todense() #these should be the correct original coordinates
        assert np.array_equal(np.nonzero(true_coords[:,0])[0],np.sort(np.unique(subcomponentsMerged)))
        true_coords = torch.tensor(true_coords)

    subcomponentsMerged = torch.tensor(subcomponentsMerged)
    subcomponentsMapping = torch.tensor(subcomponentsMapping)
    edge_idx_merged = torch.tensor(edge_idx_merged)
    residueNBondsMapping = torch.tensor(res_n_bonds)

    # if remove hs flag is set_shift indices such that they are correct again
    if remove_hs:
        # recompute atom feats
        atom_feat = torch.from_numpy(np.asarray(rec_atom_featurizer(rec)))
        hs = (atom_feat[:, 1] == 0)
        for i, subcomponentIdx in enumerate(subcomponentsMerged):
            # move the left by number of hs lying to the left
            subcomponentsMerged[i] -= hs[0:subcomponentIdx].sum()
        for i, edge_idx in enumerate(edge_idx_merged):
            # move the left by number of hs lying to the left
            edge_idx_merged[i] -= torch.tensor(
                [hs[0:edge_idx[0]].sum(), hs[0:edge_idx[1]].sum()])

    return subcomponentsMerged, subcomponentsMapping, edge_idx_merged, residueNBondsMapping, resIds, true_coords

def set_sidechain_rotation_masks(complex_graph, rec, accept_atom_function, remove_hs=False, true_model=None, true_model_matching=None):
    """Calls get_sidechain_rotation_masks and sets the corresponding fields in the complex graph"""
    subcomponentsMerged, subcomponentsMapping, edge_idx_merged, residueNBondsMapping, pdbIds, true_coords = \
        get_sidechain_rotation_masks(rec, accept_atom_function, remove_hs, true_model, true_model_matching)

    #                                     subcomp1 , subcomp2
    # flattend list of components, e.g. [[1,2,4]   ,[9,5,13]]   -> [1,2,4,9,5,13]
    complex_graph["flexResidues"].subcomponents = subcomponentsMerged

    # 2d tensor Mapping subcomponents to the corresponding indices in subcomponents
    #        subcomp1  , subcomp2
    # e.g. [[0,2]      ,[2,7],[...]]   , so the atom idxs for subcomponent 1 are subcomponets[0:2], for subcomponent2 they are subcomponents[2:7] etc.
    complex_graph["flexResidues"].subcomponentsMapping = subcomponentsMapping

    # edge idxs of the subcomponents, defining the rotatable bond which belongs to a subcomponent
    complex_graph["flexResidues"].edge_idx = edge_idx_merged

    # list of number of bonds for each flexRes : [nBonds_flexRes1, nBonds_flexRes2,....]
    complex_graph["flexResidues"].residueNBondsMapping = residueNBondsMapping
    
    complex_graph["flexResidues"].pdbIds = pdbIds

    complex_graph["flexResidues"].true_sc_pos = true_coords

    complex_graph["flexResidues"].num_nodes = complex_graph["flexResidues"].edge_idx.shape[0]

    return complex_graph


def count_pdb_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning to be caught
            warnings.simplefilter('always', PDBConstructionWarning)

            # Call the function
            result = func(*args, **kwargs)

            # Count specific warnings
            num_warnings = len([warn for warn in w if warn.category == PDBConstructionWarning])
            if num_warnings > 0:
                logging.warning(f"Number of PDBConstructionWarnings in {func.__name__}: {num_warnings}")

            return result

    return wrapper
