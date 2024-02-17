import os

import plotly.express as px
import time
from argparse import FileType, ArgumentParser

import tempfile
import numpy as np
import pandas as pd
import wandb
from Bio.PDB import PDBParser
from biopandas.pdb import PandasPdb
from rdkit import Chem

from tqdm import tqdm

from datasets.pdbbind import read_mol
from datasets.process_mols import read_molecule
from utils.utils import read_strings_from_txt, get_symmetry_rmsd
from datasets.steric_clash import get_steric_clash_atom_pairs, get_rec_elements, get_ligand_elements, \
    get_steric_clash_per_flexble_sidechain_atom

parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--run_name', type=str, default='gnina_results', help='')
parser.add_argument('--complex_names_path', type=str, default='data/splits/timesplit_test', help='')
parser.add_argument('--data_dir', type=str, default='data/PDBBIND_atomCorrected', help='')
parser.add_argument('--results_path', type=str, help='The output directory of gnina')
parser.add_argument('--file_suffix', type=str, default='_baseline_ligand.pdb', help='')
parser.add_argument('--project', type=str, default='ligbind_inf', help='')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--file_to_exclude', type=str, default='rank1.sdf', help='')
parser.add_argument('--all_dirs_in_results', action='store_true', default=False,
                    help='Evaluate all directories in the results path instead of using directly looking for the names')
parser.add_argument('--num_predictions', type=int, default=10, help='')
parser.add_argument('--no_id_in_filename', action='store_true', default=False, help='')
parser.add_argument('--skip_complexes_path', type=str, default=None, help='')

parser.add_argument('--results_path_flex', type=str,
                    help='The output directory of get_orig_flex.py (i.e., processed gnina results). Will also be used to store the results.')

# Parameters for rigid
parser.add_argument('--protein_file', type=str, default='', help='Protein file suffix')

# Parameters for flex
parser.add_argument('--full_pdb_suffix', type=str, default='_full', help='The suffix of the full pdb file to insert the flex predictions into, so that we can compute the steric clashes.')
parser.add_argument('--orig_rec_suffix', type=str, default='_orig', help='')
parser.add_argument('--flex', action='store_true', default=False, help='If flexible residues have been produced')
parser.add_argument('--orig_rec_suffix_before', type=str, default=None, help='The before suffix to compare the orig_rec_suffix to. For example, to compare apo with holo use _gnina')
parser.add_argument('--flex_rec_suffix', default='', type=str, help='The suffix of the flexible resiudes')
args = parser.parse_args()

N = args.num_predictions


def load_pdb_with_models(path):
    with open(path, 'r') as file:
        content = file.read()
        model_text = ['MODEL' + model for model in content.split('MODEL') if len(model) > 0]

    models = []
    parser = PDBParser()

    for model in model_text:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(model.encode())
            tmp.flush()
            models.append(parser.get_structure('random_id', tmp.name)[0])

    return models


print('Reading paths and names.')
names = read_strings_from_txt(args.complex_names_path)
names_no_rec_overlap = read_strings_from_txt(f'data/splits/timesplit_test_no_rec_overlap')
skips = [] if args.skip_complexes_path is None else read_strings_from_txt(args.skip_complexes_path)
if len(skips) > 0:
    print('Skipping', len(skips), 'complexes.')

results_path_containments = os.listdir(args.results_path)

if args.wandb:
    wandb.init(
        entity='coarse-graining-mit',
        settings=wandb.Settings(start_method="fork"),
        project=args.project,
        name=args.run_name,
        config=args
    )

all_times = []
successful_names_list = []
rmsds_list = []
holo_apo_sc_rmsds_before_list = []
sc_rmsds_list = []
rec_lig_steric_clashes = []
centroid_distances_list = []
min_cross_distances_list = []
min_self_distances_list = []
without_rec_overlap_list = []
start_time = time.time()
errors = 0
for i, name in enumerate(tqdm(names)):
    if name in skips:
        print('Skipping', name, 'because it is in the skip list.')
        continue

    try:
        try:
            mol = read_mol(args.data_dir, name, remove_hs=True)
            mol = Chem.RemoveAllHs(mol)
            orig_ligand_pos = np.array(mol.GetConformer().GetPositions())
            orig_ligand_elements = [a.GetSymbol() for a in mol.GetAtoms()]
        except Exception as e:
            print('Could not read', name, 'because of', e)
            errors += 1
            continue


        if args.all_dirs_in_results:
            directory_with_name = [directory for directory in results_path_containments if name in directory][0]
            ligand_pos = []
            debug_paths = []
            for i in range(args.num_predictions):
                file_paths = sorted(os.listdir(os.path.join(args.results_path, directory_with_name)))
                if args.file_to_exclude is not None:
                    file_paths = [path for path in file_paths if not args.file_to_exclude in path]
                file_path = [path for path in file_paths if f'rank{i + 1}_' in path][0]
                mol_pred = read_molecule(os.path.join(args.results_path, directory_with_name, file_path), remove_hs=True,
                                         sanitize=True)
                mol_pred = Chem.RemoveAllHs(mol_pred)
                ligand_pos.append(mol_pred.GetConformer().GetPositions())
                debug_paths.append(file_path)
            ligand_pos = np.asarray(ligand_pos)
        else:
            if not os.path.exists(
                    os.path.join(args.results_path, name, f'{"" if args.no_id_in_filename else name}{args.file_suffix}')):
                print('skipping because path did not exists:',
                      os.path.join(args.results_path, name, f'{"" if args.no_id_in_filename else name}{args.file_suffix}'))
                continue
            mol_pred = read_molecule(
                os.path.join(args.results_path, name, f'{"" if args.no_id_in_filename else name}{args.file_suffix}'),
                remove_hs=True, sanitize=True)
            if mol_pred == None:
                print("Skipping ", name, ' because RDKIT could not read it.')
                continue
            mol_pred = Chem.RemoveAllHs(mol_pred)
            ligand_pos = np.asarray(
                [np.array(mol_pred.GetConformer(i).GetPositions()) for i in range(args.num_predictions)])
        try:
            rmsd = get_symmetry_rmsd(mol, orig_ligand_pos, [l for l in ligand_pos], mol_pred)
        except Exception as e:
            print("Using non corrected RMSD because of the error:", e)
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

        if args.flex:
            rec_path = os.path.join(args.results_path_flex, name, f'{name}{args.orig_rec_suffix}.pdb')
            if not os.path.exists(rec_path):
                raise FileNotFoundError(rec_path)

            rec = load_pdb_with_models(rec_path)
            rec_atoms = [list(model.get_atoms()) for model in rec]
            rec_atoms_names = [[atom.get_name() for atom in model.get_atoms()] for model in rec]
            rec_atoms_coords = np.array([[atom.get_coord() for atom in model.get_atoms()] for model in rec])

            rec_path_flex = os.path.join(args.results_path_flex, name, f'{name}{args.flex_rec_suffix}.pdb')
            if not os.path.exists(rec_path_flex):
                raise FileNotFoundError(rec_path_flex)

            rec_flex = load_pdb_with_models(rec_path_flex)
            rec_flex_atoms = [list(model.get_atoms()) for model in rec_flex]
            rec_flex_atoms_names = [[atom.get_name() for atom in model.get_atoms()] for model in rec_flex]
            rec_flex_atoms_coords = np.array([[atom.get_coord() for atom in model.get_atoms()] for model in rec_flex])

            assert rec_atoms_coords.shape == rec_flex_atoms_coords.shape
            assert rec_atoms_names == rec_flex_atoms_names

            filterSCHs = np.array(
                [[atom.element != 'H' and atom.name not in {"CA", "N", "C", "O", "OXT"} for atom in model.get_atoms()] for model
                 in rec_flex])

            filtered_rec_atom_coords = rec_atoms_coords[filterSCHs].reshape(filterSCHs.shape[0], -1, 3)
            filtered_rec_flex_atom_coords = rec_flex_atoms_coords[filterSCHs].reshape(filterSCHs.shape[0], -1, 3)

            if args.orig_rec_suffix_before is not None:
                rec_path_before = os.path.join(args.results_path_flex, name, f'{name}{args.orig_rec_suffix_before}.pdb')
                if not os.path.exists(rec_path_before):
                    raise FileNotFoundError(rec_path_before)

                rec_before = load_pdb_with_models(rec_path_before)
                rec_before_atoms = [list(model.get_atoms()) for model in rec_before]
                rec_before_atoms_names = [[atom.get_name() for atom in model.get_atoms()] for model in rec_before]
                rec_before_atoms_coords = np.array([[atom.get_coord() for atom in model.get_atoms()] for model in rec_before])

                assert rec_atoms_coords.shape == rec_before_atoms_coords.shape
                assert rec_atoms_names == rec_before_atoms_names

                holo_apo_sc_rmsds_before = np.sqrt(((rec_atoms_coords - rec_before_atoms_coords) ** 2).sum(axis=2).mean(axis=1))
                assert all(holo_apo_sc_rmsds_before == holo_apo_sc_rmsds_before[0])
                holo_apo_sc_rmsds_before = holo_apo_sc_rmsds_before[0]

                holo_apo_sc_rmsds_before_list.append(holo_apo_sc_rmsds_before)

            full_rec_path = os.path.join(args.results_path_flex, name, f'{name}{args.full_pdb_suffix}.pdb')
            if not os.path.exists(full_rec_path):
                raise FileNotFoundError(full_rec_path)

            full_rec_flex = load_pdb_with_models(full_rec_path)
            full_rec_flex_atoms = [list(model.get_atoms()) for model in full_rec_flex]
            full_rec_flex_atoms_elements = np.array([[atom.element for atom in model.get_atoms()] for model in full_rec_flex])
            full_rec_flex_atoms_coords = np.array([[atom.get_coord() for atom in model.get_atoms()] for model in full_rec_flex])

            rec_lig_steric_clash = np.sum(
                get_steric_clash_atom_pairs(full_rec_flex_atoms_coords, ligand_pos,
                                            full_rec_flex_atoms_elements[0], orig_ligand_elements,
                                            filter1=full_rec_flex_atoms_elements[0] != 'H'), axis=(1, 2))
            rec_lig_steric_clashes.append(rec_lig_steric_clash)

            sc_rmsds = np.sqrt(((filtered_rec_atom_coords - filtered_rec_flex_atom_coords) ** 2).sum(axis=2).mean(axis=1))
            sc_rmsds_list.append(sc_rmsds)
        else:
            rec_path = os.path.join(args.data_dir, name, f'{name}_{args.protein_file}.pdb')
            if not os.path.exists(rec_path):
                raise FileNotFoundError(rec_path)

            parser = PDBParser(QUIET=True)
            rec = parser.get_structure('random_id', rec_path)[0]
            rec_atoms_elements = np.array([atom.element for atom in rec.get_atoms()])
            rec_atoms_coords = np.array([atom.get_coord() for atom in rec.get_atoms()])
            rec_atoms_coords = rec_atoms_coords[None, :]

            rec_lig_steric_clash = np.sum(
                get_steric_clash_atom_pairs(np.broadcast_to(rec_atoms_coords, (
                    args.num_predictions, rec_atoms_coords.shape[1], rec_atoms_coords.shape[2])), ligand_pos,
                                            rec_atoms_elements, orig_ligand_elements,
                                            filter1=rec_atoms_elements != 'H'), axis=(1, 2))
            rec_lig_steric_clashes.append(rec_lig_steric_clash)


        cross_distances = np.linalg.norm(rec_atoms_coords[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
        self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
        self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
        min_cross_distances_list.append(np.min(cross_distances, axis=(1, 2)))
        min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))
        rmsds_list.append(rmsd)
        centroid_distances_list.append(
            np.linalg.norm(ligand_pos.mean(axis=1) - orig_ligand_pos[None, :].mean(axis=1), axis=1))
        successful_names_list.append(name)
        without_rec_overlap_list.append(1 if name in names_no_rec_overlap else 0)
    except Exception as e:
        print('Error while processing', name)
        raise e

print(errors, "of the complexes failed")

performance_metrics = {}
for overlap in ['', 'no_overlap_']:
    if 'no_overlap_' == overlap:
        without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
        rmsds = np.array(rmsds_list)[without_rec_overlap]
        if args.flex:
            sc_rmsds = np.array(sc_rmsds_list)[without_rec_overlap]
        if len(holo_apo_sc_rmsds_before_list) > 0:
            holo_apo_sc_rmsds_before = np.array(holo_apo_sc_rmsds_before_list)[without_rec_overlap]

        centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
        min_cross_distances = np.array(min_cross_distances_list)[without_rec_overlap]
        min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
        successful_names = np.array(successful_names_list)[without_rec_overlap]
        rec_lig_steric_clashes = np.array(rec_lig_steric_clashes)[without_rec_overlap]
    else:
        rmsds = np.array(rmsds_list)
        sc_rmsds = np.array(sc_rmsds_list)
        holo_apo_sc_rmsds_before = np.array(holo_apo_sc_rmsds_before_list)
        centroid_distances = np.array(centroid_distances_list)
        min_cross_distances = np.array(min_cross_distances_list)
        min_self_distances = np.array(min_self_distances_list)
        successful_names = np.array(successful_names_list)
        rec_lig_steric_clashes = np.array(rec_lig_steric_clashes)

    np.save(os.path.join(args.results_path_flex, f'{overlap}rmsds.npy'), rmsds)
    np.save(os.path.join(args.results_path_flex, f'{overlap}sc_rmsds.npy'), sc_rmsds)
    if len(holo_apo_sc_rmsds_before) > 0:
        np.save(os.path.join(args.results_path_flex, f'{overlap}holo_apo_sc_rmsds_before.npy'), holo_apo_sc_rmsds_before)
    np.save(os.path.join(args.results_path_flex, f'{overlap}names.npy'), successful_names)
    np.save(os.path.join(args.results_path_flex, f'{overlap}min_cross_distances.npy'), np.array(min_cross_distances))
    np.save(os.path.join(args.results_path_flex, f'{overlap}min_self_distances.npy'), np.array(min_self_distances))
    np.save(os.path.join(args.results_path_flex, f'{overlap}complex_names.npy'), np.array(successful_names))
    np.save(os.path.join(args.results_path_flex, f'{overlap}rec_lig_steric_clashes.npy'), np.array(successful_names))

    performance_metrics.update({
        f'{overlap}steric_clash_fraction': (100 * (min_cross_distances < 0.4).sum() / len(
            min_cross_distances) / args.num_predictions).__round__(2),
        f'{overlap}self_intersect_fraction': (
                    100 * (min_self_distances < 0.4).sum() / len(min_self_distances) / args.num_predictions).__round__(
            2),
        f'{overlap}top1_mean_rmsd': rmsds[:, 0].mean().round(2),
        f'{overlap}top1_rmsds_below_2': (100 * (rmsds[:, 0] < 2).sum() / len(rmsds[:, 0])).round(2),
        f'{overlap}top1_rmsds_below_5': (100 * (rmsds[:, 0] < 5).sum() / len(rmsds[:, 0])).round(2),
        f'{overlap}top1_rmsds_percentile_25': np.percentile(rmsds[:, 0], 25).round(2),
        f'{overlap}top1_rmsds_percentile_50': np.percentile(rmsds[:, 0], 50).round(2),
        f'{overlap}top1_rmsds_percentile_75': np.percentile(rmsds[:, 0], 75).round(2),

        f'{overlap}mean_rmsd': rmsds.mean().round(2),
        f'{overlap}rmsds_below_2': (100 * (rmsds < 2).sum() / len(rmsds) / N).round(2),
        f'{overlap}rmsds_below_5': (100 * (rmsds < 5).sum() / len(rmsds) / N).round(2),
        f'{overlap}rmsds_percentile_25': np.percentile(rmsds, 25).round(2),
        f'{overlap}rmsds_percentile_50': np.percentile(rmsds, 50).round(2),
        f'{overlap}rmsds_percentile_75': np.percentile(rmsds, 75).round(2),

        f'{overlap}mean_centroid': centroid_distances[:, 0].mean().__round__(2),
        f'{overlap}centroid_below_2': (
                    100 * (centroid_distances[:, 0] < 2).sum() / len(centroid_distances[:, 0])).__round__(2),
        f'{overlap}centroid_below_5': (
                    100 * (centroid_distances[:, 0] < 5).sum() / len(centroid_distances[:, 0])).__round__(2),
        f'{overlap}centroid_percentile_25': np.percentile(centroid_distances[:, 0], 25).round(2),
        f'{overlap}centroid_percentile_50': np.percentile(centroid_distances[:, 0], 50).round(2),
        f'{overlap}centroid_percentile_75': np.percentile(centroid_distances[:, 0], 75).round(2),
    })

    if args.flex:
        performance_metrics.update({
            f'{overlap}mean_sidechain_rmsd': sc_rmsds.mean(),
            f'{overlap}sidechain_rmsds_below_0.25': (100 * (sc_rmsds < 0.25).sum() / len(sc_rmsds) / N).round(2),
            f'{overlap}sidechain_rmsds_below_0.5': (100 * (sc_rmsds < 0.5).sum() / len(sc_rmsds) / N).round(2),
            f'{overlap}sidechain_rmsds_below_1': (100 * (sc_rmsds < 1).sum() / len(sc_rmsds) / N).round(2),
            f'{overlap}sidechain_rmsds_below_2': (100 * (sc_rmsds < 2).sum() / len(sc_rmsds) / N).round(2),
            f'{overlap}sidechain_rmsds_percentile_25': np.percentile(sc_rmsds, 25).round(2),
            f'{overlap}sidechain_rmsds_percentile_50': np.percentile(sc_rmsds, 50).round(2),
            f'{overlap}sidechain_rmsds_percentile_75': np.percentile(sc_rmsds, 75).round(2),

            f'{overlap}top1_mean_sidechain_rmsd': sc_rmsds[:, 0].mean(),
            f'{overlap}top1_sidechain_rmsds_below_0.25': (100 * (sc_rmsds[:, 0] < 0.25).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top1_sidechain_rmsds_below_0.5': (100 * (sc_rmsds[:, 0] < 0.5).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top1_sidechain_rmsds_below_1': (100 * (sc_rmsds[:, 0] < 1).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top1_sidechain_rmsds_below_2': (100 * (sc_rmsds[:, 0] < 2).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top1_sidechain_rmsds_percentile_25': np.percentile(sc_rmsds[:, 0], 25).round(2),
            f'{overlap}top1_sidechain_rmsds_percentile_50': np.percentile(sc_rmsds[:, 0], 50).round(2),
            f'{overlap}top1_sidechain_rmsds_percentile_75': np.percentile(sc_rmsds[:, 0], 75).round(2),
        })

    performance_metrics.update({
        f'{overlap}top1_rec_lig_steric_clashes_fraction': (
                    100 * (rec_lig_steric_clashes[:, 0] > 0).sum() / len(rec_lig_steric_clashes)).round(2),
        f'{overlap}top1_rec_lig_steric_clashes_mean': (rec_lig_steric_clashes[:, 0]).mean().round(2),
        f'{overlap}top1_rec_lig_steric_clashes_mean_if_clash': rec_lig_steric_clashes[:, 0][
            rec_lig_steric_clashes[:, 0] > 0].mean().round(2),
    })

    top5_rmsds = np.min(rmsds[:, :5], axis=1)
    top5_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][
                              :, 0]
    top5_min_cross_distances = min_cross_distances[
                                   np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]
    top5_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][
                              :, 0]
    performance_metrics.update({
        f'{overlap}top5_steric_clash_fraction': (
                    100 * (top5_min_cross_distances < 0.4).sum() / len(top5_min_cross_distances)).__round__(2),
        f'{overlap}top5_self_intersect_fraction': (
                    100 * (top5_min_self_distances < 0.4).sum() / len(top5_min_self_distances)).__round__(2),
        f'{overlap}top5_rmsds_below_2': (100 * (top5_rmsds < 2).sum() / len(top5_rmsds)).__round__(2),
        f'{overlap}top5_rmsds_below_5': (100 * (top5_rmsds < 5).sum() / len(top5_rmsds)).__round__(2),
        f'{overlap}top5_rmsds_percentile_25': np.percentile(top5_rmsds, 25).round(2),
        f'{overlap}top5_rmsds_percentile_50': np.percentile(top5_rmsds, 50).round(2),
        f'{overlap}top5_rmsds_percentile_75': np.percentile(top5_rmsds, 75).round(2),

        f'{overlap}top5_centroid_below_2': (
                    100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)).__round__(2),
        f'{overlap}top5_centroid_below_5': (
                    100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)).__round__(2),
        f'{overlap}top5_centroid_percentile_25': np.percentile(top5_centroid_distances, 25).round(2),
        f'{overlap}top5_centroid_percentile_50': np.percentile(top5_centroid_distances, 50).round(2),
        f'{overlap}top5_centroid_percentile_75': np.percentile(top5_centroid_distances, 75).round(2),
    })

    if args.flex:
        top5_sc_rmsds = np.min(sc_rmsds[:, :5], axis=1)

        performance_metrics.update({
            f'{overlap}top5_mean_sidechain_rmsd': top5_sc_rmsds.mean().round(2),
            f'{overlap}top5_sidechain_rmsds_below_0.25': (100 * (top5_sc_rmsds < 0.25).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top5_sidechain_rmsds_below_0.5': (100 * (top5_sc_rmsds < 0.5).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top5_sidechain_rmsds_below_1': (100 * (top5_sc_rmsds < 1).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top5_sidechain_rmsds_below_2': (100 * (top5_sc_rmsds < 2).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top5_sidechain_rmsds_percentile_25': np.percentile(top5_sc_rmsds, 25).round(2),
            f'{overlap}top5_sidechain_rmsds_percentile_50': np.percentile(top5_sc_rmsds, 50).round(2),
            f'{overlap}top5_sidechain_rmsds_percentile_75': np.percentile(top5_sc_rmsds, 75).round(2),
        })

    top10_rmsds = np.min(rmsds[:, :10], axis=1)

    top10_centroid_distances = centroid_distances[
                                   np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
    top10_min_cross_distances = min_cross_distances[
                                    np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
    top10_min_self_distances = min_self_distances[
                                   np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
    performance_metrics.update({
        f'{overlap}top10_self_intersect_fraction': (
                    100 * (top10_min_self_distances < 0.4).sum() / len(top10_min_self_distances)).__round__(2),
        f'{overlap}top10_steric_clash_fraction': (
                    100 * (top10_min_cross_distances < 0.4).sum() / len(top10_min_cross_distances)).__round__(2),
        f'{overlap}top10_rmsds_below_2': (100 * (top10_rmsds < 2).sum() / len(top10_rmsds)).__round__(2),
        f'{overlap}top10_rmsds_below_5': (100 * (top10_rmsds < 5).sum() / len(top10_rmsds)).__round__(2),
        f'{overlap}top10_rmsds_percentile_25': np.percentile(top10_rmsds, 25).round(2),
        f'{overlap}top10_rmsds_percentile_50': np.percentile(top10_rmsds, 50).round(2),
        f'{overlap}top10_rmsds_percentile_75': np.percentile(top10_rmsds, 75).round(2),

        f'{overlap}top10_centroid_below_2': (
                    100 * (top10_centroid_distances < 2).sum() / len(top10_centroid_distances)).__round__(2),
        f'{overlap}top10_centroid_below_5': (
                    100 * (top10_centroid_distances < 5).sum() / len(top10_centroid_distances)).__round__(2),
        f'{overlap}top10_centroid_percentile_25': np.percentile(top10_centroid_distances, 25).round(2),
        f'{overlap}top10_centroid_percentile_50': np.percentile(top10_centroid_distances, 50).round(2),
        f'{overlap}top10_centroid_percentile_75': np.percentile(top10_centroid_distances, 75).round(2),
    })

    if args.flex:
        top10_sc_rmsds = np.min(sc_rmsds[:, :10], axis=1)

        performance_metrics.update({
            f'{overlap}top10_mean_sidechain_rmsd': top10_sc_rmsds.mean(),
            f'{overlap}top10_sidechain_rmsds_below_0.25': (100 * (top10_sc_rmsds < 0.25).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top10_sidechain_rmsds_below_0.5': (100 * (top10_sc_rmsds < 0.5).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top10_sidechain_rmsds_below_1': (100 * (top10_sc_rmsds < 1).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top10_sidechain_rmsds_below_2': (100 * (top10_sc_rmsds < 2).sum() / len(sc_rmsds)).round(2),
            f'{overlap}top10_sidechain_rmsds_percentile_25': np.percentile(top10_sc_rmsds, 25).round(2),
            f'{overlap}top10_sidechain_rmsds_percentile_50': np.percentile(top10_sc_rmsds, 50).round(2),
            f'{overlap}top10_sidechain_rmsds_percentile_75': np.percentile(top10_sc_rmsds, 75).round(2),
        })
for k in performance_metrics:
    print(k, performance_metrics[k])

if args.wandb:
    wandb.log(performance_metrics)
    histogram_metrics_list = [('rmsd', rmsds[:, 0]),
                              ('sc_rmsd', sc_rmsds[:]),
                              ('centroid_distance', centroid_distances[:, 0]),
                              ('mean_rmsd', rmsds[:, 0]),
                              ('mean_sc_rmsd', sc_rmsds[:]),
                              ('mean_centroid_distance', centroid_distances[:, 0])]
    histogram_metrics_list.append(('top5_rmsds', top5_rmsds))
    histogram_metrics_list.append(('top5_sc_rmsds', top5_sc_rmsds))
    histogram_metrics_list.append(('top5_centroid_distances', top5_centroid_distances))
    histogram_metrics_list.append(('top10_rmsds', top10_rmsds))
    histogram_metrics_list.append(('top10_sc_rmsds', top10_sc_rmsds))
    histogram_metrics_list.append(('top10_centroid_distances', top10_centroid_distances))

    os.makedirs(f'.plotly_cache/baseline_cache', exist_ok=True)
    images = []
    for metric_name, metric in histogram_metrics_list:
        d = {args.results_path: metric}
        df = pd.DataFrame(data=d)
        fig = px.ecdf(df, width=900, height=600, range_x=[0, 40])
        fig.add_vline(x=2, annotation_text='2 A;', annotation_font_size=20, annotation_position="top right",
                      line_dash='dash', line_color='firebrick', annotation_font_color='firebrick')
        fig.add_vline(x=5, annotation_text='5 A;', annotation_font_size=20, annotation_position="top right",
                      line_dash='dash', line_color='green', annotation_font_color='green')
        fig.update_xaxes(title=f'{metric_name} in Angstrom', title_font={"size": 20}, tickfont={"size": 20})
        fig.update_yaxes(title=f'Fraction of predictions with lower error', title_font={"size": 20},
                         tickfont={"size": 20})
        fig.update_layout(autosize=False, margin={'l': 0, 'r': 0, 't': 0, 'b': 0}, plot_bgcolor='white',
                          paper_bgcolor='white', legend_title_text='Method', legend_title_font_size=17,
                          legend=dict(yanchor="bottom", y=0.1, xanchor="right", x=0.99, font=dict(size=17), ), )
        fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridcolor='lightgrey')

        fig.write_image(os.path.join(f'.plotly_cache/baseline_cache', f'{metric_name}.png'))
        wandb.log({metric_name: wandb.Image(os.path.join(f'.plotly_cache/baseline_cache', f'{metric_name}.png'),
                                            caption=f"{metric_name}")})
        images.append(
            wandb.Image(os.path.join(f'.plotly_cache/baseline_cache', f'{metric_name}.png'), caption=f"{metric_name}"))
    wandb.log({'images': images})
