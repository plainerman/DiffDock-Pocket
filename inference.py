import copy
import functools
import logging
import os
import time
from typing import Mapping, Optional

import torch

import yaml
import math
import multiprocessing
import traceback

import tempfile
from argparse import ArgumentParser, Namespace, FileType

from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

from datasets.process_mols import write_mol_with_coords, parse_pdb_from_path
from datasets.pdbbind import PDBBind, load_protein_ligand_df
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, get_available_devices, get_default_device, ensure_device
from utils.visualise import PDBFile, SidechainPDBFile
from tqdm import tqdm
from utils import esm as esm_utils
from utils.download import download_and_extract
from utils.posebusters_em import optimize_ligand_in_pocket
from pathlib import Path
from openmm.unit import megajoule, mole


if os.name != 'nt':  # The line does not work on Windows
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

RDLogger.DisableLog('rdApp.*')

REPOSITORY_URL = 'https://github.com/plainerman/DiffDock-Pocket'


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--complex_name', type=str, default='unnamed_complex', help='Name that the complex will be saved with')
    parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path and --ligand parameters')
    parser.add_argument('--protein_path', '--experimental_protein', type=str, default=None, help='Path to the protein .pdb file')

    parser.add_argument('--ligand', type=str, default='COc(cc1)ccc1C#N', help='Either a SMILES string or the path to a molecule file that rdkit can read')
    parser.add_argument('--flexible_sidechains', type=str, default=None, help='Specify which amino acids will be flexible. E.g., A:130-B:140 will make amino acid with id 130 in chain A, and id 140 in chain B flexible.')
    parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
    parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
    parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--rigid', action='store_true', default=False, help='Override the arguments of the model and use a rigid model. Caution: In our tests this resulted in worse performance.')
    parser.add_argument('--relax', action='store_true', default=False, help='Perform energy minimization on the top-1 ligand pose. See https://github.com/maabuu/posebusters_em for more information.')

    parser.add_argument('--pocket_center_x', type=float, default=None, help='The x coordinate for the pocket center')
    parser.add_argument('--pocket_center_y', type=float, default=None, help='The x coordinate for the pocket center')
    parser.add_argument('--pocket_center_z', type=float, default=None, help='The x coordinate for the pocket center')

    parser.add_argument('--tag', type=str, default='v1.0.0', help='GitHub release tag that will be used to download a model if none is specified.')
    parser.add_argument('--model_cache_dir', type=str, default='.cache/model', help='Folder from where to load/restore the trained model')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
    parser.add_argument('--filtering_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--filtering_ckpt', type=str, default='best_model.pt', help='Checkpoint to use for the confidence model')

    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--cache_path', type=str, default='.cache/data', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
    parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--ode', action='store_true', default=False, help='Use ODE formulation for inference')
    parser.add_argument('--inference_steps', type=int, default=30, help='Number of denoising steps')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
    parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='')
    parser.add_argument('--inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
    parser.add_argument('--inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
    parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')
    parser.add_argument('--keep_local_structures', action='store_true', default=False, help='Keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit')
    parser.add_argument('--skip_existing', action='store_true', default=False, help='If the output directory already exists, skip the inference')

    # This is for low temperature sampling for each individual parameter
    # see Illuminating protein space with a programmable generative model, Appendix B
    # The default values will probably only work nicely for the model trained presented in the paper
    # If you train your own model, you have to fine-tune these parameters on the validation set and pick the best ones
    parser.add_argument('--temp_sampling_tr', type=float,       default=0.9766350103728372)
    parser.add_argument('--temp_psi_tr', type=float,            default=1.5102572175711826)
    parser.add_argument('--temp_sampling_rot', type=float,      default=6.077432837220868)
    parser.add_argument('--temp_psi_rot', type=float,           default=0.8141168207563049)
    parser.add_argument('--temp_sampling_tor', type=float,      default=6.761568162335063)
    parser.add_argument('--temp_psi_tor', type=float,           default=0.7661845361370018)
    parser.add_argument('--temp_sampling_sc_tor', type=float,   default=1.4487910576602347)
    parser.add_argument('--temp_psi_sc_tor', type=float,        default=1.339614553802453)
    parser.add_argument('--temp_sigma_data', type=float,        default=0.48884149503636976)

    return parser


@ensure_device
def infer_single_complex(idx: int, protein_ligand_info_row: Mapping, model: torch.nn.Module, args, score_model_args,
                         filtering_args=None, filtering_model=None, filtering_model_args=None,
                         filtering_complex_dict=None,
                         t_schedule=None, tr_schedule=None,
                         device=None):

    orig_complex_graph = protein_ligand_info_row["complex_graph"].to(device)

    complex_name = orig_complex_graph.name
    spc = args.samples_per_complex

    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    sidechain_tor_schedule = tr_schedule

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    for m in (model, filtering_model):
        if m is not None:
            m = m.to(device)
            m.eval()

    if (filtering_model is not None and not (
            filtering_args.use_original_model_cache or filtering_args.transfer_weights) and complex_name
            not in filtering_complex_dict.keys()):
        print(f"HAPPENING | The filtering dataset did not contain {complex_name}. We are skipping this complex.")

    data_list = []
    try:
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(spc)]
        write_dir = f'{args.out_dir}/index{idx}___{complex_name.replace("/", "-")}'
        if os.path.exists(write_dir) and args.skip_existing:
            return 0

        randomize_position(data_list, score_model_args.no_torsion, args.no_random, score_model_args.tr_sigma_max,
                           flexible_sidechains=False if args.rigid else score_model_args.flexible_sidechains)

        pdb = None
        lig = orig_complex_graph.mol
        if args.save_visualisation:
            visualization_list = []
            sidechain_visualization_list = []

            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs:
                mol_pred = RemoveHs(mol_pred)

            for graph in data_list:
                pdb = PDBFile(mol_pred)
                pdb.add(mol_pred, 0, 0)
                pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                # Ligand with first noise applied
                pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                visualization_list.append(pdb)

                if not args.rigid and score_model_args.flexible_sidechains:
                    animation = [orig_complex_graph["atom"].pos + orig_complex_graph.original_center,
                                 orig_complex_graph["atom"].pos + orig_complex_graph.original_center,
                                 graph["atom"].pos + graph.original_center]

                    sidechain_visualization_list.append(animation)
        else:
            visualization_list = None
            sidechain_visualization_list = None

        if filtering_model is not None and not (
                filtering_args.use_original_model_cache or filtering_args.transfer_weights):
            filtering_data_list = [copy.deepcopy(filtering_complex_dict[complex_name]) for _ in range(spc)]
        else:
            filtering_data_list = None

        data_list, confidence = sampling(data_list=data_list, model=model,
                                         inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                         tor_schedule=tor_schedule, sidechain_tor_schedule=sidechain_tor_schedule,
                                         t_schedule=t_schedule,
                                         t_to_sigma=t_to_sigma, model_args=score_model_args,
                                         confidence_model=filtering_model,
                                         device=device,
                                         visualization_list=visualization_list,
                                         sidechain_visualization_list=sidechain_visualization_list,
                                         no_random=args.no_random,
                                         ode=args.ode, filtering_data_list=filtering_data_list,
                                         filtering_model_args=filtering_model_args,
                                         asyncronous_noise_schedule=score_model_args.asyncronous_noise_schedule,
                                         batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise,
                                         temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot,
                                                        args.temp_sampling_tor, args.temp_sampling_sc_tor],
                                         temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor,
                                                   args.temp_psi_sc_tor],
                                         flexible_sidechains=False if args.rigid else score_model_args.flexible_sidechains)

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for
             complex_graph in data_list])
        atom_pos = np.asarray(
            [complex_graph['atom'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for
             complex_graph in data_list])

        rec_struc = parse_pdb_from_path(protein_ligand_info_row["experimental_protein"])
        # Similarly as in pdb preprocess, we sort the atoms by the name and put hydrogens at the end
        for res in rec_struc.get_residues():
            res.child_list.sort(key=lambda atom: PDBBind.order_atoms_in_residue(res, atom))
            res.child_list = [atom for atom in res.child_list if
                              not score_model_args.remove_hs or atom.element != 'H']

        if confidence is not None and isinstance(filtering_args.rmsd_classification_cutoff, list):
            confidence = confidence[:, 0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            ligand_pos = ligand_pos[re_order]
            atom_pos = atom_pos[re_order]

        os.makedirs(write_dir, exist_ok=True)
        ligand_path = None
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)
            if rank == 0:
                ligand_path = os.path.join(write_dir, f'rank{rank + 1}.sdf')
                write_mol_with_coords(mol_pred, pos, ligand_path)
            write_mol_with_coords(mol_pred, pos,
                                  os.path.join(write_dir, f'rank{rank + 1}_confidence{confidence[rank]:.2f}.sdf'))

        # if flexibility is enabled, this will be changed to the predicted flexible protein
        protein_path = protein_ligand_info_row['experimental_protein']
        if not args.rigid and score_model_args.flexible_sidechains:
            for rank, pos in enumerate(atom_pos):
                out = SidechainPDBFile(copy.deepcopy(rec_struc), data_list[rank]['flexResidues'], [atom_pos[rank]])
                if rank == 0:
                    protein_path = os.path.join(write_dir, f'rank{rank + 1}_protein.pdb')
                    out.write(protein_path)
                out.write(os.path.join(write_dir, f'rank{rank + 1}_confidence{confidence[rank]:.2f}_protein.pdb'))

        if args.relax:
            if ligand_path is None:
                raise ValueError("The ligand path is not set. This should not happen.")
            if protein_path is None:
                raise ValueError("The protein path is not set. This should not happen.")

            opt = optimize_ligand_in_pocket(
                protein_file=Path(protein_path),
                ligand_file=Path(ligand_path),
                output_file=Path(ligand_path).with_name('rank1_relaxed.sdf'),
                temp_base_dir=args.cache_path,
                add_solvent=False,
                name=orig_complex_graph.name,
            )

            energy_before = opt["energy_before"].value_in_unit(megajoule / mole)
            energy_after = opt["energy_after"].value_in_unit(megajoule / mole)

            print(
                f"{Path(ligand_path)} has been relaxed with protein {Path(protein_path)}, "
                + f"E_start: {energy_before:.2f} MJ/mol, "
                + f"E_end: {energy_after:.2f} MJ/mol, "
                + f"ΔE: {energy_after - energy_before:.2f} MJ/mol"
            )

        if args.save_visualisation:
            if confidence is not None:
                for rank, batch_idx in enumerate(re_order):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank + 1}_reverseprocess.pdb'))
            else:
                for rank, batch_idx in enumerate(ligand_pos):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank + 1}_reverseprocess.pdb'))

            if not args.rigid and score_model_args.flexible_sidechains:
                for rank, batch_idx in enumerate(re_order):
                    pdbWriter = SidechainPDBFile(copy.deepcopy(rec_struc), data_list[rank]['flexResidues'],
                                                 sidechain_visualization_list[rank])

                    pdbWriter.write(os.path.join(os.path.join(write_dir, f'rank{rank + 1}_reverseprocess_protein.pdb')))

    except Exception as e:
        print("Failed on", complex_name, type(e))
        print(e)
        stack_trace = traceback.format_exc()
        print(stack_trace)
        return 0
    finally:
        del data_list

    return +1


@ensure_device
def infer_multiple_complexes(protein_ligand_df, *args, **kwargs):
    count_succeeded = 0
    num_input = protein_ligand_df.shape[0]
    with tqdm(total=num_input, desc="Docking inference") as pbar:
        for idx, protein_ligand_info_row in protein_ligand_df.iterrows():
            complex_name = protein_ligand_info_row["complex_name"]
            pbar.set_postfix_str(s=f"Row {idx}, complex {complex_name}", refresh=True)
            count_succeeded += infer_single_complex(idx, protein_ligand_info_row, *args, **kwargs)
            pbar.update()
    return count_succeeded


def main(args):
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model_dir is None or args.filtering_dir is None:
        base_model_dir = os.path.join(args.model_cache_dir, args.tag)
        os.makedirs(base_model_dir, exist_ok=True)

        if args.model_dir is None:
            logging.debug(f'--model_dir is not set. Using tag: {args.tag}')
            args.model_dir = download_and_extract(f'{REPOSITORY_URL}/releases/download/{args.tag}/score_model.zip', base_model_dir, 'score_model')

        if args.filtering_model_dir is None:
            logging.debug(f'--filtering_model_dir is not set. Using tag: {args.tag}')
            args.filtering_model_dir = download_and_extract(f'{REPOSITORY_URL}/releases/download/{args.tag}/confidence_model.zip', base_model_dir, 'confidence_model')

    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))

    with open(f'{args.filtering_model_dir}/model_parameters.yml') as f:
        filtering_args = Namespace(**yaml.full_load(f))

    if args.protein_ligand_csv is not None:
        protein_ligand_df = load_protein_ligand_df(args.protein_ligand_csv, strict=False)
    elif args.protein_path is not None:
        # Turn single entries into a one-row dataframe
        df = pd.DataFrame({'complex_name': [args.complex_name],
                           'experimental_protein': [args.protein_path],
                           'ligand': [args.ligand],
                           'pocket_center_x': [args.pocket_center_x],
                           'pocket_center_y': [args.pocket_center_y],
                           'pocket_center_z': [args.pocket_center_z],
                           'flexible_sidechains': [args.flexible_sidechains]})
        protein_ligand_df = load_protein_ligand_df(None, df=df)
    else:
        raise ValueError('Either --protein_ligand_csv or --protein_path has to be specified')

    if "computational_protein" in protein_ligand_df.columns:
        # Don't use computational protein for inference
        print("WARN: Dropping the column 'computational_protein' from the dataframe."
              "This column is only used during training and will be ignored during inference.")
        protein_ligand_df.drop(columns=["computational_protein"], inplace=True)

    device = get_default_device()
    print(f"DiffDock-Pocket default device: {device}")

    os.makedirs(args.cache_path, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.cache_path) as dataset_cache:
        dataset_cache = os.path.join(dataset_cache, 'testset')
        test_dataset = PDBBind(transform=None,
                               protein_ligand_df=protein_ligand_df,
                               chain_cutoff=np.inf,
                               receptor_radius=score_model_args.receptor_radius,
                               cache_path=dataset_cache,
                               remove_hs=score_model_args.remove_hs,
                               max_lig_size=None,
                               c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                               matching=False,
                               keep_original=False,
                               conformer_match_sidechains=False,
                               use_original_conformer_fallback=True,
                               popsize=score_model_args.matching_popsize,
                               maxiter=score_model_args.matching_maxiter,
                               all_atoms=score_model_args.all_atoms,
                               require_ligand=True,
                               num_workers=args.num_workers,
                               keep_local_structures=args.keep_local_structures,
                               pocket_reduction=score_model_args.pocket_reduction,
                               pocket_buffer=score_model_args.pocket_buffer,
                               pocket_cutoff=score_model_args.pocket_cutoff,
                               pocket_reduction_mode=score_model_args.pocket_reduction_mode,
                               flexible_sidechains=False if args.rigid else score_model_args.flexible_sidechains,
                               flexdist=score_model_args.flexdist,
                               flexdist_distance_metric=score_model_args.flexdist_distance_metric,
                               fixed_knn_radius_graph=not score_model_args.not_fixed_knn_radius_graph,
                               knn_only_graph=not score_model_args.not_knn_only_graph,
                               include_miscellaneous_atoms=score_model_args.include_miscellaneous_atoms,
                               use_old_wrong_embedding_order=score_model_args.use_old_wrong_embedding_order)
        # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        filtering_test_dataset = filtering_complex_dict = None
        if args.filtering_model_dir is not None:
            if not (filtering_args.use_original_model_cache or filtering_args.transfer_weights): # if the filtering model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
                print('HAPPENING | filtering model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the filtering model now.')
                filtering_test_dataset = PDBBind(transform=None,
                                                 protein_ligand_df=protein_ligand_df,
                                                 chain_cutoff=np.inf,
                                                 receptor_radius=filtering_args.receptor_radius,
                                                 cache_path=dataset_cache,
                                                 remove_hs=filtering_args.remove_hs,
                                                 max_lig_size=None,
                                                 c_alpha_max_neighbors=filtering_args.c_alpha_max_neighbors,
                                                 matching=False,
                                                 keep_original=False,
                                                 conformer_match_sidechains=False,
                                                 use_original_conformer_fallback=True,
                                                 popsize=filtering_args.matching_popsize,
                                                 maxiter=filtering_args.matching_maxiter,
                                                 all_atoms=filtering_args.all_atoms,
                                                 require_ligand=True,
                                                 num_workers=args.num_workers,
                                                 keep_local_structures=args.keep_local_structures,
                                                 pocket_reduction=filtering_args.pocket_reduction,
                                                 pocket_buffer=filtering_args.pocket_buffer,
                                                 pocket_cutoff=filtering_args.pocket_cutoff,
                                                 pocket_reduction_mode=filtering_args.pocket_reduction_mode,
                                                 flexible_sidechains=False if args.rigid else filtering_args.flexible_sidechains,
                                                 flexdist=filtering_args.flexdist,
                                                 flexdist_distance_metric=filtering_args.flexdist_distance_metric,
                                                 fixed_knn_radius_graph=not filtering_args.not_fixed_knn_radius_graph,
                                                 knn_only_graph=not filtering_args.not_knn_only_graph,
                                                 include_miscellaneous_atoms=filtering_args.include_miscellaneous_atoms,
                                                 use_old_wrong_embedding_order=filtering_args.use_old_wrong_embedding_order)
                filtering_complex_dict = {d.name: d for d in filtering_test_dataset}

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.filtering_model_dir is not None:
        if filtering_args.transfer_weights:
            with open(f'{filtering_args.original_model_dir}/model_parameters.yml') as f:
                filtering_model_args = Namespace(**yaml.full_load(f))
        else:
            filtering_model_args = filtering_args

        filtering_model = get_model(filtering_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
        state_dict = torch.load(f'{args.filtering_model_dir}/{args.filtering_ckpt}', map_location=torch.device('cpu'))
        filtering_model.load_state_dict(state_dict, strict=True)
        filtering_model = filtering_model.to(device)
        filtering_model.eval()
    else:
        filtering_model = None
        filtering_args = None
        filtering_model_args = None

    t_max = 1
    tr_schedule = get_t_schedule(sigma_schedule=args.sigma_schedule, inference_steps=args.inference_steps,
                                 inf_sched_alpha=args.inf_sched_alpha, inf_sched_beta=args.inf_sched_beta,
                                 t_max=t_max)
    t_schedule = None
    print('common tr schedule', tr_schedule)

    failures, skipped = 0, 0
    print('Size of test dataset: ', len(test_dataset))

    devices = get_available_devices(max_devices=args.num_workers)
    num_processes = len(devices)
    chunks = np.array_split(test_dataset.protein_ligand_df, num_processes)

    process_chunk = functools.partial(infer_multiple_complexes, model=model, args=args,
                                      score_model_args=score_model_args,
                                      filtering_args=filtering_args, filtering_model=filtering_model,
                                      filtering_model_args=filtering_model_args,
                                      filtering_complex_dict=filtering_complex_dict,
                                      t_schedule=t_schedule, tr_schedule=tr_schedule)

    if num_processes > 1:
        print(f"Starting {num_processes} processes.")
        with torch.multiprocessing.Pool(processes=num_processes) as pool:
            a_results = []
            for device, chunk in zip(devices, chunks):
                print(f"Starting process on device {device}")
                async_result = pool.apply_async(process_chunk, (chunk,), {"device": device})
                a_results.append(async_result)

            del test_dataset.protein_ligand_df, test_dataset, chunks, chunk
            pool.close()
            pool.join()

        print(f"Completed inferences")
    else:
        num_inferences = process_chunk(test_dataset.protein_ligand_df, device=device)
        print(f"Completed {num_inferences} / {len(test_dataset)} inferences")

    print(f'Results are in {args.out_dir}')


if __name__ == "__main__":
    mp_method = "spawn"
    sharing_strategy = "file_system"
    logging.debug(f"Torch multiprocessing method: {mp_method}. Sharing strategy: {sharing_strategy}")
    torch.multiprocessing.set_start_method(mp_method)
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    parser = _get_parser()
    _args = parser.parse_args()
    with torch.no_grad():
        main(_args)
