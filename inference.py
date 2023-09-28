import copy
import os
import torch
import yaml
import math

import tempfile
from argparse import ArgumentParser, Namespace, FileType

from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

from datasets.process_mols import write_mol_with_coords, parse_receptor_structure, parse_pdb_from_path
from datasets.pdbbind import PDBBind
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile, SidechainPDBFile
from tqdm import tqdm
from utils import esm as esm_utils

RDLogger.DisableLog('rdApp.*')
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--complex_name', type=str, default='unnamed_complex', help='Name that the complex will be saved with')
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path and --ligand parameters')
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein .pdb file')
# parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None') #TODO: implement protein_sequence
parser.add_argument('--ligand', type=str, default='COc(cc1)ccc1C#N', help='Either a SMILES string or the path to a molecule file that rdkit can read')
parser.add_argument('--flexible_sidechains', type=str, default=None, help='Specify which amino acids will be flexible. E.g., A:130-B:140 will make amino acid with id 130 in chain A, and id 140 in chain B flexible.')
parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')
parser.add_argument('--rigid', action='store_true', default=False, help='Override the arguments of the model and use a rigid model')

parser.add_argument('--pocket_center_x', type=float, default=None, help='The x coordinate for the pocket center')
parser.add_argument('--pocket_center_y', type=float, default=None, help='The x coordinate for the pocket center')
parser.add_argument('--pocket_center_z', type=float, default=None, help='The x coordinate for the pocket center')

parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--filtering_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--filtering_ckpt', type=str, default='best_model.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--cache_path', type=str, default='data/cache', help='Folder from where to load/restore cached dataset')
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
parser.add_argument('--skip_existing', action='store_true', default=False, help='Keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit')

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

os.makedirs(args.out_dir, exist_ok=True)

with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))


if args.filtering_model_dir is not None:
    with open(f'{args.filtering_model_dir}/model_parameters.yml') as f:
        filtering_args = Namespace(**yaml.full_load(f))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_none(x):
    if isinstance(x, list):
        return [to_none(y) for y in x]
    return None if math.isnan(x) else x


def center_to_torch(x, y, z):
    if x is None or y is None or z is None:
        return None
    return torch.tensor([x, y, z], dtype=torch.float32)


if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    complex_name_list = df['complex_name'].tolist()
    protein_path_list = df['protein_path'].tolist()
    ligand_descriptions = df['ligand'].tolist()
    pocket_centers_x = to_none(df['pocket_center_x'].tolist())
    pocket_centers_y = to_none(df['pocket_center_y'].tolist())
    pocket_centers_z = to_none(df['pocket_center_z'].tolist())
    flexible_sidechains = to_none(df['flexible_sidechains'].tolist())
    ligand_descriptions = [(ligand, center_to_torch(x, y, z)) for ligand, x, y, z in zip(ligand_descriptions, pocket_centers_x, pocket_centers_y, pocket_centers_z, flexible_sidechains)]

elif args.protein_path is not None:
    complex_name_list = [args.complex_name]
    protein_path_list = [args.protein_path]
    ligand_descriptions = [(args.ligand, center_to_torch(args.pocket_center_x, args.pocket_center_y, args.pocket_center_z), args.flexible_sidechains)]
else:
    #TODO: add support for protein sequence to pdb file?
    raise ValueError('Either --protein_ligand_csv or --protein_path has to be specified')

# create the ESM embeddings for the protein_path_list
esm_embeddings = None if score_model_args.esm_embeddings_path is None else esm_utils.esm_embeddings_from_complexes(complex_name_list, protein_path_list)

os.makedirs(args.cache_path, exist_ok=True)
with tempfile.TemporaryDirectory(dir=args.cache_path) as dataset_cache:
    dataset_cache = os.path.join(dataset_cache, 'testset')
    test_dataset = PDBBind(transform=None, root='',
                           protein_path_list=protein_path_list, ligand_descriptions=ligand_descriptions,
                           chain_cutoff=np.inf,
                           receptor_radius=score_model_args.receptor_radius,
                           cache_path=dataset_cache,
                           remove_hs=score_model_args.remove_hs,
                           max_lig_size=None,
                           c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                           matching=False,
                           keep_original=False,
                           popsize=score_model_args.matching_popsize,
                           maxiter=score_model_args.matching_maxiter,
                           all_atoms=score_model_args.all_atoms,
                           esm_embeddings_path=esm_embeddings,
                           require_ligand=True,
                           num_workers=args.num_workers,
                           keep_local_structures=args.keep_local_structures,
                           pocket_reduction=score_model_args.pocket_reduction, pocket_buffer=score_model_args.pocket_buffer,
                           pocket_cutoff=score_model_args.pocket_cutoff,
                           pocket_reduction_mode=score_model_args.pocket_reduction_mode,
                           flexible_sidechains=False if args.rigid else score_model_args.flexible_sidechains,
                           flexdist=score_model_args.flexdist,
                           flexdist_distance_metric=score_model_args.flexdist_distance_metric,
                           fixed_knn_radius_graph=not score_model_args.not_fixed_knn_radius_graph,
                           knn_only_graph=not score_model_args.not_knn_only_graph,
                           include_miscellaneous_atoms=score_model_args.include_miscellaneous_atoms,
                           use_old_wrong_embedding_order=score_model_args.use_old_wrong_embedding_order)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args.filtering_model_dir is not None:
        if not (filtering_args.use_original_model_cache or filtering_args.transfer_weights): # if the filtering model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
            print('HAPPENING | filtering model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the filtering model now.')
            filtering_test_dataset = PDBBind(transform=None, root='',
                                             protein_path_list=protein_path_list, ligand_descriptions=ligand_descriptions,
                                             chain_cutoff=np.inf,
                                             receptor_radius=filtering_args.receptor_radius,
                                             cache_path=dataset_cache,
                                             remove_hs=filtering_args.remove_hs,
                                             max_lig_size=None,
                                             c_alpha_max_neighbors=filtering_args.c_alpha_max_neighbors,
                                             matching=False,
                                             keep_original=False,
                                             popsize=filtering_args.matching_popsize,
                                             maxiter=filtering_args.matching_maxiter,
                                             all_atoms=filtering_args.all_atoms,
                                             esm_embeddings_path=esm_embeddings,
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
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
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
rot_schedule = tr_schedule
tor_schedule = tr_schedule
sidechain_tor_schedule = tr_schedule
print('common t schedule', tr_schedule)

failures, skipped, confidences_list = 0, 0, []
N = args.samples_per_complex
print('Size of test dataset: ', len(test_dataset))
for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
    if filtering_model is not None and not (filtering_args.use_original_model_cache or filtering_args.transfer_weights) and orig_complex_graph.name[0] not in filtering_complex_dict.keys():
        skipped += 1
        print(f"HAPPENING | The filtering dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex.")
        continue

    try:
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        write_dir = f'{args.out_dir}/index{idx}_{data_list[0]["name"][0].replace("/", "-")}'
        if os.path.exists(write_dir) and args.skip_existing: continue

        randomize_position(data_list, score_model_args.no_torsion, args.no_random, score_model_args.tr_sigma_max,
                           flexible_sidechains=False if args.rigid else score_model_args.flexible_sidechains)

        pdb = None
        lig = orig_complex_graph.mol[0]
        if args.save_visualisation:
            visualization_list = []
            sidechain_visualization_list = []

            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)
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

        if filtering_model is not None and not (filtering_args.use_original_model_cache or filtering_args.transfer_weights):
            filtering_data_list = [copy.deepcopy(filtering_complex_dict[orig_complex_graph.name[0]]) for _ in range(N)]
        else:
            filtering_data_list = None

        data_list, confidence = sampling(data_list=data_list, model=model,
                                         inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule, sidechain_tor_schedule=sidechain_tor_schedule,
                                         device=device, t_to_sigma=t_to_sigma, model_args=score_model_args, no_random=args.no_random,
                                         ode=args.ode, visualization_list=visualization_list, sidechain_visualization_list=sidechain_visualization_list,
                                         confidence_model=filtering_model, filtering_data_list=filtering_data_list, filtering_model_args=filtering_model_args,
                                         asyncronous_noise_schedule=score_model_args.asyncronous_noise_schedule, t_schedule=t_schedule,
                                         batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise,
                                         temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot,
                                                        args.temp_sampling_tor, args.temp_sampling_sc_tor],
                                         temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor,
                                                   args.temp_psi_sc_tor],
                                         flexible_sidechains=False if args.rigid else score_model_args.flexible_sidechains)
        ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])
        atom_pos = np.asarray([complex_graph['atom'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

        rec_struc = parse_pdb_from_path(protein_path_list[idx])
        # Similarly as in pdb preprocess, we sort the atoms by the name and put hydrogens at the end
        for res in rec_struc.get_residues():
            res.child_list.sort(key=lambda atom: PDBBind.order_atoms_in_residue(res, atom))
            res.child_list = [atom for atom in res.child_list if
                              not score_model_args.remove_hs or atom.element != 'H']

        if confidence is not None and isinstance(filtering_args.rmsd_classification_cutoff, list):
            confidence = confidence[:,0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            confidences_list.append(confidence)
            ligand_pos = ligand_pos[re_order]
            atom_pos = atom_pos[re_order]

        os.makedirs(write_dir, exist_ok=True)
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)
            if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
            write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))

        if not args.rigid and score_model_args.flexible_sidechains:
            for rank, pos in enumerate(atom_pos):
                out = SidechainPDBFile(copy.deepcopy(rec_struc), data_list[rank]['flexResidues'], [atom_pos[rank]])
                if rank == 0: out.write(os.path.join(write_dir, f'rank{rank+1}_protein.pdb'))
                out.write(os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}_protein.pdb'))

        if args.save_visualisation:
            if confidence is not None:
                for rank, batch_idx in enumerate(re_order):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
            else:
                for rank, batch_idx in enumerate(ligand_pos):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

            if not args.rigid and score_model_args.flexible_sidechains:
                for rank, batch_idx in enumerate(re_order):
                    pdbWriter = SidechainPDBFile(copy.deepcopy(rec_struc), data_list[rank]['flexResidues'],
                                                 sidechain_visualization_list[rank])

                    pdbWriter.write(os.path.join(os.path.join(write_dir, f'rank{rank+1}_reverseprocess_protein.pdb')))
    except Exception as e:
        print("Failed on", orig_complex_graph["name"], type(e))
        print(e)
        failures += 1

print(f'Failed for {failures} complexes')
print(f'Skipped {skipped} complexes')
print(f'Results are in {args.out_dir}')
