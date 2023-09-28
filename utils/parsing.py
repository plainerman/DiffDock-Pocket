
from argparse import ArgumentParser,FileType

def parse_train_args():

    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--log_dir', type=str, default='workdir', help='Folder in which to save model and logs')
    parser.add_argument('--restart_dir', type=str, help='Folder of previous training model from which to restart')
    parser.add_argument('--cache_path', type=str, default='data/cacheNew', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--data_dir', type=str, default='data/PDBBind_processed/', help='Folder containing original structures')
    parser.add_argument('--split_train', type=str, default='data/splits/timesplit_no_lig_overlap_train', help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='data/splits/timesplit_no_lig_overlap_val', help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='data/splits/timesplit_test', help='Path of file defining the split')
    parser.add_argument('--test_sigma_intervals', action='store_true', default=False, help='Whether to log loss per noise interval')
    parser.add_argument('--val_inference_freq', type=int, default=5, help='Frequency of epochs for which to run expensive inference on val data')
    parser.add_argument('--train_inference_freq', type=int, default=None, help='Frequency of epochs for which to run expensive inference on train data')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps for inference on val')
    parser.add_argument('--num_inference_complexes', type=int, default=100, help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument('--inference_earlystop_metric', type=str, default='valinf_rmsds_lt2', help='This is the metric that is addionally used when val_inference_freq is not None')
    parser.add_argument('--inference_earlystop_goal', type=str, default='max', help='Whether to maximize or minimize metric')
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--project', type=str, default='ligbind_tr', help='')
    parser.add_argument('--run_name', type=str, default='', help='')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='CUDA optimization parameter for faster training')
    parser.add_argument('--num_dataloader_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='pin_memory arg of dataloader')
    parser.add_argument('--dataloader_drop_last', action='store_true', default=False, help='drop_last arg of dataloader')


    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--scheduler', type=str, default=None, help='LR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of the LR scheduler')
    parser.add_argument('--adamw', action='store_true', default=False, help='Use AdamW optimizer instead of Adam')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--restart_lr', type=float, default=None, help='If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.')
    parser.add_argument('--w_decay', type=float, default=0.0, help='Weight decay added to loss')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--use_ema', action='store_true', default=False, help='Whether or not to use ema for the model weights')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='decay rate for the exponential moving average model parameters ')

    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=0, help='If positive, the number of training and validation complexes is capped') # TODO change
    parser.add_argument('--all_atoms', action='store_true', default=False, help='Whether to use the all atoms model')
    parser.add_argument('--multiplicity', type=int, default=1, help='')
    parser.add_argument('--chain_cutoff', type=float, default=10, help='Cutoff on whether to include non-interacting chains')
    parser.add_argument('--receptor_radius', type=float, default=30, help='Cutoff on distances for receptor edges')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='Maximum number of neighbors for each residue')
    parser.add_argument('--atom_max_neighbors', type=int, default=8, help='Maximum number of atom neighbours for receptor')
    parser.add_argument('--matching_popsize', type=int, default=20, help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=20, help='Differential evolution maxiter parameter in matching')
    parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms in ligand')
    parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='Number of conformers to match to each ligand')
    parser.add_argument('--esm_embeddings_path', type=str, default=None, help='If this is set then the LM embeddings at that path will be used for the receptor features')
    parser.add_argument('--not_full_dataset', action='store_true', default=False, help='') # TODO remove
    parser.add_argument('--pocket_mode_graph', action='store_true', default=False, help='Only use the AAs around the pocket when constructing the graph')
    parser.add_argument('--use_full_size_protein_file', action='store_true', default=False, help='')
    parser.add_argument('--pocket_reduction', action='store_true', default=False, help='Remove atoms from receptor that are not in the binding pocket')
    parser.add_argument('--pocket_reduction_mode', type=str, default='center-dist', choices=['center-dist', 'ligand-dist'], help='Defines how it is determined which atoms are kept')
    parser.add_argument('--pocket_buffer', type=float, default=10, help='Buffer that will be added to the radius of the pocket')
    parser.add_argument('--pocket_cutoff', type=float, default=5, help='This defines the center of the pocket. Mean of all C-Alpha of receptor within max pocket_cutoff distance to any ligand atom')
    parser.add_argument('--skip_no_pocket_atoms', action='store_true', default=False, help='If there are no atoms in the --pocket_cutoff, then those complexes will be skipped.')
    parser.add_argument('--not_fixed_knn_radius_graph', action='store_true', default=False, help='Use knn graph and radius graph with closest neighbors instead of random ones as with radius_graph')
    parser.add_argument('--not_knn_only_graph', action='store_true', default=False, help='Use knn graph only and not restrict to a specific radius')
    parser.add_argument('--include_miscellaneous_atoms', action='store_true', default=False, help='include non amino acid atoms for the receptor')
    parser.add_argument('--use_old_wrong_embedding_order', action='store_true', default=False, help='for backward compatibility to prevent the chain embedding order')
    parser.add_argument('--match_protein_file', type=str, default='protein_processed_fix', help='specify the protein we will use to conformer match the --protein_file argument')
    parser.add_argument('--conformer_match_sidechains', action='store_true', default=False, help='Conformer match the sidechains from --protein_file with the --match_protein_file')
    parser.add_argument('--conformer_match_score', type=str, default="dist", help='The scoring function used for conformer matching. Can be either "dist", "nearest" or "exp". All take the distance to the holo structure, nearest and exp also optimize steric clashes. Nearest takes the closest steric clash, exp weights the steric clashes with something similar to an rbf kernel.')
    parser.add_argument('--compare_true_protein', action='store_true', default = False, help="whether to calculate the rmsd to the holo structure (i.e., match_protein_file). this is only possible with flexible sidechains and if the proein_file is an apo structure. This is only applied to the validation set")
    parser.add_argument('--match_max_rmsd', type=float, default=2.0, help='Specify the maximum RMSD when conformer matching sidechains. This RMSD will only be calculated in the pocket with pocket_buffer. This parameter only influences the training set, and has no impact on validation.')
    parser.add_argument('--use_original_conformer', action='store_true', default=False, help='use the original conformer structure for training if the matching rmsd is further away than match_max_rmsd value')
    parser.add_argument('--use_original_conformer_fallback', action='store_true', default=False, help='use the original conformer structure for training if the protein_file does not exist. This only effects training.')

    # Diffusion
    parser.add_argument('--tr_weight', type=float, default=0.25, help='Weight of translation loss')
    parser.add_argument('--rot_weight', type=float, default=0.25, help='Weight of rotation loss')
    parser.add_argument('--tor_weight', type=float, default=0.25, help='Weight of torsional loss')
    parser.add_argument('--sc_tor_weight', type=float, default=0.25, help='Weight of torsional loss')
    parser.add_argument('--confidence_weight', type=float, default=0.33, help='Weight of confidence loss')
    parser.add_argument('--rot_sigma_min', type=float, default=0.1, help='Minimum sigma for rotational component')
    parser.add_argument('--rot_sigma_max', type=float, default=1.65, help='Maximum sigma for rotational component')
    parser.add_argument('--tr_sigma_min', type=float, default=0.1, help='Minimum sigma for translational component')
    parser.add_argument('--tr_sigma_max', type=float, default=30, help='Maximum sigma for translational component')
    parser.add_argument('--tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional component')
    parser.add_argument('--tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional component')
    parser.add_argument('--sidechain_tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional components of sidechains')
    parser.add_argument('--sidechain_tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional components of sidechains')
    parser.add_argument('--no_torsion', action='store_true', default=False, help='If set only rigid matching')

    parser.add_argument('--flexible_sidechains', action='store_true', default=False, help='Diffuse over side chain torsions for residues within flexdist of pocket')
    parser.add_argument('--flexdist', type=float, default=3.5, help='If a residue has at least one atom within flexdist of the pocket, it will be made flexible')
    parser.add_argument('--flexdist_distance_metric', type=str, default='L2', help='Distance metric used to select residues within flexdist to pocket center')
    parser.add_argument('--separate_noise_schedule', action='store_true', default=False, help='Use different t for tr, rot, and tor')
    parser.add_argument('--asyncronous_noise_schedule', action='store_true', default=False, help='')
    parser.add_argument('--sampling_alpha', type=float, default=1, help='Alpha parameter of beta distribution for sampling t')
    parser.add_argument('--sampling_beta', type=float, default=1, help='Beta parameter of beta distribution for sampling t')
    parser.add_argument('--rot_alpha', type=float, default=1,help='Alpha parameter of beta distribution for sampling t')
    parser.add_argument('--rot_beta', type=float, default=1,help='Beta parameter of beta distribution for sampling t')
    parser.add_argument('--tor_alpha', type=float, default=1,help='Alpha parameter of beta distribution for sampling t')
    parser.add_argument('--tor_beta', type=float, default=1,help='Beta parameter of beta distribution for sampling t')
    parser.add_argument('--sidechain_tor_alpha', type=float, default=1,help='Alpha parameter of beta distribution for sampling t')
    parser.add_argument('--sidechain_tor_beta', type=float, default=1,help='Beta parameter of beta distribution for sampling t')
    parser.add_argument('--inf_pocket_knowledge', action='store_true', default=False, help='')
    parser.add_argument('--inf_pocket_cutoff', type=float, default=5, help='')

    # Model
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--norm_by_sigma', action='store_true', default=False, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=32, help='Embedding size for the distance')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='Embeddings size for the cross distance')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80, help='Maximum cross distance in case not dynamic')
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='Whether to use the dynamic distance cutoff')
    parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    parser.add_argument('--smooth_edges', action='store_true', default=False, help='Whether to apply additional smoothing weight to edges')
    parser.add_argument('--odd_parity', action='store_true', default=False, help='Whether to impose odd parity in output')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='Type of diffusion time embedding')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Size of the embedding of the diffusion time')
    parser.add_argument('--embedding_scale', type=int, default=1000, help='Parameter of the diffusion time embedding')
    parser.add_argument('--sh_lmax', type=int, default=2, help='Size of the embedding of the diffusion time')
    parser.add_argument('--use_old_atom_encoder', action='store_true', default=False, help='option to use old atom encoder for backward compatibility')

    # Confidence Predictor in Model
    parser.add_argument('--include_confidence_prediction', action='store_true', default=False,help='Whether to predict an additional confidence metric for each predicted structure')
    parser.add_argument('--high_confidence_threshold', type=float, default=5.0,help='If this is 0 then the confidence predictor tries to predict the centroid_distance. Otherwise it is the Ångström below which a prediction is labeled as good for supervising the confidence predictor')
    parser.add_argument('--tr_only_confidence', action='store_true', default=True, help='Whether to only supervise the confidence predictor with the translation')
    parser.add_argument('--confidence_no_batchnorm', action='store_true', default=False, help='')
    parser.add_argument('--confidence_dropout', type=float, default=0.0, help='MLP dropout in confidence readout')

    parser.add_argument('--not_fixed_center_conv', action='store_true', default=False, help='')
    parser.add_argument('--protein_file', type=str, default='protein_processed', help='')
    parser.add_argument('--no_aminoacid_identities', action='store_true', default=False, help='')

    args = parser.parse_args()

    if args.flexible_sidechains and not args.all_atoms:
        raise ValueError("--all_atoms needs to be activated if --flexible_sidechains is used")
    if args.pocket_reduction and args.flexible_sidechains:
        if args.flexdist > args.pocket_buffer:
            print("WARN: The specified flexdist of", args.flexdist, "is larger than the pocket_buffer of", args.pocket_buffer)

    if args.compare_true_protein and not args.flexible_sidechains:
        raise ValueError("Comparing to a true protein file is only meaningful when there are flexible sidechains")

    if args.conformer_match_score != "dist" and args.conformer_match_score != "nearest" and args.conformer_match_score != "exp":
        raise ValueError("Conformer match score must be either 'dist', 'nearest' or 'exp")

    return args
