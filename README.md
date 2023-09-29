# DiffDock-Pocket: Diffusion for Pocket-Level Docking with Sidechain Flexibility
[![python](https://img.shields.io/badge/language-python%20-%2300599C.svg?style=flat-square)](https://github.com/plainerman/DiffDock-Pocket)

DiffDock-Pocket is a binding-pocket specific molecular docking program that uses diffusion to sample ligand and sidechain poses, by Michael Plainer, Marcella Toth, Simon Dobers, Hannes Stark, Gabriele Corso, Celine Marquet, and Regina Barzilay.

In this repository, you will find the code to train a model, run inference, visualizations, and the weights we have been using to generate the numbers presented in the paper. 
This repository is originally a fork from DiffDock, so some of the commands may seem familiar - but it has been extended, adapted and changed in so many places that you cannot expect any compatability of the two programs. Be aware, and do NOT mix up these two programs!

Feel free to create any issues, or PRs if you have any problems with this repository!

![Alt Text](visualizations/pocket-visualization.gif)


## Setup Environment
You need to install the required packages

    pytorch
    pyg
    pyyaml
    scipy
    networkx
    biopython
    rdkit-pypi
    e3nn
    spyrmsd
    pandas

Then, you will also need [ESMFold](https://github.com/facebookresearch/esm). The third command will fail if you do not have `nvcc` (i.e., a properly set up GPU). If you only want to run DiffDock-Pocket on existing complexes, this is not a problem. Only a fully installed ESMFold setup allows to generate ESMFold proteins from a sequence. You can create them on a different machine / in a different environment, or simply on the [web](https://esmatlas.com/resources?action=fold).

    pip install "fair-esm[esmfold]"
    # OpenFold and its remaining dependency
    pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
    pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'


## Running DiffDock-Pocket on custom complexes
This is a very basic example, that uses a ligand and a protein. The `--keep_local_structures` does not modify the ligand and uses it to compute the pocket and flexible sidechains.

    python inference.py --protein_path example_data/3dpf_protein.pdb --ligand example_data/3dpf_ligand.sdf --batch_size 32 --samples_per_complex 40 --keep_local_structures --save_visualisation

Or for a computationally generated structure

    python inference.py --protein_path example_data/3dpf_protein_esm.pdb --ligand example_data/3dpf_ligand.sdf --batch_size 32 --samples_per_complex 40 --keep_local_structures --save_visualisation

You can easily specify the pocket and flexible sidechains if you want. Then you can also drop `--keep_local_structures`. Although it might not make a significant difference which residues are flexible for score-based models, since our model was trained only with flexibility close to the pocket center, we advise using our provided trained model only in that way to prevent out-of-distribution data.

    python inference.py --protein_path example_data/3dpf_protein.pdb --ligand example_data/3dpf_ligand.sdf --batch_size 32 --samples_per_complex 40 --save_visualisation --pocket_center_x 9.7742 --pocket_center_y 27.2863 --pocket_center_z 14.6573 --flexible_sidechains A:160-A:193-A:197-A:198-A:222-A:224-A:227

Or if you do not have a ligand at hand, you can also use the SMILES representation. Simply use

    --ligand [H]/N=C1/C(=O)C(=O)[C@@]1([H])[N@@+]1([H])C([H])([H])c2c([H])c(C([H])([H])N([H])C(=O)c3nc4sc5c(c4c(=O)n3[H])C([H])([H])C([H])([H])S(=O)(=O)C5([H])[H])c([H])c([H])c2C([H])([H])C1([H])[H]

Also see `data/protein_ligand_example_csv` with `--protein_ligand_csv protein_ligand_example_csv.csv` to specify inference on multiple complexes at once.

# Reproduce the paper numbers
Download the data and place it as described in the "Dataset" section above.

## Dataset
We are using PDBBind which you can download from many sources, e.g. [from zenedo](https://zenodo.org/record/6034088) by the authors of EquiBind. Unpack it to `data/PDBBind_processed`.

### Generate the ESM2 embeddings for the proteins
First run:

    python pdbbind_lm_embedding_preparation.py

Use the generated file `data/pdbbind_sequences.fasta` to generate the ESM2 language model embeddings and then extracting them.

    python scripts/extract.py esm2_t33_650M_UR50D pdbbind_sequences.fasta embeddings_output --repr_layers 33 --include per_tok


This generates the `embeddings_output` directory which you have to copy into the `data` folder to have `data/embeddings_output`.
Then run the command:

    python data/esm_embeddings_to_pt.py

### Training a model yourself and using those weights
Train the large score model:

    python -m train --run_name big_score_model --test_sigma_intervals --esm_embeddings_path data/esm2_3billion_embeddings.pt --cache_path data/cache --log_dir workdir --data_dir data/PDBBind_processed --lr 1e-3 --tr_sigma_min 0.1 --tr_sigma_max 5 --rot_sigma_min 0.03 --rot_sigma_max 1.55 --tor_sigma_min 0.03 --sidechain_tor_sigma_min 0.03 --batch_size 16 --ns 60 --nv 10 --num_conv_layers 6 --distance_embed_dim 64 --cross_distance_embed_dim 64 --sigma_embed_dim 64 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --sampling_alpha 1 --sampling_beta 1 --remove_hs --c_alpha_max_neighbors 24 --atom_max_neighbors 8 --receptor_radius 15 --num_dataloader_workers 1 --cudnn_benchmark --rot_alpha 1 --rot_beta 1 --tor_alpha 1 --tor_beta 1 --val_inference_freq 5 --use_ema --scheduler_patience 30 --n_epochs 750 --all_atom --sh_lmax 1 --split_train data/splits/timesplit_no_lig_overlap_train --split_val data/splits/timesplit_no_lig_overlap_val_aligned --pocket_reduction --pocket_buffer 10 --flexible_sidechains --flexdist 3.5 --flexdist_distance_metric prism --protein_file protein_esmfold_aligned_tr_fix --compare_true_protein --conformer_match_sidechains --conformer_match_score exp --match_max_rmsd 2 --use_original_conformer_fallback --use_original_conformer

The model weights are saved in the `workdir` directory.

Train a small score model with higher maximum translation sigma that will be used to generate the samples for training the confidence model:

    python -m train --run_name small_score_model --test_sigma_intervals --esm_embeddings_path data/esm2_3billion_embeddings.pt --cache_path data/cache --log_dir workdir --data_dir data/PDBBind_processed --lr 1e-3 --tr_sigma_min 0.1 --tr_sigma_max 15 --rot_sigma_min 0.03 --rot_sigma_max 1.55 --tor_sigma_min 0.03 --sidechain_tor_sigma_min 0.03 --batch_size 16 --ns 32 --nv 6 --num_conv_layers 5 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --sampling_alpha 1 --sampling_beta 1 --remove_hs --c_alpha_max_neighbors 24 --atom_max_neighbors 12 --receptor_radius 15 --num_dataloader_workers 1 --cudnn_benchmark --rot_alpha 1 --rot_beta 1 --tor_alpha 1 --tor_beta 1 --val_inference_freq 5 --use_ema --scheduler_patience 30 --n_epochs 500 --all_atom --sh_lmax 1 --split_train data/splits/timesplit_no_lig_overlap_train --split_val data/splits/timesplit_no_lig_overlap_val_aligned --pocket_reduction --pocket_buffer 10 --flexible_sidechains --flexdist 3.5 --flexdist_distance_metric prism --protein_file protein_esmfold_aligned_tr_fix --compare_true_protein --conformer_match_sidechains --conformer_match_score exp --match_max_rmsd 2 --use_original_conformer_fallback --use_original_conformer

The score model used to generate the samples to train the confidence model does not have to be the same as the score model that is used with that confidence model during inference.

Train the confidence model by running the following:

    python -m filtering.filtering_train --run_name confidence_model --original_model_dir workdir/small_score_model --ckpt best_ema_inference_epoch_model.pt --inference_steps 20 --samples_per_complex 7 --batch_size 16 --n_epochs 100 --lr 3e-4 --scheduler_patience 50 --ns 24 --nv 6 --num_conv_layers 5 --dynamic_max_cross --scale_by_sigma --dropout 0.1 --all_atoms --sh_lmax 1 --split_train data/splits/timesplit_no_lig_overlap_train --split_val data/splits/timesplit_no_lig_overlap_val_aligned --log_dir workdir --cache_path data/cache_filtering--data_dir data/PDBBind_processed --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 15 --esm_embeddings_path data/esm2_3billion_embeddings.pt --main_metric loss --main_metric_goal min --best_model_save_frequency 5 --rmsd_classification_cutoff 2 --sc_rmsd_classification_cutoff 1 --protein_file protein_esmfold_aligned_tr_fix --use_original_model_cache --pocket_reduction --pocket_buffer 10 --cache_creation_id 1 --cache_ids_to_combine 1 2 3 4


first with `--cache_creation_id 1` then `--cache_creation_id 2` etc. up to 4

Now everything is trained and you can run inference with your new model :).
