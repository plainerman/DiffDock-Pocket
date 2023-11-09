#!/bin/bash
set -e
set -x

batch_size=8
samples_per_complex=4

python inference.py --protein_path example_data/3dpf_protein.pdb --ligand example_data/3dpf_ligand.sdf --batch_size $batch_size --samples_per_complex $samples_per_complex --keep_local_structures --save_visualisation
