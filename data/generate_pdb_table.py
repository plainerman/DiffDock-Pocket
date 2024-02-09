#!/bin/env python
__doc__ = """ Generate a table of PDB and ligand files for training, based on a root directory of PDB files. 

For example:

python data/generate_pdb_table.py "data/PDBBind_atomCorrected" "esmfold_data_table" \
--experimental_name "protein_processed_fix" --computational_name "protein_esmfold_aligned_tr_fix" \
--val_frac 0.2 --seed 0
"""

import datetime
import os
from argparse import ArgumentParser

import pandas as pd


def _get_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=str, help="Root directory of PDB files")
    parser.add_argument("output_base", type=str,
                        help="Base for output files. There may be multiple output files, which will"
                             "have different suffixes.")

    parser.add_argument("--experimental_name", type=str, default="protein_processed_fix",
                        help="Expect experimental PDB files to be named {pdb_id}_{experimental_name}.pdb")

    parser.add_argument("--computational_name", type=str, default="protein_esmfold_aligned_tr_fix",
                        help="Expect computational PDB files to be named {pdb_id}_{computational_name}.pdb")

    parser.add_argument("--ligand_name", type=str, default="ligand",
                        help="Expect ligand files to be named {pdb_id}_{ligand_name}.{ligand_extension}")

    parser.add_argument("--ligand_extension", type=str, default="mol2", choices=["sdf", "mol2"],
                        help="Extension of ligand files")

    parser.add_argument("--strict", action="store_true",
                        help="Attempt to parse files before adding them to the table, rather than"
                             "just checking for existence.")

    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample a subset of the files. This should be the absolute number to sample.")

    parser.add_argument(f"--val_frac", type=float, default=None,
                        help="If provided, will generate split files for train/val with the given val fraction."
                             "If --sampling is used, this is applied *after*. "
                             "So the actual val fraction may be different.")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--verbose", action="store_true",
                        help="Print more information about the files being skipped.")

    return parser


def get_sequence_simple(file_path):
    # Get the approximate amino acid sequence from a PDB file
    # Don't parse the full structure, just get the sequence
    # BioPython parser is pretty slow and way overkill for this task
    seq = []
    last_aa = None
    last_chain = None
    lines = open(file_path, 'r').readlines()
    keep_atoms = {'ATOM', 'HETATM'}
    for line in lines:
        line = line.strip()
        words = line.split()
        if words[0] in keep_atoms:
            a_marker = words[2]
            cur_aa = words[3]
            cur_chain = words[4][0]
            # This is only usually true
            # cur_aa_pos = words[5]

            if a_marker == "CA":
                # Look at C-alpha atoms only
                if last_chain is not None and cur_chain != last_chain:
                    seq.append(':')
                last_chain = cur_chain

                if cur_aa != last_aa:
                    seq.append(cur_aa)

    return seq


def count_amino_acids(pdb_filename):
    """
    Count the number of amino acids in a PDB file.
    The BioPython parser is pretty slow and way overkill for this task.
    """
    seq = get_sequence_simple(pdb_filename)
    num_res = len(seq)
    return num_res


if __name__ == "__main__":

    parser = _get_parser()
    args = parser.parse_args()

    root_dir = args.root_dir
    max_protein_length = 1023

    def _make_full_path(_sub_dir, file_name):
        return os.path.abspath(os.path.join(root_dir, _sub_dir, file_name))

    all_rows = []
    total_count = 0
    for sub_dir in os.listdir(root_dir):
        if len(sub_dir) != 4:
            continue

        keep_row = True
        total_count += 1

        pdb_id = sub_dir
        exp_path = _make_full_path(sub_dir, f"{pdb_id}_{args.experimental_name}.pdb")
        comp_path = _make_full_path(sub_dir, f"{pdb_id}_{args.computational_name}.pdb")
        ligand_path = _make_full_path(sub_dir, f"{pdb_id}_{args.ligand_name}.{args.ligand_extension}")

        check_exist_paths = [exp_path, comp_path, ligand_path]
        for path in check_exist_paths:
            if not os.path.exists(path):
                if args.verbose:
                    print(f"Skipping {sub_dir} due to missing file {path}")
                keep_row &= False

        if args.strict:
            for protein_path in [exp_path, comp_path]:
                try:
                    cur_protein_len: int = count_amino_acids(protein_path)
                    keep_row &= cur_protein_len < max_protein_length
                except Exception as e:
                    print(f"Skipping {protein_path} due to error: {e}")
                    keep_row = False
                    continue

        if keep_row:
            cur_row = {"complex_name": pdb_id,
                       "experimental_protein": exp_path,
                       "computational_protein": comp_path,
                       "ligand": ligand_path}

            all_rows.append(cur_row)

        if args.verbose and len(all_rows) % 100 == 0:
            print(f"{datetime.datetime.now()} - Processed {len(all_rows)} rows...")

    df = pd.DataFrame(all_rows)
    if df.shape[0] == 0:
        print(f"No files found under {root_dir} with the expected names.")
        exit(0)

    df = df.sort_values("complex_name")

    if args.sample is not None and args.sample < df.shape[0]:
        df = df.sample(n=args.sample, random_state=args.seed)

    if args.val_frac is not None:
        val_frac = args.val_frac
        train_df = df.sample(frac=1-val_frac, random_state=args.seed)
        val_df = df.drop(train_df.index)

        train_df["complex_name"].to_csv(f"{args.output_base}_train", header=False, index=False)
        val_df["complex_name"].to_csv(f"{args.output_base}_val", header=False, index=False)

    print(f"Checked {total_count} directories. Writing {df.shape[0]} rows to {args.output_base}.csv")

    df.to_csv(f"{args.output_base}.csv", index=False)
