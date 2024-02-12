import logging
import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from esm import FastaBatchedDataset, pretrained
from rdkit.Chem import AddHs, MolFromSmiles
from torch_geometric.data import Dataset, HeteroData
import esm

from datasets.process_mols import count_pdb_warnings
from utils.utils import ensure_device

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}


def get_sequence_simple(file_path):
    # Get the approximate amino acid sequence from a PDB file
    # Don't parse the full structure, just get the sequence
    seq = []
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

                if cur_aa in three_to_one:
                    seq.append(three_to_one[cur_aa])

    return "".join(seq)


def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure(os.path.basename(file_path), file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash - .')

        if sequence is None:
            sequence = seq
        else:
            sequence += (":" + seq)

    return sequence


@count_pdb_warnings
def get_sequences(protein_files) -> List[Optional[str]]:
    new_sequences = [None]*len(protein_files)
    for ind, path in enumerate(protein_files):
        if path is not None:
            new_sequences[ind] = get_sequences_from_pdbfile(path)
    return new_sequences


def compute_esm_embeddings_df(df, column="esm_embeddings"):
    if column in df.columns:
        return df
    # Create the ESM embeddings for proteins
    print(f"Computing ESM embeddings for {len(df)} proteins...")
    esm_embeddings = esm_embeddings_from_complexes(df["complex_name"],
                                                   df["experimental_protein"])

    # Set the ESM embeddings in the dataframe
    # Need to be careful with this because the ESM embeddings are a list of lists,
    # pandas seems to want one element per row rather than one list.
    df[column] = None
    df[column] = df[column].astype(object)
    df[column] = esm_embeddings

    return df


@ensure_device
def compute_ESM_embeddings(model, alphabet, labels, sequences, device=None) -> Dict[str, torch.Tensor]:
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    include = "per_tok"
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if device is not None:
                toks = toks.to(device)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()

            del representations

    del dataset, data_loader

    return embeddings


@ensure_device
def esm_embeddings_from_complexes(complex_names, protein_files, device=None) -> List[List[torch.Tensor]]:
    model_location = "esm2_t33_650M_UR50D"
    model, alphabet = pretrained.load_model_and_alphabet(model_location)

    model.eval()
    if device is not None:
        model = model.to(device)

    protein_sequences: List[str] = get_sequences(protein_files)
    all_labels, all_sequences = [], []

    # More efficient to calculate embeddings in batches
    # So we split the chains up for each protein complex,
    # create a numbered label for each chain, and then
    # make a list of lists at the end.

    for complex_name, protein_sequence in zip(complex_names, protein_sequences):
        cur_seqs = protein_sequence.split(':')
        all_sequences.extend(cur_seqs)
        all_labels.extend([f"{complex_name}_chain_{j}" for j in range(len(cur_seqs))])

    lm_embeddings: Dict[str, torch.Tensor] = compute_ESM_embeddings(model, alphabet, all_labels, all_sequences,
                                                                    device=device)
    lm_embeddings_list: List[List[torch.Tensor]] = []

    for complex_name, protein_sequence in zip(complex_names, protein_sequences):
        cur_seqs = protein_sequence.split(':')
        cur_complex_embeddings = [lm_embeddings[f'{complex_name}_chain_{j}'] for j in range(len(cur_seqs))]
        lm_embeddings_list.append(cur_complex_embeddings)

    del model
    return lm_embeddings_list


def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None
