import os
import warnings

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

cutoff = 10
data_dir = 'data/PDB'
names = os.listdir(data_dir)
#%%
from Bio import SeqIO
biopython_parser = PDBParser()

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
'MSE':  'M', # this is almost the same AA as MET. The sulfur is just repleaced by Selen
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

sequences = []
ids = []
fail_count = 0
for name in tqdm(names):
    if name == '.DS_Store': continue
    rec_path = os.path.join(data_dir, name, f'{name}_protein.pdb')
    if not os.path.exists(rec_path):
        print(f"We are skipping {name} because there was no {name}_protein.pdb")
        fail_count += 1
        continue
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        structure = structure[0]
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
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex ', name, '. Replacing it with a dash - .')
        sequences.append(seq)
        ids.append(f'{name}_chain_{i}')
records = []
print(f'number of complexes that failed: ', fail_count)
for (index, seq) in zip(ids,sequences):
    record = SeqRecord(Seq(seq), str(index))
    record.description = ''
    records.append(record)
SeqIO.write(records, "data/pdb_sequences.fasta", "fasta")


