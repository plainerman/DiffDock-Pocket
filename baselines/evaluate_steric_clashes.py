import os
from argparse import ArgumentParser

import numpy as np

from tqdm import tqdm
from datasets.process_mols import read_molecule
from datasets.steric_clash import get_steric_clash_atom_pairs
from Bio.PDB import PDBParser

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/PDBBind_processed/', help='Folder containing original structures')
parser.add_argument('--protein_file', type=str, default='protein_processed_fix', help='Protein file suffix')
parser.add_argument('--remove_ligand_hydrogens', action='store_true', default=False, help='')
parser.add_argument('--remove_receptor_hydrogens', action='store_true', default=False, help='')
parser.add_argument('--complex_names_path', type=str, default='data/splits/timesplit_test', help='')
parser.add_argument('--limit_complexes', type=int, default=0, help='')

args = parser.parse_args()

parser = PDBParser(QUIET=True)

logs = {'clashes': [], 'clashes_receptor': {}, 'clashes_ligand': {}}

def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]

def read_mol(pdbbind_dir, name, remove_hs):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.sdf'), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs


def compute_steric_clashes(pdb_id, remove_receptor_Hs, remove_ligand_Hs):
    receptor = parser.get_structure('random_id', os.path.join(args.data_dir, pdb_id, f'{pdb_id}_{args.protein_file}.pdb'))[0]
    atoms = list(receptor.get_atoms())
    rec_elements = [a.element for a in atoms]
    rec_pos = np.array([a.coord for a in atoms])
    if remove_receptor_Hs:
        atoms = [a for a in atoms if a.element != 'H']
        rec_elements = [a.element for a in atoms]
        rec_pos = np.array([a.coord for a in atoms])

    ligs = read_mols(args.data_dir, pdb_id, remove_hs=remove_ligand_Hs)
    for i, mol in enumerate(ligs):
        lig_pos = np.array(mol.GetConformer().GetPositions())
        lig_elements = [a.GetSymbol() for a in mol.GetAtoms()]

        clashes = get_steric_clash_atom_pairs(rec_pos[None, :], lig_pos[None, :], rec_elements, lig_elements)

        logs['clashes'].append(clashes.sum())

        # print('Ligand', i, 'has', clashes.sum(), 'clashes')
        for _, clash_rec, clash_lig in zip(*np.where(clashes)):
            rec_clash_elem = rec_elements[clash_rec]
            lig_clash_elem = lig_elements[clash_lig]

            if rec_clash_elem not in logs['clashes_receptor']:
                logs['clashes_receptor'][rec_clash_elem] = 0
            if lig_clash_elem not in logs['clashes_ligand']:
                logs['clashes_ligand'][lig_clash_elem] = 0

            logs['clashes_receptor'][rec_clash_elem] += 1
            logs['clashes_ligand'][lig_clash_elem] += 1
            # print(f'{rec_clash_elem} ({clash_rec}) with {lig_clash_elem} ({clash_lig})')


if __name__ == '__main__':
    complexes = read_strings_from_txt(args.complex_names_path)
    if args.limit_complexes > 0:
        complexes = complexes[:args.limit_complexes]

    errors = 0
    for c in tqdm(complexes):
        try:
            compute_steric_clashes(c, args.remove_receptor_hydrogens, args.remove_ligand_hydrogens)
        except Exception as e:
            print(e)
            errors += 1

    logs['clashes'] = np.array(logs['clashes'])

    print("Total errors:", errors)
    print("Total ligands:", len(logs['clashes']))
    print("Total steric clashes:", logs['clashes'].sum())
    print("Average number of steric clashes:", logs['clashes'].mean().round(2))
    print("Average number of steric clashes of the cases with clashes:", logs['clashes'][logs['clashes'] > 0].mean().round(2))
    print(f"Fraction of dockings with steric clash: {round(100 * (logs['clashes'] > 0).sum() / len(logs['clashes']), 2)}%")

    logs['clashes_receptor'] = dict(sorted(logs['clashes_receptor'].items(), key=lambda item: -item[1]))
    logs['clashes_ligand'] = dict(sorted(logs['clashes_ligand'].items(), key=lambda item: -item[1]))
    print(logs['clashes_receptor'])
    print(logs['clashes_ligand'])
