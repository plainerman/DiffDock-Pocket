#!/usr/bin/env python3

"""
Given the original (full) rigid receptor and the output of --out_flex,
which contains side chain coordinates from flexible docking with smina/gnina,
output the full restored receptors with the flexible residues re-inserted.
   
This is a reworked version from gnina and is more stable and can handle matching apo/holo (if they have equal atoms).
This adapted version does NOT insert all residues, but only those that are flexible in the docking.
"""
import traceback
import prody
import argparse
import os
from Bio.PDB import PDBParser, PDBIO
from tqdm import tqdm
from contextlib import nullcontext
import numpy as np

parser = argparse.ArgumentParser(
    description="Assemble full receptor from flexible docking results."
)
parser.add_argument('--complex_names_path', type=str, default='data/splits/timesplit_test', help='')
parser.add_argument("--rigid", type=str, help="Rigid receptor (pdb) dir")
parser.add_argument("--flex", type=str, help="Flexible sidechains from docking (pdb) dir")
parser.add_argument("--protein_suffix", type=str, help="Suffix of the rigid protein")
parser.add_argument("--out", type=str, help="Output file dir")
parser.add_argument("--out_suffix", type=str, default="", help="")
parser.add_argument("--out_suffix_full", type=str, default="_full", help="The same as out_suffix but with all residues; flexible and rigid")
parser.add_argument("--out_suffix_orig", type=str, default="_orig", help="")
parser.add_argument("--out_suffix_gnina", type=str, default="_gnina", help="")
parser.add_argument("--fail_log", type=str, default="fail.log", help="")
parser.add_argument("--original_gnina_suffix", type=str, default=None,
                    help="The suffix for the proteins used as the original gnina input."
                         "If none is specified the protein_suffix is used.")
args = parser.parse_args()

backbone = {
    "N",
    "O",
    "H",
    "HN",
}  # C and CA are included in the flex part, but don't move


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def run(names):
    fail = []
    for i, name in tqdm(enumerate(names), total=len(names)):
        try:
            rigidname = os.path.join(args.rigid, name, f'{name}{args.protein_suffix}.pdb')
            flexname = os.path.join(args.flex, name, f'{name}_flex_residues.pdb')
            out_path = os.path.join(args.out, name)
            os.makedirs(out_path, exist_ok=True)

            parser = PDBParser()

            target_structure = parser.get_structure('random_id', rigidname)[0]

            original_gnina_suffix = args.protein_suffix if args.original_gnina_suffix is None else args.original_gnina_suffix
            gnina_input_name = os.path.join(args.rigid, name, f'{name}{original_gnina_suffix}.pdb')
            input_structure = parser.get_structure('random_id', gnina_input_name)[0]

            # We use prody because BioPDB does not support multiple models in one PDB file
            flex = prody.parsePDB(flexname)
            flexres = list(set(zip(flex.getChids(), flex.getResnums(), flex.getIcodes())))
            flexres.sort()

            # Print flexible residues
            print("Flexres:", flexres)

            target_structure_residues = list(target_structure.get_residues())
            input_structure_residues = list(input_structure.get_residues())

            assert len(target_structure_residues) == len(
                input_structure_residues), f'len(target_structure_residues) != len(input_structure_residues) for {name}'

            writer = PDBIO()

            with open(os.path.join(out_path, f'{name}{args.out_suffix}.pdb'), "w") as output_file, \
                    open(os.path.join(out_path, f'{name}{args.out_suffix_orig}.pdb'), "w") as output_file_orig, \
                    open(os.path.join(out_path, f'{name}{args.out_suffix}{args.out_suffix_full}.pdb'), "w") as output_file_full:
                with open(os.path.join(out_path, f'{name}{args.out_suffix_gnina}.pdb'), "w")\
                        if args.original_gnina_suffix is not None else nullcontext() as output_file_gnina:

                    for ci in range(flex.numCoordsets()):  # Loop over different MODELs (MODEL/ENDMDL)
                        output_file_orig.write(f"MODEL        {ci}\n")
                        output_file_full.write(f"MODEL        {ci}\n")
                        output_file.write(f"MODEL        {ci}\n")
                        if args.original_gnina_suffix is not None:
                            output_file_gnina.write(f"MODEL        {ci}\n")

                        # we store the changes in a dictionary so that we can revert them
                        # we need the same target_residue for the next MODEL iteration
                        reverts = []

                        for (chain_id, resnum, icode) in flexres:
                            resatoms = flex[chain_id].select("resnum %d and not name H" % resnum)

                            matched_residue = input_structure[chain_id][int(resnum)]
                            target_residue = target_structure_residues[input_structure_residues.index(matched_residue)]

                            assert target_residue.get_resname() == matched_residue.get_resname(), f'target_residue.get_resname() != matched_residue.get_resname() for {name}'

                            # The ordering between the matched_residue and target_residue might be different
                            assert sorted([a.name for a in target_residue.get_atoms() if a.element != 'H']) ==\
                                   sorted([a.name for a in matched_residue.get_atoms() if a.element != 'H']), f'Atoms do not match between the two pdb files for {name} and {(chain_id, resnum, icode)}'

                            atoms_to_set = [a.name for a in matched_residue.get_atoms() if a.element != 'H' and a.name not in backbone]
                            # ensure that the names are unique
                            assert len(atoms_to_set) == len(set(atoms_to_set)), f'Not all atoms are unique for {name} and {(chain_id, resnum, icode)}'
                            # ensure that all positions are present
                            assert len(atoms_to_set) == len(resatoms), f'Not all atoms were matched for {name} and {(chain_id, resnum, icode)}'

                            matched_residue_names = [a.name for a in matched_residue.get_atoms() if a.element != 'H']

                            # remove the hydrogen atoms
                            target_residue.child_list = [a for a in target_residue.child_list if a.element != 'H']

                            # sort the atoms in target residue the same way as the atoms in the matched residue
                            target_residue.child_list.sort(key=lambda atom: matched_residue_names.index(atom.name))

                            # Write the current ORIGINAL residue to the output PDB file
                            writer.set_structure(target_residue)
                            writer.save(output_file_orig, write_end=False, preserve_atom_numbering=True)

                            # Write the current original gnina input residue to the output PDB file
                            if args.original_gnina_suffix is not None:
                                writer.set_structure(matched_residue)
                                writer.save(output_file_gnina, write_end=False, preserve_atom_numbering=True)

                            # override the atom positions, but do not go by index but atom name because we reordered
                            revert = {}
                            for aname, atom in zip(atoms_to_set, resatoms):
                                revert[aname] = target_residue[aname].coord.copy()
                                target_residue[aname].set_coord(atom.getCoordsets(ci))

                            reverts.append((target_residue, revert))

                            # Write the current residue to the output PDB file
                            writer.set_structure(target_residue)
                            writer.save(output_file, write_end=False, preserve_atom_numbering=True)

                        writer.set_structure(target_structure)
                        writer.save(output_file_full, write_end=False, preserve_atom_numbering=True)

                        for target_residue, revert in reverts:
                            for aname, pos in revert.items():
                                target_residue[aname].set_coord(pos)

                        output_file_orig.write("ENDMDL\n")
                        output_file.write("ENDMDL\n")
                        output_file_full.write("ENDMDL\n")
                        if args.original_gnina_suffix is not None:
                            output_file_gnina.write("ENDMDL\n")

        except Exception as e:
            print("Failed on protein:", name)
            print(e)
            print(traceback.format_exc())
            fail.append(name)

    print("Failed on proteins:", fail)
    with open(os.path.join(args.out, args.fail_log), 'w') as fp:
        fp.write("\n".join(str(item) for item in fail))


if __name__ == "__main__":
    names = read_strings_from_txt(args.complex_names_path)
    run(names)
