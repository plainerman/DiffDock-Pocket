from typing import List

from Bio.PDB import PDBIO
from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
import rdkit.Chem 
from rdkit import Geometry
from collections import defaultdict
import copy
import numpy as np
import torch
import Bio 
from torch import tensor

from utils.torsion import get_sidechain_rotation_mask


class PDBFile:
    def __init__(self, mol):
        self.parts = defaultdict(dict)
        self.mol = copy.deepcopy(mol)
        [self.mol.RemoveConformer(j) for j in range(mol.GetNumConformers()) if j]        
    def add(self, coords, order, part=0, repeat=1):
        if type(coords) in [rdkit.Chem.Mol, rdkit.Chem.RWMol]:
            block = MolToPDBBlock(coords).split('\n')[:-2]
            self.parts[part][order] = {'block': block, 'repeat': repeat}
            return
        elif type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.double().numpy()
        for i in range(coords.shape[0]):
            self.mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
        block = MolToPDBBlock(self.mol).split('\n')[:-2]
        self.parts[part][order] = {'block': block, 'repeat': repeat}
        
    def write(self, path=None, limit_parts=None):
        is_first = True
        str_ = ''
        for part in sorted(self.parts.keys()):
            if limit_parts and part >= limit_parts:
                break
            part = self.parts[part]
            keys_positive = sorted(filter(lambda x: x >=0, part.keys()))
            keys_negative = sorted(filter(lambda x: x < 0, part.keys()))
            keys = list(keys_positive) + list(keys_negative)
            for key in keys:
                block = part[key]['block']
                times = part[key]['repeat']
                for _ in range(times):
                    if not is_first:
                        block = [line for line in block if 'CONECT' not in line]
                    is_first = False
                    str_ += 'MODEL\n'
                    str_ += '\n'.join(block)
                    str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)


class SidechainPDBFile:
    # implements visualization for a receptor with different sidechain conformations 
    def __init__(self, rec_structure: Bio.PDB.Structure.Structure, flex_residues_info, conformations: List[tensor]) -> None:
        self.rec_structure = rec_structure
        self.flex_residues_info = flex_residues_info
        self.sidechain_conformations = conformations

    def write(self, outfile: str):
        pending_ids = self.flex_residues_info.pdbIds if hasattr(self.flex_residues_info, 'pdbIds') else []
        pending_ids = list(map(tuple, pending_ids))
        flex_res_to_index = [] if len(pending_ids) == 0 else [0] + self.flex_residues_info.residueNBondsMapping.cumsum(axis=0).cpu().tolist()

        writer = PDBIO()
        # Open the output PDB file
        with open(outfile, "w") as output_file:
            # Loop through each frame of the animation
            for modelIndex, sidechain_conformation in enumerate(self.sidechain_conformations):

                #atoms = [atom for atom in self.rec_structure.get_atoms() if atom.element != "H"] if self.remove_hs else list(self.rec_structure.get_atoms())

                # Update the coordinates of each atom in the structure
                done_ids = []
                for res in self.rec_structure.get_residues():
                    full_id = res.get_full_id()
                    simplified_res_id = (full_id[1], full_id[2], full_id[3][1])
                    if simplified_res_id in pending_ids:
                        # here we do a very tricky mapping to go from the atom list we have right now, to each atom of this residue
                        # the problem is that if pocket reduction is enabled, those two do not match up otherwise
                        sidechain_index = pending_ids.index(simplified_res_id)
                        new_sidechain_positions = flex_res_to_index[sidechain_index], flex_res_to_index[sidechain_index + 1]
                        flex_res_subcomponents = self.flex_residues_info.subcomponentsMapping[new_sidechain_positions[0]:new_sidechain_positions[1]]
                        flex_res_subcomponents = [self.flex_residues_info.subcomponents[m[0]:m[1]].cpu().tolist() for m in flex_res_subcomponents]

                        cur_res_subcomponents = get_sidechain_rotation_mask(res, 0)['subcomponents']

                        assert [len(s) for s in cur_res_subcomponents] == [len(s) for s in flex_res_subcomponents], "Subcomponents of the residue in the PDB file do not match the subcomponents in the flexResiduesInfo object!"

                        flex_res_subcomponents = np.unique(np.concatenate(flex_res_subcomponents))
                        cur_res_subcomponents = np.unique(np.concatenate(cur_res_subcomponents))

                        assert len(flex_res_subcomponents) == len(cur_res_subcomponents), "Subcomponents of the residue in the PDB file do not match the subcomponents in the flexResiduesInfo object!"

                        atoms = list(res.get_atoms())
                        for flex_i, cur_i in zip(flex_res_subcomponents, cur_res_subcomponents):
                            if isinstance(sidechain_conformation[flex_i], torch.Tensor):
                                atoms[cur_i].set_coord(sidechain_conformation[flex_i].cpu().numpy())
                            else:
                                atoms[cur_i].set_coord(sidechain_conformation[flex_i])

                        done_ids.append(simplified_res_id)

                assert len(set(done_ids)) == len(done_ids), "Some residues were repeated in the PDB file!"
                done_ids = set(done_ids)
                pend_ids = set(pending_ids)
                if done_ids != pend_ids:
                    print("Warning! Not all flexible residues were present in the PDB file!")
                    print(f"Pending IDs: {pend_ids}")
                    print(f"Done IDs: {done_ids}")
                    print("Missing residues: ", pend_ids - done_ids)
                    print(f"Rec structure: {self.rec_structure}")
                    raise Exception("Missing residues in the PDB file!")

                if len(self.sidechain_conformations) > 1:
                    output_file.write("MODEL\n")

                # Write the current frame to the output PDB file
                writer.set_structure(self.rec_structure)
                writer.save(output_file, write_end=False)

                if len(self.sidechain_conformations) > 1:
                    output_file.write("ENDMDL\n")
