# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cleans up a PDB file using pdbfixer in preparation for OpenMM simulations.
fix_pdb uses a third-party tool. We also support fixing some additional edge
cases like removing chains of length one (see clean_structure).
"""

# adapted from AlphaFold

import io

import pdbfixer
from rdkit import Chem
from rdkit.Chem import AllChem
from simtk.openmm import app
from simtk.openmm.app import element


def fix_pdb(complex_file_name, alterations_info):
    """Apply pdbfixer to the contents of a PDB file; return a PDB string result.
    1) Replaces nonstandard residues.
    2) Removes heterogens (non protein residues) including water.
    3) Adds missing residues and missing atoms within existing residues.
    4) Adds hydrogens assuming pH=7.0.
    5) KeepIds is currently true, so the fixer must keep the existing chain and
       residue identifiers. This will fail for some files in wider PDB that have
       invalid IDs.
    Args:
      pdbfile: Input PDB file handle.
      alterations_info: A dict that will store details of changes made.
    Returns:
      A PDB string representing the fixed structure.
    """
    fixer = pdbfixer.PDBFixer(filename=complex_file_name)
    fixer.findNonstandardResidues()
    alterations_info['nonstandard_residues'] = fixer.nonstandardResidues
    fixer.replaceNonstandardResidues()
    _remove_heterogens(fixer, alterations_info, keep_water=False)
    fixer.findMissingResidues()
    alterations_info['missing_residues'] = fixer.missingResidues
    fixer.findMissingAtoms()
    alterations_info['missing_heavy_atoms'] = fixer.missingAtoms
    alterations_info['missing_terminals'] = fixer.missingTerminals
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()
    out_handle = io.StringIO()
    app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle,
                          keepIds=True)
    return out_handle.getvalue()


def clean_structure(pdb_structure, alterations_info):
    """Applies additional fixes to an OpenMM structure, to handle edge cases.
    Args:
      pdb_structure: An OpenMM structure to modify and fix.
      alterations_info: A dict that will store details of changes made.
    """
    _replace_met_se(pdb_structure, alterations_info)
    _remove_chains_of_length_one(pdb_structure, alterations_info)


def _remove_heterogens(fixer, alterations_info, keep_water):
    """Removes the residues that Pdbfixer considers to be heterogens.
    Args:
      fixer: A Pdbfixer instance.
      alterations_info: A dict that will store details of changes made.
      keep_water: If True, water (HOH) is not considered to be a heterogen.
    """
    initial_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            initial_resnames.add(residue.name)
    fixer.removeHeterogens(keepWater=keep_water)
    final_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            final_resnames.add(residue.name)
    alterations_info['removed_heterogens'] = (
        initial_resnames.difference(final_resnames))


def _replace_met_se(pdb_structure, alterations_info):
    """Replace the Se in any MET residues that were not marked as modified."""
    modified_met_residues = []
    for res in pdb_structure.iter_residues():
        name = res.get_name_with_spaces().strip()
        if name == 'MET':
            s_atom = res.get_atom('SD')
            if s_atom.element_symbol == 'Se':
                s_atom.element_symbol = 'S'
                s_atom.element = element.get_by_symbol('S')
                modified_met_residues.append(s_atom.residue_number)
    alterations_info['Se_in_MET'] = modified_met_residues


def _remove_chains_of_length_one(pdb_structure, alterations_info):
    """Removes chains that correspond to a single amino acid.
    A single amino acid in a chain is both N and C terminus. There is no force
    template for this case.
    Args:
      pdb_structure: An OpenMM pdb_structure to modify and fix.
      alterations_info: A dict that will store details of changes made.
    """
    removed_chains = {}
    for model in pdb_structure.iter_models():
        valid_chains = [c for c in model.iter_chains() if len(c) > 1]
        invalid_chain_ids = [c.chain_id for c in model.iter_chains() if len(c) <= 1]
        model.chains = valid_chains
        for chain_id in invalid_chain_ids:
            model.chains_by_id.pop(chain_id)
        removed_chains[model.number] = invalid_chain_ids
    alterations_info['removed_chains'] = removed_chains


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            AllChem.ComputeGasteigerCharges(mol)

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol