"""
Thank you to the authors of PoseBusters for providing this energy minimization code.
It has been taken from https://github.com/maabuu/posebusters_em/blob/f75057c89cc81f3d10bbaddb16e8aabb6c006642/energy_minimization.py and adapted to fit our inference script

Energy minimization of a ligand in a protein pocket as used in the PoseBusters paper.

This code is based on the OpenMM user guide:
http://docs.openmm.org/latest/userguide
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile

from openff.toolkit.topology import Molecule
from openmm import LangevinIntegrator, Platform, System, XmlSerializer, unit
from openmm.app import ForceField, Modeller, PDBFile, Simulation
from openmm.unit import kelvin, kilojoule, molar, mole, nanometer, picosecond
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from pdbfixer import PDBFixer
from rdkit.Chem.rdmolfiles import MolFromMolFile, MolToMolFile
from rdkit.Chem.rdmolops import AddHs, RemoveHs

FORCE_FIELDS_IMPLICIT = ["amber14-all.xml", "implicit/gbn2.xml"]
FORCE_FIELDS_EXPLICIT = ["amber14-all.xml", "amber14/tip3pfb.xml"]

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_molecule(mol_path: Path, **kwargs) -> Molecule:
    mols = Molecule.from_file(str(mol_path), file_format="sdf", **kwargs)
    molecule = mols[0] if isinstance(mols, list) else mols
    molecule.name = str(mol_path)
    return molecule


def prep_ligand(ligand_file: Path, temp_file: Path, allow_undefined_stereo: bool = True) -> Molecule:
    if not temp_file.exists():
        mol = MolFromMolFile(str(ligand_file), sanitize=True)
        mol = AddHs(mol, addCoords=True)

        temp_file.parent.mkdir(parents=True, exist_ok=True)
        MolToMolFile(mol, str(temp_file))

    return load_molecule(temp_file, allow_undefined_stereo=allow_undefined_stereo)


def prep_protein(protein_file: Path, temp_file: Path, add_solvent: bool = False) -> PDBFile:
    if not temp_file.exists():
        fixer = PDBFixer(str(protein_file))

        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(False)  # false also removes water
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        if add_solvent:
            fixer.addSolvent(fixer.topology.getUnitCellDimensions())

        temp_file.parent.mkdir(parents=True, exist_ok=True)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(temp_file, "w", encoding="utf-8"))

    return PDBFile(str(temp_file))


def get_modeller(
    protein_complex: PDBFile,
    ligand: Molecule,
) -> Modeller:
    modeller = Modeller(protein_complex.topology, protein_complex.positions)
    topology = ligand.to_topology().to_openmm()
    positions = ligand.conformers[0].magnitude * unit.angstrom
    modeller.add(topology, positions)
    return modeller


def deserialize(path: str) -> System:
    with open(path, "r", encoding="utf-8") as file:
        system = XmlSerializer.deserialize(file.read())
    return system


def serialize(system: System, path: str) -> str:
    with open(path, "w", encoding="utf-8") as file:
        file.write(XmlSerializer.serialize(system))
    return path


def get_fastest_platform() -> Platform:
    platforms = [Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())]
    speeds = [platform.getSpeed() for platform in platforms]
    platform = platforms[speeds.index(max(speeds))]
    return platform


def generate_system(
    modeller: Modeller,
    ligand: Molecule,
    num_particles_protein: int,
    name: str,
    force_fields: List[str],
) -> System:
    logger.info(f"Generating system for {name}")

    # setup forcefield
    forcefield = ForceField(*force_fields)
    smirnoff = SMIRNOFFTemplateGenerator(molecules=ligand)
    forcefield.registerTemplateGenerator(smirnoff.generator)

    # setup system
    system = forcefield.createSystem(modeller.topology)
    for i in range(num_particles_protein):
        system.setParticleMass(i, 0.0)

    return system


def setup_simulation(modeller: Modeller, system: System, platform: Platform) -> Simulation:
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picosecond)
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    return simulation


def save_with_rdkit(
    molecule: Molecule,
    file_path: Path,
    conformer_index: int = 0,
    name: str | None = None,
):
    # use rdkit because .to_file method does not allow picking conformation and only writes first one
    mol = molecule.to_rdkit()
    mol = RemoveHs(mol)
    if name is not None:
        mol.SetProp("_Name", name)
    MolToMolFile(mol, str(file_path), confId=conformer_index)


def optimize_ligand_in_pocket(
    protein_file: Path,
    ligand_file: Path,
    output_file: Optional[Path] = None,
    tolerance: float = 0.01,
    allow_undefined_stereo: bool = True,
    temp_base_dir: Path = Path("."),
    name: str | None = None,
    add_solvent: bool = False,
    platform_name: str = "fastest",
) -> Dict[str, Any]:
    name = protein_file.stem if name is None else name

    with tempfile.TemporaryDirectory(dir=temp_base_dir) as temp_dir:
        temp_dir = Path(temp_dir)
        protein_cache = temp_dir / f"{name}_prepped_protein.pdb"
        protein_complex = prep_protein(protein_file=protein_file, temp_file=protein_cache, add_solvent=False)

        ligand_cache = temp_dir / f"{name}_prepped_ligand.sdf"
        ligand = prep_ligand(ligand_file=ligand_file, temp_file=ligand_cache, allow_undefined_stereo=allow_undefined_stereo)

        modeller = get_modeller(protein_complex, ligand)

        if add_solvent:
            force_fields = FORCE_FIELDS_EXPLICIT
            dimensions = protein_complex.getTopology().getUnitCellDimensions()
            modeller.addSolvent(dimensions, model="tip3p", padding=1.0 * nanometer, ionicStrength=0.15 * molar)
        else:
            force_fields = FORCE_FIELDS_IMPLICIT

        num_particles_protein = len(protein_complex.positions)
        num_particles_ligand = len(ligand.conformers[0].magnitude)
        num_particles_total = len(modeller.getPositions())
        assert num_particles_ligand == num_particles_total - num_particles_protein

        # generate system
        system = generate_system(
            modeller=modeller,
            ligand=ligand,
            force_fields=force_fields,
            num_particles_protein=num_particles_protein,
            name=name,
        )

        platform = get_fastest_platform() if platform_name == "fastest" else Platform.getPlatformByName(platform_name)
        simulation = setup_simulation(modeller, system, platform)

        # save initial state
        state_before = simulation.context.getState(getEnergy=True, getPositions=False)
        energy_before = state_before.getPotentialEnergy()

        # minimize
        logger.info(f"Minimizing {name}")
        simulation.minimizeEnergy(tolerance=tolerance * kilojoule / mole / nanometer, maxIterations=0)

        # save final state
        logger.info(f"Saving {name}")
        state_after = simulation.context.getState(getEnergy=True, getPositions=True)
        energy_after = state_after.getPotentialEnergy()

        # save ligand
        ligand_positions = state_after.getPositions(asNumpy=True)[-num_particles_ligand:]
        ligand.add_conformer(ligand_positions)
        if output_file is not None:
            save_with_rdkit(ligand, output_file, conformer_index=1, name=name)

    return dict(energy_before=energy_before, energy_after=energy_after, ligand=ligand)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("protein_file", type=Path)
    parser.add_argument("ligand_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("-t", "--temp_dir", type=Path, default=Path("."))
    parser.add_argument("--add_solvent", action="store_true", default=False)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--platform", type=str, default="fastest")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    protein_file_path = Path(args.protein_file)
    ligand_file_path = Path(args.ligand_file)
    output_file_path = Path(args.output_file)
    temp_directory = Path(args.temp_dir)

    if not protein_file_path.exists():
        raise FileNotFoundError(f"File does not exist: {protein_file_path}")
    if not ligand_file_path.exists():
        raise FileNotFoundError(f"File does not exist: {ligand_file_path}")

    opt = optimize_ligand_in_pocket(
        protein_file=protein_file_path,
        ligand_file=ligand_file_path,
        output_file=output_file_path,
        temp_base_dir=temp_directory,
        name=args.name,
        platform_name=args.platform,
        add_solvent=args.add_solvent,
    )

    energy_before = opt["energy_before"].value_in_unit(kilojoule / mole)
    energy_after = opt["energy_after"].value_in_unit(kilojoule / mole)
    opt_mol = opt["ligand"]

    logger.info(
        f"{ligand_file_path}, "
        + f"E_start: {energy_before:.2f} kJ/mol, "
        + f"E_end: {energy_after:.2f} kJ/mol, "
        + f"ΔE: {energy_after - energy_before:.2f} kJ/mol"
    )