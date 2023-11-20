import os
import glob
from pdbfixer import PDBFixer
from openmm.app import PDBFile

base_dir = 'posebusters_benchmark_set'
suffix = '_protein.pdb'
new_suffix = '_protein_fix.pdb'
for in_file in list(glob.glob(f'posebusters_benchmark_set/*/*{suffix}')):
    prefix = os.path.basename(in_file).removesuffix(suffix)
    out_file = os.path.join(os.path.dirname(in_file), prefix + new_suffix)
    print(f'{in_file} -> {out_file}')

    fixer = PDBFixer(filename=in_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    # fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    with open(out_file, 'w+') as out:
        PDBFile.writeFile(fixer.topology, fixer.positions, out)
