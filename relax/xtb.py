import warnings

import rdkit, os, subprocess
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem, Geometry
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import rdmolfiles
from rdkit.Geometry import Point3D
import torch

biopython_parser = PDBParser()
def parsePDB(pdbid, pdbbind_dir):
    rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein_processed.pdb')
    if not os.path.exists(rec_path):
        rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein_obabel_reduce.pdb')
        if not os.path.exists(rec_path):
            rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein.pdb')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]
    return rec

my_dir = f"/tmp/{os.getpid()}"
if not os.path.isdir(my_dir):
    if not os.path.isdir("../utils/tmp"):
        os.makedirs("../utils/tmp", exist_ok=True)
    os.makedirs(my_dir, exist_ok=True)


def coords_to_xyz(coords, out):
    s = str(len(coords)) + '\n\n'
    for elem, loc in coords:
        s += elem + '\t'
        s += '\t'.join(map(str, loc)) + '\n'
    if out: open(out, 'w').write(s)
    return s


def xtb_energy(coords, path_xtb='xtb', gfnff=False):
    path = f"/tmp/{os.getpid()}.xyz"
    # rdmolfiles.MolToXYZFile(mol, path)
    coords_to_xyz(coords, path)
    cmd = [path_xtb, path, '--iterations', str(1000)]
    if gfnff: cmd += ["--gfnff"]
    n_tries = 3
    result = {}
    for i in range(n_tries):
        try:
            out = subprocess.check_output(cmd, stderr=open('/dev/null', 'w'), cwd=my_dir)
            break
        except subprocess.CalledProcessError as e:
            if i == n_tries - 1:
                print('xtb_energy did not converge')
                return result

    runtime = out.split(b'\n')[-8].split()
    result['runtime'] = float(runtime[-2]) + 60 * float(runtime[-4]) + 3600 * float(runtime[-6]) + 86400 * float(
        runtime[-8])

    energy = [line for line in out.split(b'\n') if b'TOTAL ENERGY' in line]
    result['energy'] = 627.509 * float(energy[0].split()[3])
    print('energy', result['energy'])
    return result


def xtb_optimize(lig_coords, prot_coords, radius=10, level='normal', fix=True, mode='fix', gfnff=False, path_xtb='xtb'):
    in_path = f'{my_dir}/xtb.xyz'
    out_path = f'{my_dir}/xtbopt.xyz'
    mask_path = f'{my_dir}/xtb.inp'
    if os.path.exists(out_path): os.remove(out_path)

    # compute mask
    lc = np.array([b for a, b in lig_coords]).reshape(-1, 1, 3)
    pc = np.array([b for a, b in prot_coords]).reshape(1, -1, 3)
    dist = np.square(lc - pc).sum(-1) ** 0.5
    mask = dist.min(0) < radius
    ##########

    coords = lig_coords + list(np.array(prot_coords, dtype=object)[mask])
    # rdmolfiles.MolToXYZFile(mol, in_path)

    coords_to_xyz(coords, in_path)
    with open(mask_path, 'w') as f:
        f.write(f'${mode}\natoms:{len(lig_coords) + 1}-{len(coords)}\n$end')

    cmd = [path_xtb, in_path, "--opt", level]
    if fix: cmd += ['--input', mask_path]
    if gfnff: cmd += ["--gfnff"]
    print(cmd)
    out = subprocess.check_output(cmd, stderr=open('/dev/null', 'w'), cwd=my_dir)

    runtime = out.split(b'\n')[-12].split()
    runtime = float(runtime[-2]) + 60 * float(runtime[-4]) + 3600 * float(runtime[-6]) + 86400 * float(runtime[-8])
    out = open(out_path).read().split('\n')[2:-1]
    coords = []
    for line in out:
        _, x, y, z = line.split()
        coords.append([float(x), float(y), float(z)])

    ### update ligand coords ####
    for i in range(len(lig_coords)):
        lig_coords[i][1][:] = coords[i]
        ###########

    return runtime


def optimize_complex(complex_graph):
    lig_mol = Chem.RemoveHs(complex_graph.mol[0])
    print("Num conformers", lig_mol.GetNumConformers())

    lig_pos = complex_graph['ligand'].pos
    conf = lig_mol.GetConformer()
    print(lig_pos.shape, lig_mol.GetNumAtoms())
    for i in range(lig_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(lig_pos[i, 0].item(), lig_pos[i, 1].item(), lig_pos[i, 2].item()))

    lig_mol = Chem.AddHs(lig_mol, addCoords=True)
    coords = lig_mol.GetConformers()[0].GetPositions()
    types = [atom.GetSymbol() for atom in lig_mol.GetAtoms()]
    lig_coords = list(zip(types, coords))

    name = complex_graph.name[0]
    print(complex_graph)
    rec_structure = parsePDB(name, 'data/PDBBind_processed/')
    rec_center = complex_graph.original_center.numpy()[0]
    atom_types, atom_coords = [], []

    for atom in rec_structure.get_atoms():
        atom_types.append(atom.element)
        atom_coords.append(np.asarray(atom.get_coord()) - rec_center)

    rec_coords = list(zip(atom_types, atom_coords))

    print(len(lig_coords), lig_coords)
    res = xtb_optimize(lig_coords, rec_coords, radius=5, fix=True, mode='fix', gfnff=True)
    print(lig_coords)
    print(res)
    
    r = xtb_energy(lig_coords+rec_coords, path_xtb='xtb', gfnff=True)
    
    new_lig_pos = torch.from_numpy(np.asarray([l[1] for l in lig_coords]))
    print(complex_graph['ligand'].pos.shape, new_lig_pos.shape)
    complex_graph['ligand'].pos = new_lig_pos[:complex_graph['ligand'].pos.shape[0]]

    return


if __name__ == "__main__":
    lkp = {6: 'C', 1: 'H'}

    mol = Chem.MolFromSmiles('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC')
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    coords = mol.GetConformers()[0].GetPositions()
    types = [lkp[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    prot_coords = list(zip(types, coords))

    mol = Chem.MolFromSmiles('C')
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    coords = mol.GetConformers()[0].GetPositions()
    types = [lkp[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    lig_coords = list(zip(types, coords))

    print(len(lig_coords), lig_coords)
    #print(len(prot_coords), prot_coords)
    res = xtb_optimize(lig_coords, prot_coords, radius=10, fix=True, mode='fix', gfnff=True)
    print(lig_coords)
    print(res)
