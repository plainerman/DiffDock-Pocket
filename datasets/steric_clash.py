import numpy as np
from rdkit.Chem import GetPeriodicTable

PERIODIC_TABLE = GetPeriodicTable()

# https://pubs.acs.org/doi/10.1021/jp8111556 Table 12
# and https://en.wikipedia.org/wiki/Van_der_Waals_radius
# (which takes it from https://reference.wolfram.com/language/ref/ElementData.html)
VAN_DER_WAALS_RADII = {
    'H': 1.10,
    'He': 1.40,
    'Li': 1.81,
    'Be': 1.53,
    'B': 1.92,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'Ne': 1.54,
    'Na': 2.27,
    'Mg': 1.73,
    'Al': 1.84,
    'Si': 2.10,
    'P': 1.80,
    'S': 1.80,
    'Cl': 1.75,
    'Ar': 1.88,
    'K': 2.75,
    'Ca': 2.31,
    'Sc': 2.11,  # Wikipedia
    'Ni': 1.63,  # Wikipedia
    'Cu': 1.40,  # Wikipedia
    'Zn': 1.39,  # Wikipedia
    'Ga': 1.87,
    'Ge': 2.11,
    'As': 1.85,
    'Se': 1.90,
    'Br': 1.83,
    'Kr': 2.02,
    'Rb': 3.03,
    'Sr': 2.49,
    'Pd': 1.63,  # Wikipedia
    'Ag': 1.72,  # Wikipedia
    'Cd': 1.58,  # Wikipedia
    'In': 1.93,
    'Sn': 2.17,
    'Sb': 2.06,
    'Te': 2.06,
    'I': 1.98,
    'Xe': 2.16,
    'Cs': 3.43,
    'Ba': 2.68,
    'Pt': 1.75,  # Wikipedia
    'Au': 1.66,  # Wikipedia
    'Hg': 1.55,  # Wikipedia
    'Tl': 1.96,
    'Pb': 2.02,
    'Bi': 2.07,
    'Po': 1.97,
    'At': 2.02,
    'Rn': 2.20,
    'Fr': 3.48,
    'Ra': 2.83,
    'U': 1.86,  # Wikipedia
    'default': 2.0,
}

# The distance the radius on both sides must overlap
# So distance(atom1, atom2) < r1 + r2 - 2 * OVERLAP_DISTANCE counts as a steric clash
OVERLAP_DISTANCE = 0.4

def get_atom_radius(elements):
    # Metals are not in VAN_DER_WAALS_RADII, so we take 2A as an estimate from https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf
    return np.array([VAN_DER_WAALS_RADII[element] if element in VAN_DER_WAALS_RADII else 2.0 for element in elements])


def get_ligand_elements(complex_graph):
    lig_elements = ["default"] * complex_graph["ligand"].x.size(dim=0)
    for i,x in enumerate(complex_graph["ligand"].x[:, 0]+1):
        try:
            lig_elements[i]=PERIODIC_TABLE.GetElementSymbol(x.item())
        except:
            continue
    return lig_elements


def get_rec_elements(complex_graph):
    rec_elements = ["default"] * complex_graph["atom"].x.size(dim=0)
    for i,x in enumerate(complex_graph["atom"].x[:, 1]+1):
        try:
            rec_elements[i]=PERIODIC_TABLE.GetElementSymbol(x.item())
        except:
            continue
    return rec_elements


def get_steric_clash_atom_pairs(mol_1, mol_2, elements_1, elements_2, filter1=None, filter2=None):
    #can be used for rec-lig, rec-rec, lig-lig / predicted and original versions
    #mol_1 mol_2 are the 3d coordinates of the atoms, already "tiled" to N instances - or solve better way, needed because of ligand output for N>1 samples in inference
    #the tiling: np.tile(mol, (N,1,1))

    assert len(mol_1) == len(mol_2)

    N = len(mol_1)

    mol_1 = np.array(mol_1)
    mol_2 = np.array(mol_2)
    elements_1 = np.array(elements_1)
    elements_2 = np.array(elements_2)

    assert mol_1.shape[1] == len(elements_1)
    assert mol_2.shape[1] == len(elements_2)

    if filter1 is not None:
        if len(filter1) == 0:
            return np.array([False])
        mol_1 = mol_1[:, filter1, :]
        elements_1 = elements_1[filter1]

        assert mol_1.shape[1] == len(elements_1)

    if filter2 is not None:
        if len(filter2) == 0:
            return np.array([False])
        mol_2 = mol_2[:, filter2, :]
        elements_2 = elements_2[filter2]

        assert mol_2.shape[1] == len(elements_2)

    atom_radii_1 = np.tile(get_atom_radius(elements_1), (N, 1))
    atom_radii_2 = np.tile(get_atom_radius(elements_2), (N, 1))
    cross_distances = np.linalg.norm(mol_1[:, :, None, :] - mol_2[:, None, :, :], axis=-1)
    ramanchandran_radii = atom_radii_1[:, :, None] + atom_radii_2[:, None, :] - 2 * OVERLAP_DISTANCE

    return cross_distances < ramanchandran_radii


def get_steric_clash_fraction(mol_1, mol_2, elements_1, elements_2, N=1):
    steric_clash_atom_pairs = np.sum(get_steric_clash_atom_pairs(mol_1, mol_2, elements_1, elements_2, N), axis=(1, 2))
    return np.sum(steric_clash_atom_pairs > 0) / N


def get_steric_clash_per_flexble_sidechain_atom(complex_graph, rec_rest = True):
    rec_sc_rec_rest_steric_clashes = []
    sidechain_atoms = np.array([], dtype=int)
    start = 0

    all_sidechains = np.unique(complex_graph['flexResidues'].subcomponents)

    for bond_idx in complex_graph['flexResidues'].residueNBondsMapping.cumsum(dim=0):
        current_subcomponents = complex_graph['flexResidues'].subcomponentsMapping[start:bond_idx]
        current_subcomponents = [complex_graph['flexResidues'].subcomponents[a:b] for a, b in current_subcomponents]

        start = bond_idx  # for next iteration

        cur_sidechain_atoms = np.unique([item.item() for sublist in current_subcomponents for item in sublist])
        if len(cur_sidechain_atoms) == 0:
            continue

        sidechain_atoms = np.append(sidechain_atoms, cur_sidechain_atoms)

        if rec_rest == False:
            not_sidechain_atoms = np.zeros(complex_graph["atom"].x.shape[0], dtype = bool)
            not_sidechain_atoms[all_sidechains] = True
        else:
            not_sidechain_atoms = np.ones(complex_graph["atom"].x.shape[0], dtype=bool)
        not_sidechain_atoms[sidechain_atoms] = False

        rec_sc_rec_rest_steric_clashes.append(
            get_steric_clash_atom_pairs(complex_graph["atom"].pos[None, :],
                                        complex_graph["atom"].pos[None, :],
                                        get_rec_elements(complex_graph),
                                        get_rec_elements(complex_graph),
                                        filter1=cur_sidechain_atoms, filter2=not_sidechain_atoms).sum())

    if len(sidechain_atoms) == 0:
        return 0

    return np.sum(rec_sc_rec_rest_steric_clashes) / len(sidechain_atoms)