import copy

import numpy as np
from scipy.optimize import differential_evolution
import torch
from Bio.PDB.Model import Model
from scipy.spatial.transform import Rotation as R


def optimize_rotatable_bonds(rec, true_rec, subcomponents, subcomponentsMapping, edge_idx, residueNBondsMapping, ligand,
                             score="dist", seed=0,
                             popsize=15, maxiter=1000, mutation=(0.5, 1), recombination=0.7, verbose=False):
    if len(residueNBondsMapping) == 0:
        if verbose:
            print("No rotatable bonds found for conformer matching.")
        return rec, [], 0

    complete_rmsd_start = RMSD(list(range(len(list(rec.get_atoms())))), np.array([a.coord for a in rec.get_atoms()]),
                               np.array([a.coord for a in true_rec.get_atoms()]))

    rec_clone = copy.deepcopy(rec)
    filterSCHs = []

    start = 0
    improvements = 0
    optimal_rotations = []
    for bond_idx in residueNBondsMapping.cumsum(dim=0):
        rotatable_bonds = edge_idx[start:bond_idx]

        max_bound = [np.pi] * len(rotatable_bonds)
        min_bound = [-np.pi] * len(rotatable_bonds)
        bounds = list(zip(min_bound, max_bound))

        current_subcomponents = subcomponentsMapping[start:bond_idx]
        current_subcomponents = [subcomponents[a:b] for a, b in current_subcomponents]

        opt = OptimizeConformer(rec, true_rec, rotatable_bonds, current_subcomponents, ligand, seed=seed)
        if score == "dist":
            scoring = opt.score_conformation
        elif score == "nearest":
            scoring = opt.penalty_with_nearest_rmsd
        elif score == "exp":
            scoring = opt.penalty_with_weighted_exp_all_rmsd

        ## Optimize conformations
        result = differential_evolution(scoring, bounds,
                                        maxiter=maxiter, popsize=popsize,
                                        mutation=mutation, recombination=recombination, disp=False, seed=seed)

        optimal_rotations.append(np.array(result['x']))

        filterSCHs.extend(list(opt.modified_atoms))

        before = RMSD(opt.modified_atoms, np.array([a.coord for a in rec.get_atoms()]),
                      np.array([a.coord for a in true_rec.get_atoms()]))

        if before <= opt.last_rmsd:
            if verbose:
                print("No improvement possible for this sidechain. Not applying any rotations.")
        else:
            # Apply and store the optimal rotations
            new_pos = opt.apply_rotations(result['x'])

            for atom, p in zip(rec.get_atoms(), new_pos):
                atom.coord = p

            after = RMSD(opt.modified_atoms, np.array([a.coord for a in rec.get_atoms()]),
                         np.array([a.coord for a in true_rec.get_atoms()]))

            improvements += before - after

        start = bond_idx

    complete_rmsd_end = RMSD(list(range(len(list(rec.get_atoms())))), np.array([a.coord for a in rec.get_atoms()]),
                             np.array([a.coord for a in true_rec.get_atoms()]))

    assert complete_rmsd_end <= complete_rmsd_start, "RMSD should not increase after conformer matching."

    if verbose:
        print(f"Sidechain conformer matching reduced the overall rmsd by {complete_rmsd_start - complete_rmsd_end} (from {complete_rmsd_start} to {complete_rmsd_end})")
        print(f"Average RMSD delta after sidechain conformer matching: {improvements / len(residueNBondsMapping)}")

    return rec, \
           optimal_rotations, \
           RMSD(list(set(filterSCHs)), np.array([a.coord for a in rec_clone.get_atoms()]),
                np.array([a.coord for a in true_rec.get_atoms()])) - \
           RMSD(list(set(filterSCHs)), np.array([a.coord for a in rec.get_atoms()]),
                np.array([a.coord for a in true_rec.get_atoms()]))


def RMSD(atom_ids, atoms1, atoms2):
    if len(atom_ids) == 0:
        return 0.0
    # fix the alignment, so that the indices match
    try:
        coords_1 = atoms1[atom_ids]
        coords_2 = atoms2[atom_ids]
        return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / len(coords_1))
    except Exception as e:
        print("Cannot calculate RMSD. Maybe atoms1, and atoms2 do not mach?")
        print("atom_ids:", atom_ids)
        print("atoms1:", atoms1.shape)
        print("atoms2:", atoms2.shape)
        print(e)
        raise e


class OptimizeConformer:
    def __init__(self, rec: Model, true_rec: Model, rotatable_bonds: torch.tensor,
                 current_subcomponents: list[torch.tensor], ligand: torch.tensor, seed=None):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotatable_bonds = rotatable_bonds
        self.rec_pos = np.array([a.coord for a in rec.get_atoms()])
        self.true_rec_pos = np.array([a.coord for a in true_rec.get_atoms()])
        # Convert to numpy for applying rotations
        self.current_subcomponents = [c.cpu().numpy() for c in current_subcomponents]
        self.ligand = ligand.cpu().numpy()
        # These are the atoms which get a new position
        self.modified_atoms = np.unique(np.concatenate(self.current_subcomponents).ravel())
        # Store the last calculated RMSD so that we can check if the optimization improved the score
        self.last_rmsd = None
        mask = np.ones(self.true_rec_pos.shape[0], dtype=bool)
        mask[self.modified_atoms] = False
        self.mask = mask

    def closest_pos(self, sc_pos, rest_pos):
        return np.min(np.linalg.norm(sc_pos[None, :, :] - rest_pos[:, None, :], axis=-1), axis=0)

    def penalty_with_weighted_exp_all_rmsd(self, values):
        new_pos = self.apply_rotations(values)
        ligand_pos = np.row_stack((new_pos.copy(), self.ligand))

        # mask now includes all non-modified atoms and the ligand positions
        mask = np.append(self.mask.copy(), np.ones(self.ligand.shape[0], dtype=bool))

        distance = np.linalg.norm(ligand_pos[None, mask, :] - new_pos[self.modified_atoms, None, :], axis=-1)
        weight = np.exp(-distance)
        distance_sum = np.sum(np.multiply(distance, weight), axis=1)
        weight_sum = np.sum(weight, axis=1)
        weight_all = np.multiply(weight_sum * (1 / np.sum(weight_sum)), np.sqrt(distance_sum))
        self.last_rmsd = RMSD(self.modified_atoms, new_pos, self.true_rec_pos)

        return (self.last_rmsd / np.sqrt(np.sum(weight_all))) * np.sqrt(np.sum(distance_sum))

    def penalty_with_nearest_rmsd(self, values):
        new_pos = self.apply_rotations(values)
        new_atoms = new_pos[self.modified_atoms]

        closest_pair = self.closest_pos(new_atoms, new_pos[self.mask])

        np.row_stack((closest_pair, self.closest_pos(new_atoms, self.ligand)))

        closeness_rmsd = np.sqrt(np.mean(closest_pair))  # TODO use RMSD function?

        self.last_rmsd = RMSD(self.modified_atoms, new_pos, self.true_rec_pos)

        return self.last_rmsd - closeness_rmsd

    def score_conformation(self, values):
        # 1. Apply rotations to the current sidechain
        # Note that indices in current_subcomponent are based on the whole protein (so self.rec_atoms)
        # The same holds for self.rotatable_bonds
        new_pos = self.apply_rotations(values)

        # 2. Calculate the RMSD between the current sidechain and the true sidechain
        self.last_rmsd = RMSD(self.modified_atoms, new_pos, self.true_rec_pos)
        return self.last_rmsd

    def apply_rotations(self, values):
        # cannot use modify sidechain function for now, as this does only work for complex graphs

        pos = self.rec_pos.copy()

        for torsion_update, rot_bond, subcomponent in zip(values, self.rotatable_bonds, self.current_subcomponents):
            if torsion_update != 0:
                u, v = rot_bond
                mask_rotate = subcomponent
                # get atom positions of current subcomponent
                try:
                    rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards

                    rot_vec = rot_vec * torsion_update / np.linalg.norm(rot_vec)  # idx_edge!
                    rot_mat = R.from_rotvec(rot_vec).as_matrix()

                    pos[mask_rotate] = (pos[mask_rotate] - pos[v]) @ rot_mat.T + pos[v]
                except Exception as e:
                    print(f'Skipping sidechain update because of the error:')
                    print(e)

        return pos
