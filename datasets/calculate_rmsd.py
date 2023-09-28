from pathlib import Path
from torch_geometric.data import HeteroData
import torch
from datasets.process_mols import parse_receptor, get_lig_graph_with_matching
from datasets.sidechain_conformer_matching import RMSD
from datasets.pdbbind import read_mols
from datasets.pdbbind import PDBBind, PocketSelector
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

# Call this script from the root path with PYTHONPATH="." python filtering/calculate_rmsd.py

split_files = [
    Path("data", "splits", "timesplit_no_lig_overlap_train_aligned"),
    Path("data", "splits", "timesplit_no_lig_overlap_val_aligned"),
    Path("data", "splits", "timesplit_test_aligned"),
]

pdbbind_dir = Path("") #give path to pdbbind files

THREADS = 32

def process_complex(name):
    try:
        rec_model, rec_model_match = parse_receptor(name, pdbbind_dir, False, False, "protein_esmfold_aligned_tr_fix", True, "protein_processed_fix")

        # IMPORTANT: The indices between rec_model and rec_model_match are not a 1:1 mapping
        # So we sort the atoms by element name, such that they are equal

        for res in rec_model.get_residues():
            res.child_list.sort(key=lambda atom: PDBBind.order_atoms_in_residue(res, atom))

        for res in rec_model_match.get_residues():
            res.child_list.sort(key=lambda atom: PDBBind.order_atoms_in_residue(res, atom))

        assert len(list(rec_model.get_atoms())) == len(list(rec_model_match.get_atoms())), \
            "The length of the proteins does not match"

        # Check if we have 100% atom identity (hydrogens were ignored in loading already)
        assert [a.name for a in rec_model.get_atoms()] == [a.name for a in rec_model_match.get_atoms()], \
            "The proteins do not have 100% sequence identity (excluding hydrogens)"

        rec = np.array([a.coord for a in rec_model.get_atoms()])
        rec_match = np.array([a.coord for a in rec_model_match.get_atoms()])

        # Calculate RMSD
        global_rmsd = RMSD(list(range(len(rec))), rec, rec_match)

        complex_graph = HeteroData()
        complex_graph['name'] = name
        ligs = [lig[0] for lig in read_mols(pdbbind_dir, name, remove_hs=False)]
        if len(ligs) > 1:
            raise NotImplementedError("More than one ligand per complex is not supported yet.")

        lig = ligs[0]
        get_lig_graph_with_matching(lig, complex_graph, 15, 15, True, False, 1, remove_hs=True)

        pocket_center, pocket_radius = PDBBind._calculate_binding_pocket(torch.tensor(rec, dtype=complex_graph['ligand'].pos.dtype), complex_graph['ligand'].pos, 0, 5)
        pocket_radius_buffered = pocket_radius + 10

        selector = PocketSelector()
        selector.pocket = pocket_center.cpu().detach().numpy()
        selector.radius = pocket_radius_buffered.item()
        selector.all_atoms = True

        idxs = np.array([selector.accept_residue(a.parent) for a in rec_model.get_atoms()])
        pocket_rmsd = RMSD(idxs, rec, rec_match)

        return name, global_rmsd, pocket_rmsd
    except:
        print("Skipping complex", name)
        return name, float("nan"), float("nan")


if __name__ == '__main__':
    combined = None
    for split in split_files:
        # if split already exists
        try:
            loaded = np.load(split.name + "_rmsd.npz")
            rmsd = loaded["rmsd"]
            combined = rmsd if combined is None else np.concatenate((combined, rmsd))
            print("split", split, "already stored")
        except FileNotFoundError:
            with open(split) as f:
                names = [name.rstrip() for name in f]

                with Pool(THREADS) as p:
                    processed = list(tqdm(p.imap(process_complex, names), total=len(names)))

                # save processed to file
                np.savez(split.name + "_rmsd", rmsd=np.array(processed))

    print(combined.shape)
    global_rmsd = combined[:, 1].astype(np.float32)
    pocket_rmsd = combined[:, 2].astype(np.float32)

    import seaborn as sns
    import matplotlib.pyplot as plt

    print("< 1.5", (pocket_rmsd < 1.5).astype(float).mean())
    print("< 2", (pocket_rmsd < 2).astype(float).mean())
    print("< 3", (pocket_rmsd < 3).astype(float).mean())
    print("< 4", (pocket_rmsd < 4).astype(float).mean())
    print("< 4.5", (pocket_rmsd < 4.5).astype(float).mean())
    print("< 5", (pocket_rmsd < 5).astype(float).mean())

    # show global_rmsd and pocket_rmsd in one seaborn violin plot with labels
    sns.violinplot(data=[global_rmsd, pocket_rmsd], inner="quartile", cut=2)
    plt.ylim(0, 10)
    plt.show()

    sns.boxplot(data=[global_rmsd, pocket_rmsd], showfliers=False)
    plt.show()
