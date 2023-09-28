import networkx as nx
import numpy as np
import torch, copy
import re 
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from utils.geometry import rigid_transform_Kabsch_independent_torch

"""
    Preprocessing and computation for torsional updates to conformers
"""


def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    # get all edges
    edges = pyg_data['ligand', 'ligand'].edge_index.T.numpy()
    # itereate over edges, skip every second, because graph is still directed here 
    # e.g. [[0,3] , [3,0]]  -> skip second edge and make graph undirected 
    for i in range(0, edges.shape[0], 2):
        # assure that consecutive edges in list belong to same bond 
        assert edges[i, 0] == edges[i+1, 1]

        # transform to undirected graph and delete current edge 
        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        # graph still connected ? 
        if not nx.is_connected(G2):
            # if not, get largest connected component 
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            # more than 1 vertex in component ? 
            if len(l) > 1:
                # first vertex from current edge in largest connected component ? 
                # -> rotate all vertices of the subgraph which does not contain the vertex of 
                # edge i from index 0 
                if edges[i, 0] in l:
                    # if yes, rotate the other part of the molecule
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    # if no, rotate around 
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        # graph still connected, so no rotatable bond here 
        to_rotate.append([])
        to_rotate.append([])
    # True for all edges that connect 2 otherwise unconnected structures 
    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    # initialize rotation mask with false for all edges in mask_edges 
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    # iterate over all edges in G (directed graph with duplicate edges )
    for i in range(len(G.edges())):
        # is it an edge that connectes 2 otherwise unconnected sub-structures?
        if mask_edges[i]:
            # write all vertices that should be rotated when rotating around edge i 
            # into mask_rotate
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate

def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()

    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        return modify_conformer_torsion_angles(data.pos,
                                               data.edge_index.T[data.edge_mask],
                                               data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer_torsion_angles(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new


def get_dihedrals(data_list):
    edge_index, edge_mask = data_list[0]['ligand', 'ligand'].edge_index, data_list[0]['ligand'].edge_mask
    edge_list = [[] for _ in range(torch.max(edge_index) + 1)]

    for p in edge_index.T:
        edge_list[p[0]].append(p[1])

    rot_bonds = [(p[0], p[1]) for i, p in enumerate(edge_index.T) if edge_mask[i]]

    dihedral = []
    for a, b in rot_bonds:
        c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
        d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
        dihedral.append((c.item(), a.item(), b.item(), d.item()))
    # dihedral_numpy = np.asarray(dihedral)
    # print(dihedral_numpy.shape)
    dihedral = torch.tensor(dihedral)
    return dihedral


def bdot(a, b):
    return torch.sum(a*b, dim=-1, keepdim=True)


def get_torsion_angles(dihedral, batch_pos):
    c, a, b, d = dihedral[:, 0], dihedral[:, 1], dihedral[:, 2], dihedral[:, 3]
    c_project_ab = batch_pos[:,a] + bdot(batch_pos[:,c] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    d_project_ab = batch_pos[:,a] + bdot(batch_pos[:,d] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    dshifted = batch_pos[:,d] - d_project_ab + c_project_ab
    cos = bdot(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab) / (
                torch.norm(dshifted - c_project_ab, dim=-1, keepdim=True) * torch.norm(batch_pos[:,c] - c_project_ab, dim=-1,
                                                                                       keepdim=True))
    cos = torch.clamp(cos, -1 + 1e-5, 1 - 1e-5)
    angle = torch.acos(cos)
    #assert not torch.any(torch.isnan(angle)), (angle, cos)
    sign = torch.sign(bdot(torch.cross(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab), batch_pos[:,b] - batch_pos[:,a]))
    torsion_angles = (angle * sign).squeeze(-1)

    #assert torch.all(torsion_angles > (- np.pi - 0.01)) and torch.all(torsion_angles < (np.pi + 0.01))
    return torsion_angles


def get_torsion_angles_svgd(dihedral, batch_pos):
    tau = get_torsion_angles(dihedral, batch_pos)
    tau_diff = tau.unsqueeze(1) - tau.unsqueeze(0)
    tau_diff = torch.fmod(tau_diff + 3 * np.pi, 2 * np.pi) - np.pi
    assert torch.all(tau_diff < np.pi + 0.1) and torch.all(tau_diff > -np.pi - 0.1), tau_diff
    tau_matrix = torch.sum(tau_diff ** 2, dim=-1, keepdim=True)

    return tau_matrix, tau_diff


def get_rigid_svgd(batch_pos):
    n = len(batch_pos)
    tr_diff, rot_diff = torch.zeros(n, n, 3), torch.zeros(n, n, 3)

    for i in range(n-1):
        for j in range(i+1, n):
            t, R_vec = rigid_transform_Kabsch_independent_torch(batch_pos[i].T, batch_pos[j].T)
            tr_diff[i][j], tr_diff[j][i] = t.squeeze(1), -t.squeeze(1)
            rot_diff[i][j], rot_diff[j][i] = R_vec, -R_vec

    tr_matrix = torch.sum(tr_diff ** 2, dim=-1, keepdim=True)
    rot_matrix = torch.sum(rot_diff ** 2, dim=-1, keepdim=True)
    return tr_matrix, rot_matrix, tr_diff, rot_diff


def get_sidechain_rotation_mask(residue,flexResIndexFullAtoms, true_res = None) :
    #compute rotatable bonds and rotation mask for each rotatable bond 

    # filter out non-heavy atoms
    nodes = list(filter(filter_side_chain_atoms,[atom.name for atom in residue.child_list]))
    # get mask that maps index of nodes to residue index 
    heavy_atoms_mask = [i for i, atom in enumerate(residue.get_atoms()) if atom.name in nodes]
    if true_res is not None:

        rows=[]
        coords=[]

    # build graph 
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    add_edges(G)

    mask_rotate = []
    # traverse the side chain graph from the CA downwards
    # to find rotatable bonds in correct order
    for i,edge in enumerate(nx.bfs_tree(G, "CA").edges()):
        G2 = G.to_undirected()
        G2.remove_edge(*edge)
        if not nx.is_connected(G2):
            # possible rotatable bond
            
            # find subcomponent that has second vertex of current edge in it 
            # -> this is the one that will get rotated
            connectedComponents = list(nx.connected_components(G2))
            
            # find subcomponent that contains second vertex of edge
            # which is the one to be rotated
            for k,component in enumerate(connectedComponents):
                if edge[1] in component:
                    idx = k
                    break
            if len(connectedComponents[idx]) > 1:
                G2Nodes = list(G2.nodes)
                # rotate this subcomponent
                rotComponent = []
                for j,vertex in enumerate(connectedComponents[idx]):
                    # map graph idx to residue atom index and the residue atom index to global 
                    # atom index by the flexResIndexFullAtoms offset 
                    complexGraphIdx = heavy_atoms_mask[G2Nodes.index(vertex)] + flexResIndexFullAtoms
                    rotComponent.append(complexGraphIdx)
                    if true_res is not None:
                        rows.append(complexGraphIdx)
                        coords.append(true_res[nodes[G2Nodes.index(vertex)]].get_coord())
                # (subcomponentToRotate,EdgeToRotateAround)
                mask_rotate.append((rotComponent,[heavy_atoms_mask[G2Nodes.index(edge[0])] + flexResIndexFullAtoms,heavy_atoms_mask[G2Nodes.index(edge[1])] + flexResIndexFullAtoms]))
    
    return_dict = {"subcomponents":[m[0] for m in mask_rotate],
                   "edge_idx":[m[1] for m in mask_rotate]}
     
    if true_res is not None:
        return_dict["rows"] = rows
        return_dict["coords"] = coords

    return return_dict

def filter_side_chain_atoms(atom):
    # ignores the O-H, OXT and NH2 group and drops the H atoms
    # re returns no match if we should keep the atom for the
    # side chain torsion graph
    return re.search("^(OXT)$|^C$|^O$|^N$|^H|^H$.|^H.$[1-9]",atom) is None

def add_edges(G):
    # add edges according to logic -> connect all heavy atoms in correct order
    orderDict = {"A":"B","B":"G","G":"D","D":"E","E":"Z","Z":"H","H":""}
    atoms = list(G.nodes)
    for i in range(len(G.nodes)-1):
        for j in range(i+1,len(G.nodes),1):
            cur,_next = atoms[i], atoms[j]
            # handle special 5- ring connections for his,trp
            if (cur,_next) == ("CE1","NE2") or (cur,_next) == ("NE1","CE2") or (cur,_next) == ("CD2","CE3") or (cur,_next) == ("CZ3","CH2"):
                    G.add_edge(cur,_next)
            # if both have length 3 we have to match number and identification char
            # i.e. mactch CD2 -> CE2 , but not CD1 -> CE2
            if len(cur) == len(_next) == 3:
                if orderDict[cur[1]] == _next[1] and cur[2]==_next[2]:
                    # match, e.g. CD1 -> CE1
                    G.add_edge(cur,_next)
            else:
                if orderDict[cur[1]] == _next[1]:
                    # match, e.g. CD -> CE1
                    G.add_edge(cur,_next)


def modify_sidechain_torsion_angle(pos, edge_index, mask_subcomponent, subcomponents, torsion_update, as_numpy=False):
    # modify single sidechain torsion angle 
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    assert len(edge_index) == 2 # make sure that its just a single bond 
    if torsion_update != 0:
        u, v = edge_index[0], edge_index[1]
        mask_rotate = subcomponents[mask_subcomponent[0]:mask_subcomponent[1]]
        if type(mask_rotate) != np.ndarray: mask_rotate = mask_rotate.cpu().numpy()

        try:
            rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        except Exception as e:
            print(e)
            if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
            return pos
            
        rot_vec = rot_vec * torsion_update / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()
        try:
            pos[mask_rotate] = (pos[mask_rotate] - pos[v]) @ rot_mat.T + pos[v]
        except Exception as e:
            print(f'Skipping sidechain update because of the error:')
            print(e)
            print("pos size: ", np.size(pos))
            print("edge_index: ", edge_index)
            print("mask_subcomponent: ", mask_subcomponent)
            print("subcomponents: ", subcomponents)
            print("torsion_update: ", torsion_update)
            print("mask_rotate: ", mask_rotate)
            print("v: ", v)


    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos 
    
