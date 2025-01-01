import fnmatch
import os
import random
from typing import List

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from typing import List, Tuple, Union

import dill
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def get_num_features(hetero_data: HeteroData) -> Tuple[int]:
    # Returns the number of node features and edge features in a HeteroData object.
    return hetero_data["data_i"].x.shape[1], hetero_data["data_i"].edge_attr.shape[1]


def get_mcs(pairs_list: List[HeteroData]) -> np.ndarray:
    MCS = []
    for hetero_data in pairs_list:
        MCS.append(np.array(hetero_data["mcs"]).astype(dtype=bool))
    return np.array(MCS)


def get_colors(pairs_list, explainer):
    colors = []
    for hetero_data in pairs_list:
        data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
        data_i.batch, data_j.batch = torch.zeros(
            data_i.x.size(0), dtype=torch.int64
        ), torch.zeros(data_j.x.size(0), dtype=torch.int64)
        color_pred_i, color_pred_j = (
            explainer.explain_graph(data_i),
            explainer.explain_graph(data_j),
        )
        colors.append([color_pred_i, color_pred_j])
    return colors


def make_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f"{path} created")
    else:
        print(f"{path} already exists")
        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_list_targets(n_targets, par_dir="data/"):
    """Read the .txt file containing the list of targets with n_targets protein targets"""
    list_targets = []
    list_file = os.path.join(par_dir, "list_targets_{}.txt".format(n_targets))
    if os.path.exists(list_file):
        with open(list_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                list_targets.append(line.strip())
    return list_targets


def get_list_targets(dir="data/"):
    """Get the list of targets with more than 50 pairs of ligands in their training set"""
    list_targets = []
    for target in os.listdir(dir):
        if len(target) == 8:
            count = len(fnmatch.filter(os.listdir(os.path.join(dir, target)), "*.*"))
            if count > 2:
                list_targets.append(target)
    list_file = "list_targets_{}.txt".format(len(list_targets))
    if os.path.exists(list_file) == False:
        with open(list_file, "w") as f:
            for item in list_targets:
                f.write("%s\n" % item)
    return list_targets


def get_common_nodes(mask_i, mask_j):
    """Get the common nodes between two graphs and their mapping to the same indexing."""

    idx_common_i = np.where(np.array(mask_i) == 0)[0]
    idx_common_j = np.where(np.array(mask_j) == 0)[0]
    assert len(idx_common_i) == len(idx_common_j)

    n_cmn = len(idx_common_i)

    map_i = dict({idx_common_i[i]: i for i in range(n_cmn)})
    map_j = dict({idx_common_j[i]: i for i in range(n_cmn)})
    return idx_common_i, idx_common_j, map_i, map_j


def get_substituents(data, idx_common, map):
    """Get the substituents and the active sites of the scaffold."""
    G = to_networkx(data)
    H = G.subgraph(idx_common)
    G.remove_edges_from(list(H.edges()))
    G.remove_nodes_from(list(nx.isolates(G)))
    active_sites = np.array(
        [map[i] for i in np.intersect1d(np.array(G.nodes()), idx_common)]
    ).flatten()
    substituents = [
        sorted(sub) for sub in sorted(nx.connected_components(G.to_undirected()))
    ]
    return substituents, active_sites


def get_positions(subs, idx_common, map):
    """Map active sites on the scaffold to the index of the substituent attached to it."""
    pos_sub = {i: -1 for i in range(len(idx_common))}
    for k, sub in enumerate(subs):
        active_sites = np.intersect1d(sub, idx_common)
        if len(active_sites) > 1:
            # print('Warning: multiple active sites for the same substituent.')
            for pos in active_sites:
                pos_sub[map[pos]] = k
        elif len(active_sites) == 1:
            pos_sub[map[active_sites[0]]] = k
    return pos_sub


def get_substituent_info(site, data, map):
    """Get the batch and activity of the substituent attached to the active site."""
    inverse_map = dict([(val, key) for key, val in map.items()])
    attach_node = inverse_map[site]
    batch = data.batch[attach_node]
    activity = data.y[batch]
    return batch, activity


def get_nodes(mask_i, mask_j):
    """Get the common and uncommon nodes between two graphs and their mapping to different indexing."""

    idx_common_i = np.where(np.array(mask_i) == 0)[0]
    idx_common_j = np.where(np.array(mask_j) == 0)[0]
    
    idx_uncommon_i = np.where(np.array(mask_i) == 1)[0]
    idx_uncommon_j = np.where(np.array(mask_j) == 1)[0]
    assert len(idx_common_i) == len(idx_common_j)

    n_cmn = len(idx_common_i)

    map_common_i = dict({idx_common_i[i]: i for i in range(n_cmn)})
    map_common_j = dict({idx_common_j[i]: i for i in range(n_cmn)})
    return idx_common_i, idx_common_j, idx_uncommon_i, idx_uncommon_j, map_common_i, map_common_j


# Function to map the mask value to a color
def value_to_color(value):
    color_map = plt.get_cmap('coolwarm')  # Blue to red colormap
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    return color_map(norm(value))
        
# Function to plot the molecular color graph to represent feature attribution mask      
def mol_color_graph(dataPath, maskPath, savePath):
    with open(dataPath, 'rb') as file:
        hetero_data = dill.load(file)
    with open(maskPath, 'rb') as file:
        mask_data = dill.load(file)

    smiles_list = []
    mask_list = []
    unique_smiles = {}  # Dictionary to track unique SMILES and their masks

    for i in range(len(mask_data)):
        smiles = hetero_data[i].data_i.smiles
        mask = mask_data[i][0]

        # Check if the SMILES string is already in the dictionary
        if smiles not in unique_smiles:
            unique_smiles[smiles] = mask  # Add to the dictionary
            smiles_list.append(smiles)  # Add to the list
            mask_list.append(mask)  # Add to the list
        
    for i in range(len(smiles_list)):
        # Read each molecule smiles and mask
        mol = Chem.MolFromSmiles(smiles_list[i])
        mask_vector = mask_list[i]

        # Create a dictionary of colors for each atom
        atom_colors = {i: value_to_color(mask_vector[i]) for i in range(len(mask_vector))}
        highlight_radii = {i: 0.5 for i in range(mol.GetNumAtoms())} 

        # Drawing options
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
        opts = drawer.drawOptions()

        for i in range(mol.GetNumAtoms()):
            opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()

        # Draw the molecule with the colored atoms
        drawer.DrawMolecule(mol, highlightAtoms=range(mol.GetNumAtoms()), highlightAtomColors=atom_colors,  highlightAtomRadii=highlight_radii)
        drawer.FinishDrawing()

        # Convert the drawing to an SVG
        svg = drawer.GetDrawingText()
        
        # To save the SVG to a file, you can use:
        with open(savePath + "test_" + str(i) +"_molecule_w_pred_color.svg", "w") as f:
            f.write(svg)