import os.path
from itertools import islice

import dgl
import networkx as nx
import numpy
import numpy as np
import pandas
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data
import ogb
from pubchempy import get_compounds,get_cids




BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

import torch

smile_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
              "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
              "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
              "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
              "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
              "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
              "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
              "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


def smile_one_hot(smiles):
    '''
    返回one-hot编码的特征
    :param smiles:
    :return:
    '''
    flag = torch.eye(64, dtype=torch.float)
    features = []
    padding = 128 - len(smiles)
    for char in smiles:
        feature = flag[smile_dict[char] - 1]
        features.append(feature)
    features = torch.stack(features)
    features = torch.nn.functional.pad(features, (0, 0, 0, padding))
    return features


def ont_hot_num(smiles):
    '''
    只返回数字
    :param smiles:
    :return:
    '''
    padding = 128 - len(smiles)
    features = []
    for char in smiles:
        feature = torch.tensor(smile_dict[char], dtype=torch.long)
        features.append(feature)
    features = torch.stack(features)
    features = torch.nn.functional.pad(features, (0, padding))
    return features


def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1:]


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def bond_features(bond):
    return np.array(
        one_of_k_encoding(bond.GetBondType(), BOND_LIST) + one_of_k_encoding(bond.GetBondDir(), BONDDIR_LIST)
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile2graph_dgl(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    # 节点特征
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
        # 边特征
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        feature = bond_features(bond)
        edge_feat.append(feature / sum(feature))
        edge_feat.append(feature / sum(feature))

    g = dgl.DGLGraph()
    g.add_nodes(c_size)
    g.add_edges(row, col)
    g.ndata['x'] = torch.tensor(numpy.array(features), dtype=torch.float)
    g = dgl.add_self_loop(g)
    g.ndata['pe'] = dgl.lap_pe(g, 8, padding=True)
    return g


def smile2graph_pyg(smile):
    mol = Chem.MolFromSmiles(smile)
    # 节点特征
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    # 边特征
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        feature = bond_features(bond)
        edge_feat.append(feature / sum(feature))
        edge_feat.append(feature / sum(feature))
        # edge_feat.append([1 if i else 0 for i in feature])
        # edge_feat.append([1 if i else 0 for i in feature])
    edge_index = torch.tensor(numpy.array([row, col]), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float)
    x = torch.tensor(numpy.array(features), dtype=torch.float)
    # node_num = x.shape[0]
    # # 添加自循环
    # edge_index = torch.cat((edge_index, torch.tensor([list(range(node_num)), list(range(node_num))])), dim=1)
    # edge_attr = torch.cat((edge_attr, torch.tensor([[0, 0, 0, 0, 0, 0, 0] for _ in range(node_num)])))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

if __name__ == "__main__":

    root = os.path.join('onei2')

    data = pandas.read_csv(os.path.join(root,'drugInfo.csv'))
    graphs = {}

    # 创建药物字典，获取其graph
    for i in range(data.shape[0]):
        name = data['names'][i]
        smiles = data['smiles'][i]
        g = smile2graph_pyg(smiles)

        mol = Chem.MolFromSmiles(smiles)
        fp = torch.tensor(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)), dtype=torch.float)
        # fp = torch.tensor(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=2048)), dtype=torch.float)


        one_hot = smile_one_hot(smiles)
        graphs[name] = {
            'smiles': smiles,
            'graph': g,
            'fp': fp,
            'one-hot': one_hot
        }

    pandas.DataFrame(graphs).to_pickle(os.path.join(root,'drug_pyg.pkl'))
    print('over')
