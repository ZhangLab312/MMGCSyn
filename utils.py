import os

import dgl
import numpy
import requests

import torch_geometric as pyg
from torch_geometric.data.batch import Batch
from tqdm import tqdm
import pubchempy as pcp


class download_smiles:

    def __init__(self):
        print('init over')

    def pubchempy(self,drugname_list):
        smiles_dict = {}
        not_find = []
        for drug_name in tqdm(drugname_list):
            try:
                compounds = pcp.get_compounds(drug_name, 'name')
                smiles_dict[drug_name] = compounds[0].canonical_smiles
            except:
                not_find.append(drug_name)
        return smiles_dict, not_find




def collate_dgl(batch):
    graph1, graph2, cell, label, drug1, drug2, fp1, fp2,hot1,hot2 = zip(*batch)

    # 将 DGLGraph 对象转换为元组或其他可用的数据类型

    graph1_batch = dgl.batch(graph1)
    graph2_batch = dgl.batch(graph2)
    fp1 = torch.stack(fp1)
    fp2 = torch.stack(fp2)
    hot1 = torch.stack(hot1)
    hot2 = torch.stack(hot2)

    x = {
        'drug1': {
            'name': drug1,
            'graph': graph1_batch,
            'fp': fp1,
            'one-hot': hot1
        },
        'drug2': {
            'name': drug2,
            'graph': graph2_batch,
            'fp': fp2,
            'one-hot': hot2
        },
        'cell': torch.stack(cell),
    }
    y = torch.tensor(label)
    return x, y


def collate_pyg(batch):
    graph1, graph2, cell, label, drug1, drug2, fp1, fp2, hot1, hot2,tissue= zip(*batch)
    # graph1, graph2, cell, label, drug1, drug2, fp1, fp2, hot1, hot2 = zip(*batch)

    # 将 DGLGraph 对象转换为元组或其他可用的数据类型
    graph1_batch = Batch.from_data_list(graph1)
    graph2_batch = Batch.from_data_list(graph2)
    fp1 = torch.stack(fp1)
    fp2 = torch.stack(fp2)
    hot1 = torch.stack(hot1)
    hot2 = torch.stack(hot2)

    x = {
        'drug1': {
            'name': drug1,
            'graph': graph1_batch,
            'fp': fp1,
            'one-hot': hot1
        },
        'drug2': {
            'name': drug2,
            'graph': graph2_batch,
            'fp': fp2,
            'one-hot': hot2
        },
        'cell': torch.stack(cell),
    }
    y = torch.tensor(label)
    tissue = torch.tensor(tissue)

    return x, y,tissue


import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size=128, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        try:
            representations = torch.cat([zjs, zis], dim=0)

            similarity_matrix = self.similarity_function(representations, representations)

            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

            logits = torch.cat((positives, negatives), dim=1)
            logits /= self.temperature

            labels = torch.zeros(2 * self.batch_size).to(self.device).long()
            loss = self.criterion(logits, labels)
        except:
            print(5)

        return loss / (2 * self.batch_size)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(path)
        self.trace_func = trace_func
        self.metric = None

    def __call__(self, val_loss, model, metric):

        self.metric = metric
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f}){self.metric}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves models when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}){self.metric}.  Saving models ...')
            # self.trace_func(
            #     f'{self.metric}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

