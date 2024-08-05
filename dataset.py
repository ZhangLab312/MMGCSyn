import os
import pickle
import random

import pandas
import torch
import torch.nn as nn
import numpy
import dgl


class Dataset(nn.Module):
    def __init__(self, combination, drug, cell,tissue):
        super(Dataset, self).__init__()
        # 协同组合数据
        self.combination = combination
        # 药物字典
        self.drug = drug
        # 细胞系字典
        self.cell = cell
        #细胞系类型
        self.tissue = tissue
        print('init over')

    def __len__(self):
        return self.combination.shape[0]

    def __getitem__(self, item):
        drug1=self.combination['drug1'][item]
        drug2=self.combination['drug2'][item]
        cell=self.combination['cell'][item]
        label = int(self.combination['label'][item])

        tissue = self.tissue[cell]
        try:
            graph1 = self.drug[drug1]['graph']
            fp1 = self.drug[drug1]['fp']
            hot1 = self.drug[drug1]['one-hot']

            graph2 = self.drug[drug2]['graph']
            fp2 = self.drug[drug2]['fp']
            hot2 = self.drug[drug2]['one-hot']

            cell = self.cell[cell]
        except:
            print(item)

        return graph1, graph2, cell, label, drug1, drug2, fp1, fp2, hot1, hot2,tissue
        # return graph1, graph2, cell, label, drug1, drug2, fp1, fp2, hot1, hot2
