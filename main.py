import os
import pickle
import random
import time
import warnings

import numpy
import numpy as np
import pandas
import torch
import torch.nn.functional as F
import yaml
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from dataset import Dataset
from model.model import GTNet
from utils import collate_pyg, EarlyStopping

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)
random.seed(42)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    lenth = len(train_loader)
    start = time.time()

    total_loss = []
    for i, data in enumerate(train_loader):
        x, y, tissue = data
        x['drug1']['graph'] = x['drug1']['graph'].to(device)
        x['drug1']['fp'] = x['drug1']['fp'].to(device)
        x['drug1']['one-hot'] = x['drug1']['one-hot'].to(device)
        x['drug2']['one-hot'] = x['drug2']['one-hot'].to(device)
        x['drug2']['graph'] = x['drug2']['graph'].to(device)
        x['drug2']['fp'] = x['drug2']['fp'].to(device)
        x['cell'] = x['cell'].to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output, CL_loss = model(x)
        loss = criterion(output, y) + CL_loss
        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    end = time.time()
    return (f'epoch:{epoch + 1},loss:{sum(total_loss) / len(total_loss):.4f},time:{end - start:.4f}')


def valid(model, device, train_loader, criterion):
    model.eval()
    lenth = len(train_loader)
    total_loss = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    for i, data in enumerate(train_loader):
        x, y, tissue = data
        x['drug1']['graph'] = x['drug1']['graph'].to(device)
        x['drug1']['fp'] = x['drug1']['fp'].to(device)
        x['drug1']['one-hot'] = x['drug1']['one-hot'].to(device)
        x['drug2']['one-hot'] = x['drug2']['one-hot'].to(device)
        x['drug2']['graph'] = x['drug2']['graph'].to(device)
        x['drug2']['fp'] = x['drug2']['fp'].to(device)
        x['cell'] = x['cell'].to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output, CL_loss = model(x)

        ys = F.softmax(output, 1).to('cpu').data.numpy()

        loss = criterion(output, y)
        total_loss.append(loss.item())
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
        total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
        total_labels = torch.cat((total_labels, y.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten(), sum(
        total_loss) / len(total_loss)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    root = os.path.join('data', config['dataset'])
    combination = pandas.read_csv(os.path.join(root, config['datafile'] + '.csv'))
    drug = pandas.read_pickle(os.path.join(root, 'drug_pyg.pkl'))
    flag = pandas.read_csv('data/cell.csv').iloc[1:, :]
    cell = {}
    for i in range(flag.shape[0]):
        cell[flag.iloc[i, 0].split('_')[0]] = torch.tensor(numpy.array(flag.iloc[i, 1:].astype(float)),
                                                           dtype=torch.float)

    tissue = {}
    flag = pandas.read_csv('data/drugcomb/cell_tissue.csv')
    for i in range(flag.shape[0]):
        tissue[flag.iloc[i, 0]] = flag.iloc[i, -1]

    dataset = Dataset(combination, drug, cell, tissue)

    # 跨数据集
    if config['valid']:
        combination = pandas.read_csv(os.path.join('data', config['valid_dataset'], 'raw.csv'))
        drug = pandas.read_pickle(os.path.join('data', config['valid_dataset'], 'drug_pyg.pkl'))
        dataset_valid = Dataset(combination, drug, cell, tissue)
        loader_valid = DataLoader(dataset_valid, batch_size=config['train_batch'], shuffle=True, drop_last=True,
                                  collate_fn=collate_pyg, num_workers=0)

    for i in range(5):
        if not os.path.exists(
                f'data/{config["dataset"]}/{config["modal"]}/{config["split"]}/{config["split"]}_train_{i}.csv'):
            print(
                f'data/{config["dataset"]}/{config["modal"]}/{config["split"]}/{config["split"]}_train_{i}.csv  not found')
            break

        train_index = list(
            pandas.read_csv(
                f'data/{config["dataset"]}/{config["modal"]}/{config["split"]}/{config["split"]}_train_{i}.csv')['id'])
        test_index = list(
            pandas.read_csv(
                f'data/{config["dataset"]}/{config["modal"]}/{config["split"]}/{config["split"]}_test_{i}.csv')['id'])

        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
        loader_train = DataLoader(train_dataset, batch_size=config['test_batch'], shuffle=True, drop_last=False,
                                  collate_fn=collate_pyg, num_workers=0)
        loader_test = DataLoader(test_dataset, batch_size=config['train_batch'], shuffle=True, drop_last=False,
                                 collate_fn=collate_pyg, num_workers=0)

        model = GTNet().to(device)
        # 初始化默认方法
        for m in model.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        print(f"model para is {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        filename = os.path.basename(__file__).split('.')[0]

        dir = f'result/{filename}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_path = os.path.join(config['save_path'], filename,
                                 f"{config['dataset']}_{config['modal']}_{config['split']}{i}.pt")
        stopping = EarlyStopping(
            patience=config['patient'],
            path=save_path,
            verbose=True
        )

        best_auc = 0
        if i == 0:
            print('Training on {} samples...'.format(len(loader_train.dataset)))
        for epoch in range(config['max_epoch']):
            string = train(model, device, loader_train, criterion, optimizer, epoch)
            # state = torch.load(save_path)
            # model.load_state_dict(state)
            true_label, pre_score, pre_label, valid_loss = valid(model, device, loader_test, criterion)

            AUC = roc_auc_score(true_label, pre_score)
            precision, recall, threshold = metrics.precision_recall_curve(true_label, pre_score)
            PR_AUC = metrics.auc(recall, precision)
            BACC = balanced_accuracy_score(true_label, pre_label)
            tn, fp, fn, tp = confusion_matrix(true_label, pre_label).ravel()
            TPR = tp / (tp + fn)
            PREC = precision_score(true_label, pre_label)
            ACC = accuracy_score(true_label, pre_label)
            KAPPA = cohen_kappa_score(true_label, pre_label)
            recall = recall_score(true_label, pre_label)

            stopping(valid_loss, model, [epoch, AUC, PR_AUC, ACC,PREC, KAPPA, recall])

            if stopping.early_stop:
                break

            if config['valid']:
                break
    print(5)
