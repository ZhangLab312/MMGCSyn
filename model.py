
import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
import torch.nn.functional as F

from model.DCN import DeformableConv2d
from model.SCConv import ScConv

class graph_extractor(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(graph_extractor, self).__init__()
        # molecular graph
        self.ds_conv1 = GATConv(in_channels, in_channels)
        # self.ds_bn1 = nn.BatchNorm1d(num_features_xd)
        self.ds_conv2 = GATConv(in_channels, in_channels * 2)
        # self.ds_bn2 = nn.BatchNorm1d(num_features_xd * 2)
        self.ds_conv3 = GATConv(in_channels * 2, in_channels * 4)
        # self.ds_bn3 = nn.BatchNorm1d(num_features_xd * 4)
        self.ds_fc1 = torch.nn.Linear(in_channels * 4, 1024)
        self.ds_bn4 = nn.BatchNorm1d(1024)
        self.ds_fc2 = torch.nn.Linear(1024, out_channels)
        self.ds_bn5 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = self.ds_conv1(x, edge_index)
        x = self.relu(x)
        x = self.ds_conv2(x, edge_index)
        x = self.relu(x)
        x = self.ds_conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.ds_fc1(x)
        x = self.ds_bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ds_fc2(x)
        x = self.ds_bn5(x)
        out = self.dropout(x)

        return out

class fp_extractor(nn.Module):
    def __init__(self,n_filters:int=8,output_dim:int=128):
        super(fp_extractor, self).__init__()
        # drug finger
        self.df_conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=9)
        self.df_bn1 = nn.BatchNorm1d(n_filters)
        self.df_pool1 = nn.MaxPool1d(3)
        self.df_conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=9)
        self.df_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.df_pool2 = nn.MaxPool1d(3)
        self.df_conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=9)
        self.df_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.df_pool3 = nn.MaxPool1d(3)
        self.df_fc1 = nn.Linear(32 * 72, 512)  # 3712
        self.df_bn4 = nn.BatchNorm1d(512)
        self.df_fc2 = nn.Linear(512, output_dim)  # 2944
        self.df_bn5 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.df_conv1(x)
        x = self.df_bn1(x)
        x = self.relu(x)
        x = self.df_pool1(x)
        x = self.df_conv2(x)
        x = self.df_bn2(x)
        x = self.relu(x)
        x = self.df_pool2(x)
        x = self.df_conv3(x)
        x = self.df_bn3(x)
        x = self.relu(x)
        x = self.df_pool3(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.df_fc1(x)
        x = self.df_bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.df_fc2(x)
        out = self.df_bn5(x)
        return out

class fp_dcn(nn.Module):
    def __init__(self,in_channels:int=1,output_dim:int=128,n_filters:int=8):
        super(fp_dcn, self).__init__()
        self.df_conv1 = DeformableConv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=(9,1),padding=0)
        self.df_bn1 = nn.BatchNorm2d(n_filters)
        self.df_pool1 = nn.MaxPool2d((3, 1))
        self.df_conv2 = DeformableConv2d(in_channels=n_filters, out_channels=n_filters * 2,kernel_size=(9,1),padding=0)
        self.df_bn2 = nn.BatchNorm2d(n_filters * 2)
        self.df_pool2 = nn.MaxPool2d((3, 1))
        self.df_conv3 = DeformableConv2d(in_channels=n_filters * 2, out_channels=n_filters * 4,kernel_size=(9,1),padding=0)
        self.df_bn3 = nn.BatchNorm2d(n_filters * 4)
        self.df_pool3 = nn.MaxPool2d((3, 1))
        self.df_fc1 = nn.Linear(32 * 72, 512)
        self.df_bn4 = nn.BatchNorm1d(512)
        self.df_fc2 = nn.Linear(512, output_dim)
        self.df_bn5 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.scconv = ScConv(
            op_channel=32,
            group_num=8,
            gate_treshold=0.5
        )

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)
        B, C, H, W = x.size()

        x = self.df_conv1(x)
        x = self.df_bn1(x)
        x = self.relu(x)
        x = self.df_pool1(x)
        x = self.df_conv2(x)
        x = self.df_bn2(x)
        x = self.relu(x)
        x = self.df_pool2(x)
        x = self.df_conv3(x)
        x = self.df_bn3(x)
        x = self.relu(x)
        x = self.df_pool3(x)
        x = self.scconv(x)
        x = x.view(B, -1)
        x = self.df_fc1(x)
        x = self.df_bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.df_fc2(x)
        out = self.df_bn5(x)
        return out
class cell_extractor(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        dim = [954, 2048, 512, hidden]

        self.project = nn.Sequential(
            nn.Linear(dim[0], dim[1]),
            nn.ReLU(),
            nn.BatchNorm1d(dim[1]),
            nn.Dropout(0.2),
            nn.Linear(dim[1], dim[2]),
            nn.ReLU(),
            nn.BatchNorm1d(dim[2]),
            nn.Dropout(0.2),
            nn.Linear(dim[2], dim[3]),
            nn.ReLU(),
            nn.BatchNorm1d(dim[3]),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        out = self.project(x)
        return out


class GTNet(torch.nn.Module):
    def __init__(self, drug_in=78, hidden=128):
        super(GTNet, self).__init__()

        self.graph_extractor = graph_extractor(
            in_channels=78,
            hidden_channels=128,
            out_channels=128
        )
        self.fp_extractor = fp_dcn()


        self.cell = cell_extractor(hidden)


        self.trans = nn.Transformer(
            d_model=hidden,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=200,
            norm_first=True,
            batch_first=True,
            dropout = 0.2
        )

        dim = [hidden * 8, 512, 128, 2]
        self.predict = nn.Sequential(
            nn.Linear(dim[0], dim[1]),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim[1], dim[2]),
            nn.BatchNorm1d(dim[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim[2], dim[3]),
        )

    def forward(self, x):
        batch = x['drug1']['fp'].shape[0]

        gA = x['drug1']['graph']
        fpA = x['drug1']['fp']
        # nameA = x['drug1']['name']
        # one_hotA = x['drug1']['one-hot']
        # one_hotA = self.embedding(one_hotA)

        gB = x['drug2']['graph']
        fpB = x['drug2']['fp']
        # nameB = x['drug2']['name']
        # one_hotB = x['drug2']['one-hot']
        # one_hotB = self.embedding(one_hotB)

        cell = x['cell']

        f_graphA = self.graph_extractor(gA)
        f_graphB = self.graph_extractor(gB)

        f_fpA = self.fp_extractor(fpA)
        f_fpB = self.fp_extractor(fpB)

        f_cell = self.cell(cell)

        f_cat = torch.cat((f_graphA,f_cell,f_graphB,f_cell,f_fpA,f_cell,f_fpB,f_cell), dim=1).reshape(batch,8,-1)
        # f_cat = torch.cat((f_fpA,f_cell,f_fpB,f_cell), dim=1).reshape(batch,-1)

        f_fusion = self.trans(f_cat,f_cat).reshape(batch,-1)

        out = self.predict(f_fusion)

        return out, 0