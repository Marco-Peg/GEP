import torch
import torch.nn as nn
from IGEP.constants.epi_constants import *
from torch_geometric.nn import GCNConv, GATConv


class EpiEPMP(torch.nn.Module):
    """ EpiEPMP model
    """
    def __init__(self, num_cdr_feats=NUM_CDR_FEATURES, num_ag_feats=NUM_AG_FEATURES, inner_dim=None):
        """ Initialize EpiEPMP model
        :param num_cdr_feats: number antibody cdr features
        :param num_ag_feats:    number antigen features
        :param inner_dim:    dimension of inner layers
        """
        super(EpiEPMP, self).__init__()
        self.relu = nn.ReLU()
        if inner_dim is None:
            assert num_cdr_feats == num_ag_feats
            inner_dim = num_cdr_feats

        self.gcn = GCNConv(num_cdr_feats, inner_dim)
        self.bn1 = nn.BatchNorm1d(inner_dim)
        self.aggcn = GCNConv(num_ag_feats, inner_dim)
        self.agbn1 = nn.BatchNorm1d(inner_dim)

        self.gat = GATConv(inner_dim, inner_dim, dropout=0.5)
        self.gat2 = GATConv(inner_dim, inner_dim, dropout=0.5)

        self.agbn2 = nn.BatchNorm1d(inner_dim * 2)
        self.dropout1 = nn.Dropout(0.5)
        self.agfc = nn.Linear(inner_dim * 2, 1, 1)
        self.bn2 = nn.BatchNorm1d(inner_dim * 2)
        self.dropout2 = nn.Dropout(0.15)
        self.fc = nn.Linear(inner_dim * 2, 1, 1)

        # self.all_bn2 = nn.BatchNorm1d(inner_dim)

    def forward(self, x_ab, x_ag, edge_x_ab, edge_x_ag, edge_index_d):
        """ Forward pass

        :param x_ab:      antibody features
        :param x_ag:    antigen features
        :param edge_x_ab:   antibody edges
        :param edge_x_ag:   antigen edges
        :param edge_index_d:  distance edges
        :return:    prediction antibody and antigen

        """
        x_ab = self.gcn(x_ab, edge_x_ab)
        x_ab = self.bn1(x_ab)
        x_ab = self.relu(x_ab)
        x_ag = self.aggcn(x_ag, edge_x_ag)
        x_ag = self.agbn1(x_ag)
        x_ag = self.relu(x_ag)
        x_ab = self.dropout1(x_ab)
        x_ag = self.dropout1(x_ag)
        x = torch.cat((x_ab, x_ag), dim=0)
        x = self.gat(x, edge_index_d)
        x = self.relu(x)
        x = self.gat2(x, edge_index_d)
        n_cdr = x_ab.size()[0]
        n_ag = x_ag.size()[0]
        x_1 = x[0:n_cdr, :]
        x_2 = x[n_cdr:n_cdr + n_ag, :]

        x_ag = torch.cat((x_2, x_ag), dim=1)  # Residual connection for ag
        x_ag = self.agbn2(x_ag)
        x_ag = self.relu(x_ag)
        x_ag = self.dropout2(x_ag)
        x_ag = self.agfc(x_ag)

        x_ab = torch.cat((x_1, x_ab), dim=1)  # Residual connection for ab
        x_ab = self.bn2(x_ab)
        x_ab = self.relu(x_ab)
        x_ab = self.dropout2(x_ab)
        x_ab = self.fc(x_ab)
        return x_ab, x_ag
