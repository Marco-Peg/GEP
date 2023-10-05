import torch
import torch.nn as nn
from IGEP.constants.epi_constants import *
# import models.egnn_layer as eg
from egnn_pytorch import EGNN
from torch_geometric.nn import GATConv, GCNConv



class EpiEPMP(torch.nn.Module):
  """ EpiEPMP model

    Args:
        num_cdr_feats (int): Number of features in CDR
        num_ag_feats (int): Number of features in antigen
        inner_dim (int): Dimension of inner layers
  """

  def __init__(self, num_cdr_feats=NUM_CDR_FEATURES, num_ag_feats=NUM_AG_FEATURES, inner_dim=None, num_egnn=1,
               soft_edges=True, norm_coors=True, dropout=0.5, update_coors=True, use_adj=False):
    """ Initialize EpiEPMP model
    :param num_cdr_feats: number antibody cdr features
    :param num_ag_feats:    number antigen features
    :param inner_dim:    dimension of inner layers
    """

    super(EpiEPMP, self).__init__()
    self.relu = nn.ReLU()
    self.use_adj = use_adj

    if inner_dim is None:
      assert num_cdr_feats == num_ag_feats
      inner_dim = num_cdr_feats
      self.resize = False
    else:
      self.gcn = GCNConv(num_cdr_feats, inner_dim)
      self.abbn0 = nn.BatchNorm1d(inner_dim)
      self.aggcn = GCNConv(num_ag_feats, inner_dim)
      self.agbn0 = nn.BatchNorm1d(inner_dim)
      self.resize = True
    self.inner_dim = inner_dim
    self.num_egnn = num_egnn
    self.egnn = [EGNN(dim=inner_dim, norm_coors=norm_coors, soft_edges=soft_edges, dropout=dropout,
                      update_coors=update_coors, only_sparse_neighbors=use_adj) for i in range(num_egnn)]
    self.bn1 = [nn.BatchNorm1d(inner_dim) for i in range(num_egnn)]
    self.aegnn = [EGNN(dim=inner_dim, norm_coors=norm_coors, soft_edges=soft_edges, dropout=dropout,
                       update_coors=update_coors, only_sparse_neighbors=use_adj) for i in range(num_egnn)]
    self.agbn1 = [nn.BatchNorm1d(inner_dim) for i in range(num_egnn)]
    for i in range(num_egnn):
      self.add_module(f"ab_egnn{i + 1}", self.egnn[i])
      self.add_module(f"ag_egnn{i + 1}", self.aegnn[i])
      self.add_module(f"ab_bn{i + 1}", self.bn1[i])
      self.add_module(f"ag_bn{i + 1}", self.agbn1[i])

    self.inter = GATConv(inner_dim, inner_dim, dropout=dropout)
    self.inter2 = GATConv(inner_dim, inner_dim, dropout=dropout)
    self.dropout1 = nn.Dropout(dropout)

    self.agbn2 = nn.BatchNorm1d(inner_dim * 2)
    self.agfc = nn.Linear(inner_dim * 2, 1, 1)

    self.bn2 = nn.BatchNorm1d(inner_dim * 2)
    self.dropout2 = nn.Dropout(0.15)
    self.fc = nn.Linear(inner_dim * 2, 1, 1)
    # self.all_bn2 = nn.BatchNorm1d(28)

  def forward(self, x_ab, x_ag, edge_x_ab, edge_x_ag, edge_index_d, coord_ab, coord_ag, ):
    """ Forward pass of EpiEPMP model
    :param x_ab:    antibody features
    :param x_ag:    antigen features
    :param edge_x_ab:   antibody edge index
    :param edge_x_ag:   antigen edge index
    :param edge_index_d:    distance edge index
    :param coord_ab:    antibody coordinates
    :param coord_ag:    antigen coordinates
    :return:    predicted binding affinity antigen-antibody
    """

    if self.resize:
      x_ab = self.gcn(x_ab, edge_x_ab)
      x_ab = self.abbn0(x_ab)
      x_ab = self.relu(x_ab)
      x_ab = self.dropout1(x_ab)
      x_ag = self.aggcn(x_ag, edge_x_ag)
      x_ag = self.agbn0(x_ag)
      x_ag = self.relu(x_ag)
      x_ag = self.dropout1(x_ag)

    ## antibody egnn
    coord_ab = coord_ab[None, :]
    if self.use_adj:
      adj_mat_ab = torch.zeros((x_ab.shape[0], x_ab.shape[0]), dtype=torch.bool, device=x_ag.device)
      adj_mat_ab[edge_x_ab[0], edge_x_ab[1]] = True
    else:
      adj_mat_ab = None
    for i in range(self.num_egnn):
      x_ab = x_ab[None, :]
      x_ab, coords_ab = self.egnn[i](x_ab, coord_ab, adj_mat=adj_mat_ab)
      x_ab = torch.squeeze(x_ab, dim=0)
      x_ab = self.bn1[i](x_ab)
      x_ab = self.relu(x_ab)
      x_ab = self.dropout1(x_ab)
    ## antigene egnn
    coord_ag = coord_ag[None, :]
    if self.use_adj:
      adj_mat_ag = torch.zeros((x_ag.shape[0], x_ag.shape[0]), dtype=torch.bool, device=x_ag.device)
      adj_mat_ag[edge_x_ag[0], edge_x_ag[1]] = True
    else:
      adj_mat_ag = None
    for i in range(self.num_egnn):
      x_ag = x_ag[None, :]
      x_ag, coords_ag = self.aegnn[i](x_ag, coord_ag, adj_mat=adj_mat_ag)
      x_ag = torch.squeeze(x_ag, dim=0)
      x_ag = self.agbn1[i](x_ag)
      x_ag = self.relu(x_ag)
      x_ag = self.dropout1(x_ag)

    ## concat + attention
    x = torch.cat((x_ab, x_ag), dim=0)
    x = self.inter(x, edge_index_d)
    x = self.relu(x)
    x = self.inter2(x, edge_index_d)

    n_cdr = x_ab.size()[0]
    n_ag = x_ag.size()[0]
    x_1 = x[0:n_cdr, :]
    x_2 = x[n_cdr:n_cdr + n_ag, :]
    ## antigene linear
    x_ag = torch.cat((x_2, x_ag), dim=1)  # Residual connection for ag
    x_ag = self.agbn2(x_ag)
    x_ag = self.relu(x_ag)
    x_ag = self.dropout2(x_ag)
    x_ag = self.agfc(x_ag)
    ## antibody linear
    x_ab = torch.cat((x_1, x_ab), dim=1)  # Residual connection for ab
    x_ab = self.bn2(x_ab)
    x_ab = self.relu(x_ab)
    x_ab = self.dropout2(x_ab)
    x_ab = self.fc(x_ab)
    return x_ab, x_ag


class EGNN_shared(EpiEPMP):
  """ EpiEPMP model with shared EGNN layers
  """

  def __init__(self, num_cdr_feats=NUM_CDR_FEATURES, num_ag_feats=NUM_AG_FEATURES, inner_dim=None, num_egnn=1,
               soft_edges=True, norm_coors=True, dropout=0.5, update_coors=True, use_adj=False):
    """ Initialize EpiEPMP model
    :param num_cdr_feats:   number of antibody features
    :param num_ag_feats:    number of antigen features
    :param inner_dim:   inner dimension of GCN layers
    :param num_egnn:    number of EGNN layers
    :param soft_edges:  whether to use soft edges
    :param norm_coors:  whether to normalize coordinates
    :param dropout: dropout rate
    :param update_coors:    whether to update coordinates
    :param use_adj: whether to use adjacency matrix
    """
    super(EGNN_shared, self).__init__(num_cdr_feats, num_ag_feats, inner_dim, num_egnn,
                                      soft_edges, norm_coors, dropout, update_coors, use_adj)

    del self.aegnn
    del self.agbn1

  def forward(self, x_ab, x_ag, edge_x_ab, edge_x_ag, edge_index_d, coord_ab, coord_ag, ):

    """ Forward pass
    :param x_ab:    antibody features
    :param x_ag:    antigen features
    :param edge_x_ab:   antibody edge indices
    :param edge_x_ag:   antigen edge indices
    :param edge_index_d:    distance edge indices
    :param coord_ab:    antibody coordinates
    :param coord_ag:    antigen coordinates
    :return:    antibody and antigen logits
    """

    if self.resize:
      x_ab = self.gcn(x_ab, edge_x_ab)
      x_ab = self.abbn0(x_ab)
      x_ab = self.relu(x_ab)
      x_ab = self.dropout1(x_ab)
      x_ag = self.aggcn(x_ag, edge_x_ag)
      x_ag = self.agbn0(x_ag)
      x_ag = self.relu(x_ag)
      x_ag = self.dropout1(x_ag)

    ## antibody egnn
    coord_ab = coord_ab[None, :]
    if self.use_adj:
      adj_mat_ab = torch.zeros((x_ab.shape[0], x_ab.shape[0]), dtype=torch.bool, device=x_ag.device)
      adj_mat_ab[edge_x_ab[0], edge_x_ab[1]] = True
    else:
      adj_mat_ab = None
    for i in range(self.num_egnn):
      x_ab = x_ab[None, :]
      x_ab, coords_ab = self.egnn[i](x_ab, coord_ab, adj_mat=adj_mat_ab)
      x_ab = torch.squeeze(x_ab, dim=0)
      x_ab = self.bn1[i](x_ab)
      x_ab = self.relu(x_ab)
      x_ab = self.dropout1(x_ab)

    coord_ag = coord_ag[None, :]
    if self.use_adj:
      adj_mat_ag = torch.zeros((x_ag.shape[0], x_ag.shape[0]), dtype=torch.bool, device=x_ag.device)
      adj_mat_ag[edge_x_ag[0], edge_x_ag[1]] = True
    else:
      adj_mat_ag = None
    for i in range(self.num_egnn):
      x_ag = x_ag[None, :]
      x_ag, coords_ag = self.egnn[i](x_ag, coord_ag, adj_mat=adj_mat_ag)
      x_ag = torch.squeeze(x_ag, dim=0)
      x_ag = self.bn1[i](x_ag)
      x_ag = self.relu(x_ag)
      x_ag = self.dropout1(x_ag)

    ## concat + attention
    x = torch.cat((x_ab, x_ag), dim=0)
    x = self.inter(x, edge_index_d)
    x = self.relu(x)
    x = self.inter2(x, edge_index_d)

    n_cdr = x_ab.size()[0]
    n_ag = x_ag.size()[0]
    x_1 = x[0:n_cdr, :]
    x_2 = x[n_cdr:n_cdr + n_ag, :]
    ## antigene linear
    x_ag = torch.cat((x_2, x_ag), dim=1)  # Residual connection for ag
    x_ag = self.agbn2(x_ag)
    x_ag = self.relu(x_ag)
    x_ag = self.dropout2(x_ag)
    x_ag = self.agfc(x_ag)
    ## antibody linear
    x_ab = torch.cat((x_1, x_ab), dim=1)  # Residual connection for ab
    x_ab = self.bn2(x_ab)
    x_ab = self.relu(x_ab)
    x_ab = self.dropout2(x_ab)
    x_ab = self.fc(x_ab)
    return x_ab, x_ag


class fullyEGNN_shared(EGNN_shared):
  """ EpiEPMP model with fully connected layers
  """
  def __init__(self, num_cdr_feats=NUM_CDR_FEATURES, num_ag_feats=NUM_AG_FEATURES, inner_dim=None,
               soft_edges=True, norm_coors=True, dropout=0.5, update_coors=True):
    """ Constructor
    :param num_cdr_feats:   number of antibody features
    :param num_ag_feats:    number of antigen features
    :param inner_dim:   inner dimension of the model
    :param soft_edges:  whether to use soft edges
    :param norm_coors:  whether to normalize coordinates
    :param dropout: dropout rate
    :param update_coors:    whether to update coordinates
    """

    super(EGNN_shared, self).__init__(num_cdr_feats, num_ag_feats, inner_dim)

    self.inter = EGNN(dim=self.inner_dim, norm_coors=norm_coors, soft_edges=soft_edges, dropout=dropout,
                      only_sparse_neighbors=True, update_coors=update_coors)
    self.inter2 = EGNN(dim=self.inner_dim, norm_coors=norm_coors, soft_edges=soft_edges, dropout=dropout,
                       only_sparse_neighbors=True, update_coors=update_coors)

  def forward(self, x_ab, x_ag, edge_x_ab, edge_x_ag, edge_index_d, coord_ab, coord_ag, ):
    """ Forward pass of the model

    :param x_ab:    antibody features
    :param x_ag:    antigen features
    :param edge_x_ab:   antibody edge indices
    :param edge_x_ag:   antigen edge indices
    :param edge_index_d:  distance edge indices
    :param coord_ab:  antibody coordinates
    :param coord_ag:  antigen coordinates
    :return:  antibody and antigen logits
    """
    if self.resize:
      x_ab = self.gcn(x_ab, edge_x_ab)
      x_ab = self.bn1(x_ab)
      x_ab = self.relu(x_ab)
      x_ab = self.dropout1(x_ab)
      x_ag = self.aggcn(x_ag, edge_x_ag)
      x_ag = self.agbn1(x_ag)
      x_ag = self.relu(x_ag)
      x_ag = self.dropout1(x_ag)

    ## antibody egnn
    x_ab = x_ab[None, :]
    coord_ab = coord_ab[None, :]
    x_ab, coords_ab = self.egnn(x_ab, coord_ab)

    # coord_ab = torch.squeeze(coord_ab, dim=0)
    # x_ab = self.gcn(x_ab, edge_x_ab)
    x_ab = self.bn1(x_ab.transpose(1, 2)).transpose(1, 2)
    x_ab = self.relu(x_ab)
    ## antigene egnn
    x_ag = x_ag[None, :]
    coord_ag = coord_ag[None, :]
    x_ag, coord_ag = self.egnn(x_ag, coord_ag)
    # x_ag = torch.squeeze(x_ag, dim=0)
    # coord_ag = torch.squeeze(coord_ag, dim=0)
    x_ag = self.bn1(x_ag.transpose(1, 2)).transpose(1, 2)
    x_ag = self.relu(x_ag)
    ## concat + attention
    x = torch.cat((x_ab, x_ag), dim=1)
    coord = torch.cat((coord_ab, coord_ag), dim=1)
    adj_mat_abag = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.bool, device=x_ag.device)
    adj_mat_abag[edge_index_d[0], edge_index_d[1]] = True
    x, coord = self.inter(x, coord, adj_mat=adj_mat_abag)
    x = self.relu(x)
    x, coord = self.inter2(x, coord, adj_mat=adj_mat_abag)

    x = torch.squeeze(x, dim=0)
    x_ab = torch.squeeze(x_ab, dim=0)
    x_ag = torch.squeeze(x_ag, dim=0)

    n_cdr = x_ab.size()[0]
    n_ag = x_ag.size()[0]
    x_1 = x[0:n_cdr, :]
    x_2 = x[n_cdr:n_cdr + n_ag, :]
    ## antigene linear
    x_ag = torch.cat((x_2, x_ag), dim=1)  # Residual connection for ag
    x_ag = self.agbn2(x_ag)
    x_ag = self.relu(x_ag)
    x_ag = self.dropout2(x_ag)
    x_ag = self.agfc(x_ag)
    ## antibody linear
    x_ab = torch.cat((x_1, x_ab), dim=1)  # Residual connection for ab
    x_ab = self.bn2(x_ab)
    x_ab = self.relu(x_ab)
    x_ab = self.dropout2(x_ab)
    x_ab = self.fc(x_ab)
    return x_ab, x_ag


class EpiEPMPwrap():
  """ Wrapper for the EpiEPMP model """

  def __init__(self):
    self.model = EpiEPMP()
