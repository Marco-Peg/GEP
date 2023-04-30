from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import OGEP.Models.diffusion_net as diffNet


class DiffNet_AbAgPredictor(nn.Module):
    def __init__(self, in_dim, diffN_out, diffN_widths=(64, 128, 1024), diffN_dropout=True,
                 seg_widths=(1024, 512, 256, 180, 64), pdrop=0.3,
                 with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral',
                 device=torch.device('cpu'), shared_diff=True):
        super(DiffNet_AbAgPredictor, self).__init__()
        # Basic parameters
        self.in_dim = in_dim
        self.diffN_out = diffN_out
        self.device = device
        self.shared_diff = shared_diff

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError(
            "invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # First and last affine layers
        self.first_lin = nn.Linear(in_dim, diffN_widths[0], device=device)

        self.diffN_blocks = []
        self.diffN_lins = []
        for i_block in range(len(diffN_widths)):
            if i_block > 0 and diffN_widths[i_block] != diffN_widths[i_block - 1]:
                self.diffN_lins.append(nn.Linear(diffN_widths[i_block - 1], diffN_widths[i_block], device=device))
                self.add_module("lin_" + str(i_block), self.diffN_lins[-1])
            else:
                self.diffN_lins.append(nn.Identity())
                self.add_module("Id_" + str(i_block), self.diffN_lins[-1])
            block = diffNet.DiffusionNetBlock(C_width=diffN_widths[i_block],
                                              mlp_hidden_dims=[diffN_widths[i_block], diffN_widths[i_block]],
                                              dropout=diffN_dropout,
                                              diffusion_method=diffusion_method,
                                              with_gradient_features=with_gradient_features,
                                              with_gradient_rotations=with_gradient_rotations)
            block.to(device)
            self.diffN_blocks.append(block)
            self.add_module("block_" + str(i_block), self.diffN_blocks[-1])

        if not shared_diff:
            self.diffN_blocks_ag = []
            self.diffN_lins_ag = []
            for i_block in range(len(diffN_widths)):
                if i_block > 0 and diffN_widths[i_block] != diffN_widths[i_block - 1]:
                    self.diffN_lins_ag.append(
                        nn.Linear(diffN_widths[i_block - 1], diffN_widths[i_block], device=device))
                    self.add_module("lin_ag_" + str(i_block), self.diffN_lins_ag[-1])
                else:
                    self.diffN_lins_ag.append(nn.Identity())
                    self.add_module("Id_ag_" + str(i_block), self.diffN_lins_ag[-1])
                block = diffNet.DiffusionNetBlock(C_width=diffN_widths[i_block],
                                                  mlp_hidden_dims=[diffN_widths[i_block], diffN_widths[i_block]],
                                                  dropout=diffN_dropout,
                                                  diffusion_method=diffusion_method,
                                                  with_gradient_features=with_gradient_features,
                                                  with_gradient_rotations=with_gradient_rotations)
                block.to(device)
                self.diffN_blocks_ag.append(block)
                self.add_module("block_ag_" + str(i_block), self.diffN_blocks_ag[-1])

        self.seg_module = nn.Sequential()
        self.seg_module.add_module(f"conv_{0}",
                                   torch.nn.Conv1d(diffN_widths[-1] * 2 + diffN_widths[0], seg_widths[0], 1))
        self.seg_module.add_module(f"drop_{0}", nn.Dropout(p=pdrop))
        self.seg_module.add_module(f"bn_{0}", nn.BatchNorm1d(seg_widths[0]))
        for i_block in range(1, len(seg_widths)):
            if i_block > 0:
                self.seg_module.add_module(f"relu_{i_block}", nn.ReLU())
            self.seg_module.add_module(f"conv_{i_block}", nn.Conv1d(seg_widths[i_block - 1], seg_widths[i_block], 1))
            self.seg_module.add_module(f"bn_{i_block}", nn.BatchNorm1d(seg_widths[i_block]))
        self.seg_module.add_module(f"conv_out", nn.Conv1d(seg_widths[- 1], 1, 1))

        self.to(device)

    def define_DiffBlock(self, diffN_widths, diffN_dropout, name="block_"):
        diffN_blocks = []
        diffN_lins = []
        for i_block in range(len(diffN_widths)):
            if i_block > 0 and diffN_widths[i_block] != diffN_widths[i_block - 1]:
                diffN_lins.append(nn.Linear(diffN_widths[i_block - 1], diffN_widths[i_block], device=self.device))
                self.add_module("lin_" + str(i_block), diffN_lins[-1])
            else:
                diffN_lins.append(nn.Identity())
                self.add_module("Id_" + str(i_block), diffN_lins[-1])
            block = diffNet.DiffusionNetBlock(C_width=diffN_widths[i_block],
                                              mlp_hidden_dims=[diffN_widths[i_block], diffN_widths[i_block]],
                                              dropout=diffN_dropout,
                                              diffusion_method=self.diffusion_method,
                                              with_gradient_features=self.with_gradient_features,
                                              with_gradient_rotations=self.with_gradient_rotations)
            block.to(self.device)
            diffN_blocks.append(block)
            self.add_module(name + str(i_block), diffN_blocks[-1])
            self.add_module("lin" + name + str(i_block), diffN_lins[-1])
        return diffN_blocks, diffN_lins

    def forward(self, x1, mass1, x2, mass2,
                L1=None, evals1=None, evecs1=None, gradX1=None, gradY1=None, edges1=None, faces1=None,
                L2=None, evals2=None, evecs2=None, gradX2=None, gradY2=None, edges2=None, faces2=None, ):
        if x1.shape[-1] != self.in_dim or x2.shape[-1] != self.in_dim:
            raise ValueError(
                "DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in, x1.shape[-1]))

        if len(x1.shape) == 2 or len(x2.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x1 = x1.unsqueeze(0)
            mass1 = mass1.unsqueeze(0)
            if L1 != None: L1 = L1.unsqueeze(0)
            if evals1 != None: evals1 = evals1.unsqueeze(0)
            if evecs1 != None: evecs1 = evecs1.unsqueeze(0)
            if gradX1 != None: gradX1 = gradX1.unsqueeze(0)
            if gradY1 != None: gradY1 = gradY1.unsqueeze(0)

            x2 = x2.unsqueeze(0)
            mass2 = mass2.unsqueeze(0)
            if L2 != None: L2 = L2.unsqueeze(0)
            if evals2 != None: evals2 = evals2.unsqueeze(0)
            if evecs2 != None: evecs2 = evecs2.unsqueeze(0)
            if gradX2 != None: gradX2 = gradX2.unsqueeze(0)
            if gradY2 != None: gradY2 = gradY2.unsqueeze(0)

        elif len(x1.shape) == 3 or len(x2.shape) == 3:
            appended_batch_dim = False
        else:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        # Apply the first linear layer
        x1 = self.first_lin(x1)
        x2 = self.first_lin(x2)
        # Apply each of the blocks
        for i, (lin, b) in enumerate(zip(self.diffN_lins, self.diffN_blocks)):
            x1 = lin(x1)
            x1 = b(x1, mass1, L1, evals1, evecs1, gradX1, gradY1)

            if not self.shared_diff:
                x2 = self.diffN_lins_ag[i](x2)
                x2 = self.diffN_blocks_ag[i](x2, mass2, L2, evals2, evecs2, gradX2, gradY2)
            else:
                x2 = lin(x2)
                x2 = b(x2, mass2, L2, evals2, evecs2, gradX2, gradY2)

            if i == 0:
                pointfeat1 = x1
                pointfeat2 = x2
        # Max pooling to obtain global feats
        globalfeat1 = torch.max(x1, 1, keepdim=True)[0]  # B,1,F
        globalfeat2 = torch.max(x2, 1, keepdim=True)[0]  # B,1,F

        xf1 = torch.cat([globalfeat1, globalfeat2], 2).repeat(1, pointfeat1.size()[1], 1)  # B,N1,2F
        xf1 = torch.cat([pointfeat1, xf1], 2)  # B,N1,2F+pF
        xf2 = torch.cat([globalfeat2, globalfeat1], 2).repeat(1, pointfeat2.size()[1], 1)  # B,N2,2F
        xf2 = torch.cat([pointfeat2, xf2], 2)  # B,N2,2F+pF
        x = torch.cat((xf1, xf2), 1)  # B,N1+N2,2F+pF

        ## Segmentation
        x = x.transpose(2, 1).contiguous()
        x = self.seg_module(x)
        x = x.transpose(2, 1).contiguous()

        # Remove batch dim if we added it
        if appended_batch_dim:
            x = x.squeeze(0)

        return x


## PiNet ##
class STNkd(nn.Module):
    def __init__(self, k=64, incr_dims=[64, 128, 1024, ], decr_dims=[512, 256]):
        super(STNkd, self).__init__()

        self.conv_seq = nn.Sequential()
        self.conv_seq.add_module(f"STN_{0}", torch.nn.Conv1d(k, incr_dims[0], 1))
        self.conv_seq.add_module(f"STNbn_{0}", nn.BatchNorm1d(incr_dims[0]))
        self.conv_seq.add_module(f"STNrelu_{0}", nn.ReLU())
        for i_block in range(1, len(incr_dims)):
            self.conv_seq.add_module(f"STN_{i_block}", torch.nn.Conv1d(incr_dims[i_block - 1], incr_dims[i_block], 1))
            self.conv_seq.add_module(f"STNbn_{i_block}", nn.BatchNorm1d(incr_dims[i_block]))
            self.conv_seq.add_module(f"STNrelu_{i_block}", nn.ReLU())

        self.decr_seq = nn.Sequential()
        self.decr_seq.add_module(f"STN_{len(incr_dims)}", torch.nn.Linear(incr_dims[-1], decr_dims[0]))
        self.decr_seq.add_module(f"STNbn_{len(incr_dims)}", nn.BatchNorm1d(decr_dims[0]))
        self.decr_seq.add_module(f"STNrelu_{len(incr_dims)}", nn.ReLU())
        for i_block in range(1, len(decr_dims)):
            self.decr_seq.add_module(f"STN_{len(incr_dims) + i_block}",
                                     torch.nn.Linear(decr_dims[i_block - 1], decr_dims[i_block]))
            self.decr_seq.add_module(f"STNbn_{len(incr_dims) + i_block}", nn.BatchNorm1d(decr_dims[i_block]))
            self.decr_seq.add_module(f"STNrelu_{len(incr_dims) + i_block}", nn.ReLU())
        self.decr_seq.add_module(f"STN_{len(incr_dims) + len(decr_dims)}",
                                 torch.nn.Linear(decr_dims[-1], k * k))

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv_seq(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze(2)
        x = self.decr_seq(x)

        iden = torch.autograd.Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,
                                                                                                           self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat4(nn.Module):
    def __init__(self, d=5, global_feat=True, feature_transform=False, geo=False,
                 dims=[64, 128, 1024], STN_incr_dims=[64, 128, 1024, ], STN_decr_dims=[512, 256]):
        super(PointNetfeat4, self).__init__()
        self.geo = geo
        if geo:
            self.stn = STNkd(k=3, incr_dims=STN_incr_dims, decr_dims=STN_decr_dims)
        else:
            self.stn = STNkd(k=d, incr_dims=STN_incr_dims, decr_dims=STN_decr_dims)

        self.first_seq = nn.Sequential()
        self.first_seq.add_module(f"PNconv_{0}", torch.nn.Conv1d(d, dims[0], 1))
        self.first_seq.add_module(f"PNbn_{0}", nn.BatchNorm1d(dims[0]))
        self.first_seq.add_module(f"PNrelu_{0}", nn.ReLU())

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=dims[0], incr_dims=STN_incr_dims, decr_dims=STN_decr_dims)

        self.seq = nn.Sequential()
        for i_block in range(1, len(dims)):
            self.seq.add_module(f"PNconv_{i_block - 1}", torch.nn.Conv1d(dims[i_block - 1], dims[i_block], 1))
            self.seq.add_module(f"PNbn_{i_block - 1}", nn.BatchNorm1d(dims[i_block]))
            if i_block != len(dims) - 1:
                self.seq.add_module(f"PNrelu_{i_block - 1}", nn.ReLU())
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        n_pts = x.size()[2]
        if self.geo:
            geox = x[:, 0:3, :]
            fx = x[:, 3:, :]
            trans = self.stn(geox)
            geox = geox.transpose(2, 1)
            geox = torch.bmm(geox, trans)
            geox = geox.transpose(2, 1)
            x = torch.cat((geox, fx), dim=1)
        else:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = self.first_seq(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.seq(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze(2)
        if self.global_feat:
            x = torch.unsqueeze(x, 2)
            return x, pointfeat, trans, trans_feat
        else:
            x = torch.unsqueeze(x, 2).repeat(1, 1, n_pts)
            return x, pointfeat, trans, trans_feat


# modified from PointNetDenseCls12 and PointNetDenseCls12geo
class PiNet(nn.Module):
    def __init__(self, feature_transform=False, pdrop=0.0, id=5, geo=False,
                 PN_dims=[64, 128, 1024],
                 STN_incr_dims=[64, 128, 1024], STN_decr_dims=[512, 256], seg_dims=[2112, 1024, 512, 256, 128, 64]):
        super(PiNet, self).__init__()
        # self.k = k
        self.feature_transform = feature_transform
        self.geo = geo
        self.feat = PointNetfeat4(d=id, global_feat=True, feature_transform=feature_transform, geo=geo,
                                  dims=PN_dims, STN_incr_dims=STN_incr_dims, STN_decr_dims=STN_decr_dims)

        # compact predicition
        self.conv_blocks = []
        self.seg_module = nn.Sequential()
        self.seg_module.add_module(f"conv_{0}", torch.nn.Conv1d(PN_dims[0] + 2 * PN_dims[-1], seg_dims[0], 1))
        # self.seg_module.add_module(f"bn_{0}", nn.BatchNorm1d(seg_dims[0]))
        # self.seg_module(f'relu_{0}', nn.ReLU())
        for i_block in range(1, len(seg_dims)):
            if i_block == 1:
                self.seg_module.add_module("dp", nn.Dropout(p=pdrop))
            self.seg_module.add_module(f"bn_{i_block - 1}", nn.BatchNorm1d(seg_dims[i_block - 1]))
            self.seg_module.add_module(f'relu_{i_block - 1}', nn.ReLU())
            self.seg_module.add_module(f"conv_{i_block}", torch.nn.Conv1d(seg_dims[i_block - 1], seg_dims[i_block], 1))
        self.seg_module.add_module(f"conv_{len(seg_dims)}", torch.nn.Conv1d(seg_dims[-1], 1, 1))

    def forward(self, x1, x2):
        batchsize = x1.size()[0]
        n_pts = x1.size()[2]
        x1gf, x1pf, trans1, trans_feat1 = self.feat(x1)
        x2gf, x2pf, trans2, trans_feat2 = self.feat(x2)

        # global
        xf1 = torch.cat([x1gf, x2gf], 1)  # [b, 2* PN_dims[-1] ]
        xf2 = torch.cat([x2gf, x1gf], 1)  # [b, 2* PN_dims[-1] ]

        # point feat concat with global
        xf1 = xf1.repeat(1, 1, x1pf.size()[2])  # [b, 2* PN_dims[-1], np1 ]
        xf2 = xf2.repeat(1, 1, x2pf.size()[2])  # [b, 2* PN_dims[-1], np2 ]
        x1a = torch.cat([x1pf, xf1], 1)  # [b, PN_dims[0] + 2* PN_dims[-1], np1 ]
        x2a = torch.cat([x2pf, xf2], 1)  # [b, PN_dims[0] + 2* PN_dims[-1], np2 ]
        x = torch.cat((x1a, x2a), 2)  # [b, PN_dims[0] + 2* PN_dims[-1], np1 + np2 ]

        # MLP
        x = self.seg_module(x)
        x = x.transpose(2, 1).contiguous()

        x = x.view(-1, 1)

        x = x.view(batchsize, x1.size()[2] + x2.size()[2], 1)
        return x, trans_feat1, trans_feat2
