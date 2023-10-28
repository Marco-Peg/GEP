import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import wandb
from OGEP.Models.models import DiffNet_AbAgPredictor
from Utils.dataset import datasetCreator
from Utils.losses import metrics_dict
from Utils.train_utils import run_epoch, collate_None


def run_DiffNet(model, data):
    preds = model(data["ab_feats"], data["ab_massvec"], data["ag_feats"], data["ag_massvec"],
                  L1=data["ab_L"], evals1=data["ab_evals"], evecs1=data["ab_evecs"], gradX1=data["ab_gradX"],
                  gradY1=data["ab_gradY"], faces1=data.get("ab_faces", None),
                  L2=data["ag_L"], evals2=data["ag_evals"], evecs2=data["ag_evecs"], gradX2=data["ag_gradX"],
                  gradY2=data["ag_gradY"], faces2=data.get("ag_faces", None)).cpu()
    if preds.dim() < 3:
        preds = preds.view(1, -1, 1)  # batch x n_points x 1
    return preds

def parse_params():
    parser = argparse.ArgumentParser(description='Train diffusionNet on AbAg prediction.')
    parser.add_argument('-p', '--params-files', dest='params_file', default='Param_runs/diffNet_mesh_split.json',
                        type=str, help='json file containing the params of the training')
    parser.add_argument('-s', '--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()

    # load model params
    with open(args.params_file, 'r') as f:
        params = json.load(f)

    params["seed"] = args.seed

    return args, params
