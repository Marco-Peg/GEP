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

if __name__ == "__main__":
    args, params = parse_params()
    dataset_name = params["dataset_name"]
    # system things
    if torch.cuda.is_available():
        device = params["device"]
    else:
        print("cuda not available, using cpu as deafult")
        device = torch.device("cpu")
