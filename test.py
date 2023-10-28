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

if __name__ == "__main__":

    # system things
    if torch.cuda.is_available():
        device = params["device"]
    else:
        print("cuda not available, using cpu as deafult")
        device = torch.device("cpu")
