#!/bin/bash

# conda activate base
# # Install Python and essential packages
# conda create -n IGEP python=3.8.* pip -y
# conda activate IGEP
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge tqdm wandb torchinfo torchmetrics igl -y
conda install -c plotly plotly -y
conda install -c anaconda scikit-learn pandas networkx -y
conda install -c conda-forge biopython -y
conda install -c conda-forge -c schrodinger pymol-bundle --yes 

# Install PyTorch Geometric and related packages using pip
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# Install additional package using pip
pip install egnn-pytorch
