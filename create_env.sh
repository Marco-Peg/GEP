#!/bin/bash
# conda create -n GEP python=3.8.15 pip -y
# conda activate GEP
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge tqdm wandb torchinfo torchmetrics igl -y
conda install -c anaconda -c plotly plotly scikit-learn pandas networkx -y

# processing libraries
conda install -c conda-forge -c schrodinger pymol-bundle  biopython --yes 
pip3 install pymeshlab

# IGEP
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.5.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install egnn-pytorch
pip install -U kaleido

## OGEP
pip install potpourri3d robust-laplacian
