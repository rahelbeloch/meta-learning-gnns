#!/bin/bash

module purge
module load 2021
module load Python/3.9.5-GCCcore-10.3.0;

conda env remove -y -n gnn-env

# maybe try to install the other env and then install these things in that env
#conda env create -f env.yaml

conda create -y -n gnn-env python=3.7.5 pytorch==1.7.1

#conda activate gnn-env

source activate gnn-env

pip3 install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-geometric

#pip3 install -r requirements.txt
