#!/bin/bash

module purge
module load 2021
module load Python/3.9.5-GCCcore-10.3.0;

conda env remove -y -n gnn-env-new

# maybe try to install the other env and then install these things in that env
#conda env create -f env.yaml

conda create -y -n gnn-env-new python==3.9.5 pytorch-lightning==1.5.0 nltk==3.6.5

source activate gnn-env-new

# TODO: also install GPU version of DGL?
conda install -c dglteam dgl-cuda10.2
conda install pandas

pip install torchtext==0.9

# install the correct pytorch version (for cuda and not for cpu!); Pytorch Cuda 10.2 and torch 1.8.1 work well together
conda install pytorch cudatoolkit=10.2 -c pytorch

pip3 install --no-cache-dir torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip3 install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
#pip3 install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
#pip3 install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip3 install --no-cache-dir torch-geometric

#pip3 install -r requirements.txt