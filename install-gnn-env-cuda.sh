#!/bin/bash

module purge
module load 2019
module load CUDA/10.1.243
module load Python/3.7.5-foss-2019b

conda env remove -y -n meta-gnn

# maybe try to install the other env and then install these things in that env
#conda env create -f env.yaml

conda create -y -n meta-gnn python==3.7.5 pytorch-lightning==1.5.3 nltk==3.6.5 pytorch==1.7.1

source activate meta-gnn

conda install -c dglteam dgl-cuda10.1
conda install pandas
conda install importlib_resources

pip3 install torchtext==0.8.0

# install the correct pytorch version (for cuda and not for cpu!); Pytorch Cuda 10.2 and torch 1.8.1 work well together
conda install -c anaconda cudatoolkit=10.1

pip3 install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --no-index --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install --no-index --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-geometric


#pip3 install --no-cache-dir torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#pip3 install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu101.html
#pip3 install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu101.html
##pip3 install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu101.html
##pip3 install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu101.html
#pip3 install --no-cache-dir torch-geometric

#pip3 install -r requirements.txt

python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
