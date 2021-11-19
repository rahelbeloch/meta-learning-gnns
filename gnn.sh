#!/bin/bash

module purge
module load 2019
module load Anaconda3/2018.12

conda env remove -y -n meta-env

# maybe try to install the other env and then install these things in that env
#conda env create -f env.yaml

conda create -y -n meta-env python=3.7.5 pytorch==1.7.1

source activate meta-env

conda install -c dglteam dgl-cuda10.1

pip3 install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-geometric

pip3 install -r requirements.txt
