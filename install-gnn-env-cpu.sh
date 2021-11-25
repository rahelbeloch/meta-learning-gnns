#!/bin/bash

conda env remove -y -n meta-env

# maybe try to install the other env and then install these things in that env
#conda env create -f env.yaml

conda create -y -n meta-env python=3.7.5 pytorch==1.7.1

source activate meta-env

conda install -c dglteam dgl

pip3 install --no-index torch==1.7.1+cpu -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip3 install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip3 install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
#pip3 install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
#pip3 install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip3 install torch-geometric

pip3 install -r requirements.txt
