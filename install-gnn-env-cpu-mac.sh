#!/bin/bash

conda env remove -y -n meta-gnn-env-cpu

conda create -y -n meta-gnn-env-cpu

source activate meta-gnn-env-cpu

conda install -y python==3.7.5 nltk==3.6.5
conda install -y pytorch==1.11.0 cpuonly -c pytorch

conda install -y pytorch-lightning -c conda-forge
conda install -y torchmetrics -c conda-forge

conda install -y wandb -c conda-forge
conda install -y pandas
conda install -y importlib_resources
conda install -y matplotlib
conda install -y scipy
conda install -y networkx
conda install -y nltk

conda install -y torchtext -c pytorch

pip install --no-index --no-cache-dir torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html