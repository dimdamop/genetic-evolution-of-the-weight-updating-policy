#!/bin/bash

eval "$(conda shell.bash hook)"
conda env create --prefix ./env --file conda-env.yaml
conda activate ./env
pip install numpy==1.24.3 lark==1.1.8 tensorflow==2.13
pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
