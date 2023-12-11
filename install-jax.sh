#!/bin/bash

eval "$(conda shell.bash hook)"
conda env create --prefix ./env-jax --file conda-jax.yaml
conda activate ./env-jax
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorflow==2.13
