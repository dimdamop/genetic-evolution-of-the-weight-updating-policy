#!/bin/bash

# eval "$(conda shell.bash hook)"
# conda env create --prefix ./env --file conda-env.yaml
# conda activate ./env
pip install \
    "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    numpy \
    hydra-core \
    wandb \
    lark==1.1.9 \
    tensorflow==2.16.1 \
    jumanji \
    tensorboardX \
    neptune \
    dm-haiku \
    rlax
