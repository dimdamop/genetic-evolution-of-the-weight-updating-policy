#!/bin/bash

eval "$(conda shell.bash hook)"
conda env create --prefix ./env --file conda-env.yaml
conda activate ./env
pip install -U "jax[cuda12]<0.7"
pip install -e .[dev]
