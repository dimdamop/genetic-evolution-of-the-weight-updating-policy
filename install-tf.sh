#!/bin/bash

eval "$(conda shell.bash hook)"
conda env create --prefix ./env --file conda.yaml
conda activate ./env
export LD_LIBRARY_PATH="$PWD/env/lib"
