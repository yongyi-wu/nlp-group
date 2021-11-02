#!/bin/bash

export MAIN_DIR=$(pwd)/DecompRC

# activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate decomp
export PYTHONWARNINGS="ignore" # ignore tensorflow warnings

# reproduce results
cd $MAIN_DIR
make inference
