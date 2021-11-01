#!/bin/bash

export DECOMP_DIR=$(pwd)/DecompRC/DecompRC

# activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate decomp
export PYTHONWARNINGS="ignore" # ignore tensorflow warnings

# reproduce results
cd $DECOMP_DIR
make inference
