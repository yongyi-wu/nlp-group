#!/bin/bash

export MAIN_DIR=$(pwd)

# install required packages
sudo apt install unzip

# set up environment
cd $MAIN_DIR
source $(conda info --base)/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate xlnet

# download STS-B data
cd $MAIN_DIR
mkdir -p data
cd data
wget https://dl.fbaipublicfiles.com/glue/data/STS-B.zip
unzip STS-B.zip
rm STS-B.zip

# download pretrained XLNet-large
cd $MAIN_DIR
mkdir -p model
cd model
wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
unzip cased_L-24_H-1024_A-16.zip
rm cased_L-24_H-1024_A-16.zip
