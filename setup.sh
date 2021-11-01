#!/bin/bash

export DECOMP_DIR=$(pwd)/DecompRC/DecompRC

# install required packages
sudo apt install unzip
sudo apt install make

# set up environment
source $(conda info --base)/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate decomp

# download data and weights related to DecompRC
cd $DECOMP_DIR
gdown --id 1p7VrJIEmUY9tAWmx31chhStS-KoRWOJQ
unzip DecompRC-all-models-and-data.zip
mv DecompRC-all-models-and-data/data DecompRC-all-models-and-data/model .
rm -r DecompRC-all-models-and-data.zip DecompRC-all-models-and-data

# download HotpotQA dataset
cd ${DECOMP_DIR}/data
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
mv hotpot_train_v1.1.json hotpot_train_v1.json # TODO: fix convert_hotpot2squad.py
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

# download pretrained BERT for this task
cd ${DECOMP_DIR}/model
gdown --id 1XaMX-u5ZkWGH3f0gPrDtrBK1lKDU-QFk
unzip uncased_L-12_H-768_A-12.zip
rm uncased_L-12_H-768_A-12.zip

# convert HotpotQA into SQuAD style
cd $DECOMP_DIR
python convert_hotpot2squad.py --data_dir ${DECOMP_DIR}/data --task hotpot-all
