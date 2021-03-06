#!/bin/bash

main_dir=$(pwd)

# set up environment
source $(conda info --base)/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate goemo

# download pretrained BERT-Base, cased
cd ${main_dir}/bert/
if [[ ! -d cased_L-12_H-768_A-12 ]]
then
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
    unzip cased_L-12_H-768_A-12.zip
    rm cased_L-12_H-768_A-12.zip
fi


# prepare sentiment-grouped dataset and Ekman's taxonomy dataset
cd $main_dir
for dataset in sentiment ekman
do
    data_dir=${main_dir}/data/${dataset}
    mkdir -p $data_dir
    # group labels based on specified mapping
    for split in train dev test
    do
        if [[ ! -f ${data_dir}/${split}.tsv ]]
        then
            python replace_emotions.py \
                --input data/${split}.tsv \
                --mapping_dict data/${dataset}_mapping.json \
                --output_emotion_file ${data_dir}/emotions.txt \
                --output_data ${data_dir}/${split}.tsv
        fi
    done
done

# prepare datasets for evaluating transfer learning
cd ${main_dir}/data
git clone https://github.com/sarnthil/unify-emotion-datasets.git
cd unify-emotion-datasets
python download_datasets.py --yes
cd $main_dir
python prepare_section6_datasets.py data/unify-emotion-datasets/datasets data/
rm -rf data/unify-emotion-datasets
