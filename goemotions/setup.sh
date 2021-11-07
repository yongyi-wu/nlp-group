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
    for datatype in train dev test
    do
        python replace_emotions.py \
            --input data/${datatype}.tsv \
            --mapping_dict data/${dataset}_mapping.json \
            --output_emotion_file ${data_dir}/emotions.txt \
            --output_data ${data_dir}/${datatype}.tsv
    done
done

cd ${main_dir}/data
git clone https://github.com/sarnthil/unify-emotion-datasets.git
cd unify-emotion-datasets
python download_datasets.py --yes
cd ${main_dir}/data
# prepare ISEAR dataset
mv unify-emotion-datasets/datasets/isear/ .
# prepare Emotion-Stimulus dataset
mv unify-emotion-datasets/datasets/emotion-cause/ emosti
mv emosti/Dataset/* emosti/
# prepare EmoInt dataset
mv unify-emotion-datasets/datasets/emoint/ .
cd $main_dir
python prepare_transfer_datasets.py data/ data/
rm -rf ${main_dir}/data/unify-emotion-datasets
