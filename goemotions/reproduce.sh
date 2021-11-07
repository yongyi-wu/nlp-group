#!/bin/bash

main_dir=$(pwd)
all_out_dir=${main_dir}/out
mkdir -p $all_out_dir

# activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate goemo


# SECTION V
for dataset in goemo sentiment ekman
do
    if [[ $dataset == "goemo" ]]
    then
        data_dir=${main_dir}/data
    else
        data_dir=${main_dir}/data/${dataset}
    fi
    out_dir=${all_out_dir}/${dataset}

    # finetune BERT
    cd ${main_dir}/..
    python -m goemotions.bert_classifier \
        --emotion_file ${data_dir}/emotions.txt \
        --data_dir $data_dir \
        --bert_config_file goemotions/bert/cased_L-12_H-768_A-12/bert_config.json \
        --vocab_file goemotions/bert/cased_L-12_H-768_A-12/vocab.txt \
        --multilabel \
        --output_dir $out_dir
        --init_checkpoint goemotions/bert/cased_L-12_H-768_A-12/bert_model.ckpt \

    # evaluate performance
    cd $main_dir
    python calculate_metrics.py \
        --test_data ${data_dir}/test.tsv \
        --predictions ${out_dir}/test.tsv.predictions.tsv \
        --output ${out_dir}/results.json \
        --emotion_file ${data_dir}/emotions.txt \
        --noadd_neutral
done
