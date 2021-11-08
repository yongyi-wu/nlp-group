#!/bin/bash

main_dir=$(pwd)
all_out_dir=${main_dir}/out
mkdir -p $all_out_dir
all_result_dir=${main_dir}/results
mkdir -p $all_result_dir
ckpt_saving_steps=6000

# activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate goemo


# REPRODUCING SECTION V
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
        --output_dir $out_dir \
        --init_checkpoint goemotions/bert/cased_L-12_H-768_A-12/bert_model.ckpt \
        --save_checkpoints_steps $ckpt_saving_steps

    # evaluate performance
    cd $main_dir
    python calculate_metrics.py \
        --test_data ${data_dir}/test.tsv \
        --predictions ${out_dir}/test.tsv.predictions.tsv \
        --output ${out_dir}/results.json \
        --emotion_file ${data_dir}/emotions.txt \
        --noadd_neutral

    cp ${out_dir}/results.json ${all_result_dir}/${dataset}.json
done


# REPRODUCING SECTION VI
for dataset in dialog stimulus affective crowd elect isear tec emoint ssec
do
    case $dataset in
        dialog | affective | ssec )
            multi=true
            ;;
        * )
            unset multi
            ;;
    esac
    data_dir=${main_dir}/data/${dataset}
    result_dir=${all_result_dir}/${dataset}
    mkdir -p $result_dir
    for trainsize in 100 200 500 1000 max
    do
        for setup in baseline freeze nofreeze
        do
            if [[ $setup == "baseline" ]]
            then
                checkpoint=${main_dir}/bert/cased_L-12_H-768_A-12/bert_model.ckpt
            else
                checkpoint=${all_out_dir}/goemo/model.ckpt-10852
                if [[ $setup == "freeze" ]]
                then
                    freeze=true
                else
                    unset freeze
                fi
            fi
            out_dir=${all_out_dir}/${dataset}/${trainsize}_${setup}

            # transfer learning
            cd ${main_dir}/..
            python -m goemotions.bert_classifier \
                --emotion_file ${data_dir}/emotions.txt \
                --data_dir $data_dir \
                --bert_config_file goemotions/bert/cased_L-12_H-768_A-12/bert_config.json \
                --vocab_file goemotions/bert/cased_L-12_H-768_A-12/vocab.txt \
                ${multi:+--multilabel} \
                --output_dir $out_dir \
                --init_checkpoint $checkpoint \
                --save_checkpoints_steps $ckpt_saving_steps \
                --train_fname train_${trainsize}.tsv \
                --learning_rate 2e-5 \
                --num_train_epochs 3.0 \
                ${freeze:+--freeze_layers} \
                --transfer_learning

            # evaluate performance
            cd $main_dir
            python calculate_metrics.py \
                --test_data ${data_dir}/test.tsv \
                --predictions ${out_dir}/test.tsv.predictions.tsv \
                --output ${out_dir}/results.json \
                --emotion_file ${data_dir}/emotions.txt \
                --noadd_neutral

            cp ${out_dir}/results.json ${result_dir}/${trainsize}_${setup}.json

            # save space by deleting all checkpoints
            rm ${out_dir}/model.ckpt*
        done
    done
done
# plot results
cd $main_dir
python plot_section6_results.py $all_result_dir $all_result_dir
