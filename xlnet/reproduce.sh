#!/bin/bash

export MAIN_DIR=$(pwd)

# activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xlnet

# reproduce SST-B finetuning
cd $MAIN_DIR
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier.py \
    --do_train=True \
    --do_eval=False \
    --task_name=sts-b \
    --data_dir=${MAIN_DIR}/data/STS-B \
    --output_dir=proc_data/sts-b \
    --model_dir=exp/sts-b \
    --uncased=False \
    --spiece_model_file=${MAIN_DIR}/model/xlnet_cased_L-24_H-1024_A-16/spiece.model \
    --model_config_path=${MAIN_DIR}/model/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
    --init_checkpoint=${MAIN_DIR}/model/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=4 \
    --learning_rate=5e-5 \
    --train_steps=1200 \
    --warmup_steps=120 \
    --save_steps=600 \
    --is_regression=True

# evaluate finetuning results
cd $MAIN_DIR
CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
    --do_train=False \
    --do_eval=True \
    --task_name=sts-b \
    --data_dir=${MAIN_DIR}/data/STS-B \
    --output_dir=proc_data/sts-b \
    --model_dir=exp/sts-b \
    --uncased=False \
    --spiece_model_file=${MAIN_DIR}/model/xlnet_cased_L-24_H-1024_A-16/spiece.model \
    --model_config_path=${MAIN_DIR}/model/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
    --max_seq_length=128 \
    --eval_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --eval_all_ckpt=True \
    --is_regression=True
