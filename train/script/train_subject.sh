#!/bin/bash

# Specify the config file path and the GPU devices to use
export CUDA_VISIBLE_DEVICES=1,2

# Specify the config file path
export XFL_CONFIG=./train/config/subject_512.yaml

# 禁用 wandb
export WANDB_DISABLED=true
# export WANDB_API_KEY=''

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

# 使用修改后的本地训练脚本
accelerate launch --main_process_port 41353 \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=no \
    -m src.train.train_local