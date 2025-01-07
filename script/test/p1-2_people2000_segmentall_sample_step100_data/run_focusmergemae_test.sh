#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "test" \
    --dataset_name "p1-2_people2000_segmentall_sample_step100_data" \
    --test_data_path "/root/data/p1-2_people2000_segmentall_sample_step100_data/index_84-120k_step:1.txt" \
    --data_standardization true \
    --device "cuda" \
    --batch_size 512 \
    --ckpt_path "/root/ecg_ai/ecg_self_supervised_training/ckpt/classifier/physionet_index_60-72k_step:11/physionet_index_72-78k_step:11/focusmergemae+mlp_v1/normal/202412251104/max_f1=0.8567070318649346.pth" \
    --class_n 4 \
    --model_name "FocusMergeMae" \
    --num_input_channels 1 \
    --signal_length 1200 \
    --patch_length 30 \
    --embed_dim 256 \
    --use_cls_token true