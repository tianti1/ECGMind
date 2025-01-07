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
    --ckpt_path "/root/ecg/FocusECG/ckpt/classifier/p1-2_people2000_segmentall_sample_step100_data/PatchTST/202501032100/max_f1=0.741274067718337.pth" \
    --class_n 4 \
    --model_name "PatchTST" \
    --num_input_channels 1 \
    --signal_length 1200 \
    --patch_length 30 \
    --patch_stride 30 \
    --embed_dim 256 \
    --mask_ratio 0.75 \
    --mask_type "random" \
    --use_cls_token true
