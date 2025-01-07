#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "test" \
    --dataset_name "ptb-xl-center" \
    --test_data_path "/root/data/ptb-xl-center/test.txt" \
    --data_standardization true \
    --device "cuda" \
    --batch_size 512 \
    --ckpt_path "/root/ecg/FocusECG/ckpt/classifier/ptb-xl-center/PatchTST/202501060058/max_f1=0.42404457093203396.pth" \
    --class_n 5 \
    --model_name "PatchTST" \
    --num_input_channels 1 \
    --signal_length 1200 \
    --patch_length 30 \
    --patch_stride 30 \
    --embed_dim 256 \
    --mask_ratio 0.75 \
    --mask_type "random" \
    --use_cls_token true
