#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "test" \
    --dataset_name "ptb-xl" \
    --test_data_path "/root/data/ptb-xl/test.txt" \
    --data_standardization true \
    --device "cuda" \
    --batch_size 512 \
    --ckpt_path "/root/ecg/FocusECG/ckpt/classifier/ptb-xl/PatchTST/202501052114/max_f1=0.4248042342227107.pth" \
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
