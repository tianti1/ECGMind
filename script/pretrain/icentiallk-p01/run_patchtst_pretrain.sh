#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "pretrain" \
    --dataset_name "icentiallk-p01" \
    --train_data_path "/root/data/p0-1_10s/index_0-42k_step:1.txt" \
    --val_data_path "/root/data/p0-1_10s/index_42-60k_step:1.txt" \
    --data_standardization true \
    --device "cuda" \
    --batch_size 512 \
    --max_epoch_num 1000 \
    --val_every_n_steps 40 \
    --early_stop_patience 50 \
    --learning_rate 1e-3 \
    --weight_decay 0 \
    --scheduler_patience 10 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --model_name "PatchTST" \
    --num_input_channels 1 \
    --signal_length 2500 \
    --patch_length 30 \
    --patch_stride 30 \
    --embed_dim 256 \
    --norm_layer 'LayerNorm' \
    --mask_ratio 0.75 \
    --mask_type 'period' \
    --all_encode_norm_layer 'LayerNorm' \
    --use_cls_token true
