#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "pretrain" \
    --dataset_name "chapman_ningbo_code15" \
    --train_data_path "/root/data/FocusMAE/train.txt" \
    --val_data_path "/root/data/FocusMAE/val.txt" \
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
    --model_name "FocusMae" \
    --num_input_channels 1 \
    --signal_length 2500 \
    --patch_length 50 \
    --embed_dim 256 \
    --encoder_depth 48 \
    --encoder_num_heads 16 \
    --decoder_embed_dim 64 \
    --decoder_depth 8 \
    --decoder_num_heads 8 \
    --mlp_ratio 2 \
    --mask_ratio 0.75 \
    --mask_type "period" \
    --use_cls_token true
