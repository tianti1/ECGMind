#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "pretrain" \
    --notify "true" \
    --dataset_name "chapman_ningbo_code15" \
    --train_data_path "/root/data/FocusMAE/train.txt" \
    --val_data_path "/root/data/FocusMAE/val.txt" \
    --data_standardization true \
    --device "cuda" \
    --batch_size 512 \
    --max_epoch_num 1000 \
    --val_every_n_steps 40 \
    --early_stop_patience 200 \
    --learning_rate 0.0001 \
    --weight_decay 0 \
    --scheduler_patience 10 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --model_name "ST-MEM" \
    --num_input_channels 1 \
    --signal_length 2250 \
    --patch_length 75 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --encoder_num_heads 12 \
    --decoder_embed_dim 256 \
    --decoder_depth 4 \
    --decoder_num_heads 4 \
    --mlp_ratio 4 \
    --mask_ratio 0.75 \
    --mask_type "period" \
    --use_cls_token true
