#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python pretrain.py \
    # task
    --task "pretrain" \
    # dataset
    --dataset_name "p1-2_people2000_segmentall_sample_step100_data" \
    --train_data_path "/root/data/p1-2_people2000_segmentall_sample_step100_data/index_0-42k_step:1.txt" \
    --val_data_path "/root/data/p1-2_people2000_segmentall_sample_step100_data/index_42-60k_step:1.txt" \
    --data_standardization true \
    # model
    --model_name 'FocusMae' \
    --signal_length 1200 \
    --patch_size 30 \
    --embed_dim 256 \
    --encoder_depth 48 \
    --encoder_num_heads 16 \
    --decoder_embed_dim 64 \
    --decoder_depth 8 \
    --decoder_num_heads 8 \
    --mlp_ratio 2 \
    --norm_layer 'LayerNorm' \
    --mask_ratio 0.75 \
    --mask_type 'period' \
    --all_encode_norm_layer 'LayerNorm' \
    # train
    --batch_size 512 \
    --max_epoch_num 1000 \
    --val_every_n_steps 40 \
    --early_stop_patience 50 \
    --learning_rate 0.001 \
    --weight_decay 0 \
    --scheduler_patience 20 \
    --scheduler_factor 0.5 \
    --scheduler_min_lr 1e-6
