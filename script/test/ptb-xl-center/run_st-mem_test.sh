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
    --max_epoch_num 1000 \
    --early_stop_patience 60 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_patience 20 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --pretrain_model_freeze false \
    --ckpt_path "/root/ecg/FocusECG/ckpt/classifier/ptb-xl-center/ST-MEM/202501061446/max_f1=0.41272895480680677.pth" \
    --class_n 5 \
    --model_name "ST-MEM" \
    --mlp_ratio 4 \
    --num_input_channels 1 \
    --signal_length 1200 \
    --patch_length 30 \
    --embed_dim 256 \
    --encoder_depth 48 \
    --encoder_num_heads 16 \
    --decoder_embed_dim 64 \
    --decoder_depth 8 \
    --decoder_num_heads 8