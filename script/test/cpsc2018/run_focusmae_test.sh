#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "test" \
    --dataset_name "cpsc2018" \
    --test_data_path "/root/data/cpsc2018/test.txt" \
    --data_standardization true \
    --device "cuda" \
    --batch_size 256 \
    --max_epoch_num 1000 \
    --early_stop_patience 30 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_patience 20 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --pretrain_model_freeze false \
    --ckpt_path "/root/ecg/FocusECG/ckpt/classifier/cpsc2018/FocusMae/202501081727/max_f1=0.9479843377575963.pth" \
    --classifier_head_name "mlp_v1" \
    --class_n 9 \
    --model_name "FocusMae" \
    --num_input_channels 1 \
    --signal_length 2500 \
    --patch_size 50 \
    --embed_dim 256 \
    --mask_ratio 0.75 \
    --mask_type "period" \
    --use_cls_token true
