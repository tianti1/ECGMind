#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "finetune" \
    --dataset_name "cpsc2018" \
    --train_data_path "/root/data/cpsc2018/train.txt" \
    --val_data_path "/root/data/cpsc2018/val.txt" \
    --data_standardization true \
    --device "cuda" \
    --batch_size 256 \
    --max_epoch_num 1000 \
    --early_stop_patience 30 \
    --learning_rate 1e-5 \
    --weight_decay 1e-3 \
    --scheduler_patience 20 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --pretrain_model_freeze false \
    --ckpt_path "/root/ecg_ai/FocusECG/FocusECG/ckpt/pre_train/icentiallk-p01/FocusMae/202501070232/min_val_loss=23.90247344970703.pth" \
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
