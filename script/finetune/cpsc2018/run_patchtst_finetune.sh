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
    --batch_size 512 \
    --max_epoch_num 1000 \
    --early_stop_patience 60 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_patience 20 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --pretrain_model_freeze false \
    --ckpt_path "/root/ecg/FocusECG/ckpt/pre_train/chapman_ningbo_code15/PatchTST/202501190023/min_val_loss=44.838966369628906.pth" \
    --class_n 9 \
    --model_name "PatchTST" \
    --num_input_channels 1 \
    --signal_length 2250 \
    --patch_length 75 \
    --patch_stride 75 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --encoder_num_heads 12 \
    --norm_layer 'LayerNorm' \
    --mask_ratio 0.75 \
    --mask_type 'random' \
    --all_encode_norm_layer 'LayerNorm' \
    --use_cls_token true
