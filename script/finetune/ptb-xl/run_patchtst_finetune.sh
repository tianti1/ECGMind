#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "finetune" \
    --dataset_name "ptb-xl" \
    --train_data_path "/root/data/ptb-xl/train.txt" \
    --val_data_path "/root/data/ptb-xl/val.txt" \
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
    --pretrain_model_freeze true \
    --ckpt_path "/root/ecg/FocusECG/ckpt/pre_train/chapman_ningbo_code15/PatchTST/202501142341/min_val_loss=76.25680541992188.pth" \
    --classifier_head_name "mlp_v1" \
    --class_n 5 \
    --model_name "PatchTST" \
    --num_input_channels 1 \
    --signal_length 2250 \
    --patch_length 75 \
    --patch_stride 75 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --encoder_num_heads 12 \
    --mask_ratio 0.75 \
    --mask_type "random" \
    --use_cls_token true
