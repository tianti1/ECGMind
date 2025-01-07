#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "finetune" \
    --dataset_name "ptb-xl-center" \
    --train_data_path "/root/data/ptb-xl-center/train.txt" \
    --val_data_path "/root/data/ptb-xl-center/val.txt" \
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
    --ckpt_path "/root/ecg/FocusECG/ckpt/pre_train/p1-2_people2000_segmentall_sample_step100_data/PatchTST/202501022029/min_val_loss=20.453519821166992.pth" \
    --classifier_head_name "mlp_v1" \
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
