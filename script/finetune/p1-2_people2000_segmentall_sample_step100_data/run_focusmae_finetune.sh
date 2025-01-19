#!/bin/bash

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行预训练脚本

python main.py \
    --task "finetune" \
    --dataset_name "p1-2_people2000_segmentall_sample_step100_data" \
    --train_data_path "/root/data/p0_r_center/train.txt" \
    --val_data_path "/root/data/p0_r_center/val.txt" \
    --data_standardization true \
    --device "cuda" \
    --batch_size 128 \
    --max_epoch_num 100 \
    --early_stop_patience 30 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_patience 20 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --pretrain_model_freeze true \
    --ckpt_path "/root/ecg_ai/FocusECG/FocusECG/ckpt/pre_train/p1-2_people2000_segmentall/focusmae/202412262158/min_val_loss=21.37238883972168.pth" \
    --classifier_head_name "mlp_v1" \
    --class_n 4 \
    --model_name "FocusMae" \
    --num_input_channels 1 \
    --signal_length 1200 \
    --patch_size 30 \
    --embed_dim 256 \
    --mask_ratio 0.75 \
    --mask_type "period" \
    --use_cls_token true
