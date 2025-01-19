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
    --early_stop_patience 50 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_patience 20 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --pretrain_model_freeze true \
    --ckpt_path "/root/ECG_syq/FocusECG/ckpt/pre_train/chapman_ningbo_code15/RMae/202501181523/min_val_loss=34.797340393066406.pth" \
    --classifier_head_name "mlp_v1" \
    --class_n 5 \
    --model_name "RMae" \
    --num_input_channels 1 \
    --signal_length 2250 \
    --patch_size 75 \
    --embed_dim 768 \
    --mask_ratio 0.75 \
    --mask_type "random" \
    --use_cls_token true \
    --patch_length 75 \
    --patch_stride 75 \
    --encoder_depth 12 \
    --encoder_num_heads 12 \
    --decoder_embed_dim 256 \
    --decoder_depth 4 \
    --decoder_num_heads 4 \
    --mlp_ratio 4 \
    --mask_ratio 0.75 \
    --use_cls_token true
