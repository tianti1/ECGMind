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
    --ckpt_path "/root/ecg/FocusECG/ckpt/pre_train/p1-2_people2000_segmentall_sample_step100_data/ST-MEM/202501041549/min_val_loss=20.998672485351562.pth" \
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