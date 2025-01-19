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
    --max_epoch_num 100 \
    --early_stop_patience 30 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_patience 20 \
    --scheduler_factor 0.8 \
    --scheduler_min_lr 1e-8 \
    --pretrain_model_freeze true \
    --ckpt_path "/root/ecg_ai/FocusECG/FocusECG/ckpt/classifier/cpsc2018/FocusMae/202501152044/max_f1=0.9598923100801957.pth" \
    --classifier_head_name "mlp_v1" \
    --class_n 9 \
    --model_name "FocusMae" \
    --num_input_channels 1 \
     --signal_length 2250 \
    --patch_length 75 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --encoder_num_heads 12 \
    --decoder_embed_dim 256 \
    --decoder_depth 4 \
    --decoder_num_heads 4 \
    --mlp_ratio 4 \
    --mask_ratio 0.75 \
    --mask_type "period" \
    --use_cls_token true
