#!/bin/bash
MODEL="/workspace/models/Qwen2-0.5B-Instruct" 
DATA="/workspace/qwen-main/data/train_data_law.json"
OUTPUT_DIR="/workspace/qwen-main/output/qwen2-0.5B-Instruct"
export CUDA_VISIBLE_DEVICES=0

python -u finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --bf16 True \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 5 \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_strategy "epoch" \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True

