#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_rtdm.py --output_dir "exp/maskmodelv4" \
--max_train_steps  50000 \
--resolution 512 \
--first_stage_model_config "config/psr_rtdm.yaml" \
--train_data_dir "data/train" \
--dataloader_num_workers 8 \
--train_batch_size 12 \
--optimizer_type 'adamw' \
--learning_rate 5e-5 \
--lr_scheduler "cosine" \
--lr_warmup_steps 500 \
--set_grads_to_none \
--checkpointing_steps 2000 \
--gradient_accumulation_steps 1
