#!/bin/bash
# THROUGHPUT PROBE — full fine-tuning with ZeRO-3 + CPU offload, ~25 steps, R2R only.
set -x
OUTPUT="./checkpoints/probe_fullft"
MODEL=/home/maitree-tiamat/models/navila-siglip-llama3-8b-v1.5-pretrain

torchrun --nnodes=1 --nproc_per_node=2 --master_port=29531 --master_addr=localhost --node_rank=0 \
    llava/train/train_mem.py \
    --longvila_sampler True \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path $MODEL \
    --version llama_3 \
    --seed 10 \
    --data_mixture r2r \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --max_steps 25 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_eval False \
    --save_strategy "no" \
    --fps 0.0 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none
