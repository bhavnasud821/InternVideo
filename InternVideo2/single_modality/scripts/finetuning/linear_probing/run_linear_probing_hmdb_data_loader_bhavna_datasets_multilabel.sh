#!/bin/bash

# Environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Job parameters
JOB_NAME='linear_probing_B_model_hmdb_data_loader_bhavna_datasets_multilabel_cleaned_larger'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/home/saumya/internal_vids'    # Directory containing video files
DATA_PATH='/home/saumya/internal_vids'   # Directory containing CSV annotation files and videos
MODEL_PATH='/home/saumya/pytorch_model_distilled_B14_ft_k710_f8.bin'    # Path to pretrained checkpoint

# GPU and CPU configurations
PARTITION='video'
GPUS=8
GPUS_PER_NODE=8
CPUS_PER_TASK=16

# Command to run
python run_linear_probing.py \
    --model internvideo2_base_patch14_224 \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'HMDB51' \
    --nb_classes 7 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
    --orig_t_size 8 \
    --num_workers 2 \
    --warmup_epochs 0 \
    --tubelet_size 1 \
    --epochs 20 \
    --lr 2e-3 \
    --min_lr 0 \
    --drop_path 0.0 \
    --head_drop_path 0.0 \
    --fc_drop_rate 0.3 \
    --layer_decay 1.0 \
    --layer_scale_init_value 1e-5 \
    --aa rand-m5-n2-mstd0.25-inc1 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --bf16 \
    --zero_stage 1 \
    --smoothing 0 \
    --multilabel