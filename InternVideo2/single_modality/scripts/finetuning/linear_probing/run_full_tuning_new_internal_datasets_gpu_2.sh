#!/bin/bash

# Environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Job parameters
JOB_NAME='full_tuning_B_model_16_frames_18_yolo_crops'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/home/saumya/internal_vids'    # Directory containing video files
DATA_PATH='/home/saumya/internal_vids'   # Directory containing CSV annotation files and videos
MODEL_PATH='/home/saumya/pytorch_model_distilled_B14_ft_k710_f8.bin'    # Path to pretrained checkpoint
# MODEL_PATH='/home/saumya/1B_ft_k710_f8.pth'

# GPU and CPU configurations
PARTITION='video'
GPUS=8
GPUS_PER_NODE=8
CPUS_PER_TASK=16

# Command to runc
python run_finetuning.py \
    --model internvideo2_base_patch14_224 \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'HMDB51' \
    --nb_classes 3 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --num_workers 2 \
    --warmup_epochs 0 \
    --tubelet_size 1 \
    --epochs 8 \
    --lr 2e-3 \
    --min_lr 0 \
    --drop_path 0.0 \
    --head_drop_path 0.0 \
    --fc_drop_rate 0.3 \
    --layer_decay 1.0 \
    --use_checkpoint \
    --checkpoint_num 20 \
    --layer_scale_init_value 1e-5 \
    --aa rand-m5-n2-mstd0.25-inc1 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.0 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --dist_eval \
    --bf16 \
    --zero_stage 1 \
    --smoothing 0 \
    --gpu 2 \
    --include_negative_category \
    --train_anno_path /home/saumya/internal_vids/train_bhavna_singlelabel_with_negative_18.csv \
    --test_anno_path /home/saumya/internal_vids/internal_test_bhavna_singlelabel_with_negative_18.csv \
    --test_best \
    --enable_class_weights \
    --eval_yolo_crops \
    --train_yolo_crops \
    --min_padding_ratio_positive 0.0 \
    --max_padding_ratio_positive 1.0 \
    --min_padding_ratio_negative 0.0 \
    --max_padding_ratio_negative 1.0 \
    --test_padding_ratio 0.1 \
    --spatial_augmentation_min_scale 0.08
    # --use_random_yolo_crop
    # --new_spatial_augmentation
    # --spatial_augmentation_min_scale 0.5
    # --save_training_images

