#!/bin/bash

# Environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/home/saumya/internal_vids'    # Directory containing video files
DATA_PATH='/home/saumya/internal_vids'   # Directory containing CSV annotation files and videos
# MODEL_PATH='/home/saumya/pytorch_model_distilled_B14_ft_k710_f8.bin'    # Path to pretrained checkpoint
MODEL_PATH='/home/saumya/1B_ft_k710_f8.pth'

# GPU and CPU configurations
PARTITION='video'
GPUS=8
GPUS_PER_NODE=8
CPUS_PER_TASK=16 

JOB_NAMES=("full_tuning_S_model_3_frames_37_yolo_crops_multilabel")
# JOB_NAMES=("full_tuning_B_model_37_yolo_crops_singlelabel")

for JOB_NAME in "${JOB_NAMES[@]}"; do
    OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
    LOG_DIR="./logs/${JOB_NAME}"
    # Conditionally set the model path
    if [[ "$JOB_NAME" == *"1B"* ]]; then
        MODEL_PATH='/home/saumya/1B_ft_k710_f8.pth'
        MODEL='internvideo2_1B_patch14_224'
        NUM_FRAMES=8
        echo "Using 1B model for job: ${JOB_NAME}"
    elif [[ "$JOB_NAME" == *"_B_"* ]]; then
        MODEL_PATH='/home/saumya/pytorch_model_distilled_B14_ft_k710_f8.bin'
        MODEL='internvideo2_base_patch14_224'
        NUM_FRAMES=8
        echo "Using B model for job: ${JOB_NAME}"
    else
        MODEL_PATH='/home/saumya/pytorch_model_distilled_S14_ft_k710_f8.bin'
        MODEL='internvideo2_small_patch14_224'
        NUM_FRAMES=3
        echo "Using S model for job: ${JOB_NAME}"
    fi
    # Command to run
    python run_linear_probing_inference.py \
        --model ${MODEL} \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'HMDB51' \
        --nb_classes 3 \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --steps_per_print 10 \
        --batch_size 1 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames ${NUM_FRAMES} \
        --num_workers 2 \
        --warmup_epochs 0 \
        --tubelet_size 1 \
        --epochs 10 \
        --lr 2e-3 \
        --min_lr 0 \
        --drop_path 0.0 \
        --head_drop_path 0.0 \
        --fc_drop_rate 0.3 \
        --layer_decay 1.0 \
        --use_checkpoint \
        --checkpoint_num 0 \
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
        --device cuda:1 \
        --eval \
        --test_best \
        --save_preds \
        --sample_path /home/saumya/internal_vids/test_fighting_videos_combined_cropped_longer_v4_8_frames \
        --eval_multilabel
        # --sample_path /home/saumya/internal_vids/test_camera_vids_8_frames/falling \
        # --sample_path /home/saumya/internal_vids/test_fighting_videos_combined_cropped_longer_v4_8_frames
        # --sample_path /home/saumya/internal_vids/atlantis_test_vids_combined_cropped_longer_v4_8_frames
        # --sample_path /home/saumya/internal_vids/demo_vids_combined_cropped_longer_v4_16_frames/2024-08-26_21-10-58_utc_838154f6-0930-4d33-903f-13d2eb66ea71_segment_9.mp4
        # --sample_path /home/saumya/internal_vids/gemini_generated_vids_negative_combined_cropped_longer_v4_16_frames
        # --sample_path /home/saumya/internal_vids/gemini_generated_hard_negatives_combined_cropped_longer_v4_16_frames
        # --sample_path /home/saumya/internal_vids/gemini_generated_hard_negatives \
        # --sample_path /home/saumya/internal_vids/test_camera_vids_8_frames/negative
        # --sample_path /home/saumya/internal_vids/darden_test_vids_combined_cropped_longer_v4/negative
        # --sample_path /home/saumya/internal_vids/darden_test_vids_combined_cropped_longer_v4_16_frames
        # --sample_path /home/saumya/internal_vids/darden_test_vids_combined_cropped_longer_v3/falling
        # --sample_path /home/saumya/internal_vids/test_camera_alerts_8_frames/climbing
        # --sample_path /home/saumya/internal_vids/test_vids_combined_cropped_longer/actively-taking-objects
done