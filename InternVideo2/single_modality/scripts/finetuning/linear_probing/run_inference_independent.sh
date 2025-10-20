#!/bin/bash

# Environment variables
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Job parameters
# JOB_NAME='linear_probing_B_model_hmdb_data_loader_no_mixup_no_smoothing_bhavna_datasets'
# JOB_NAME='full_tuning_B_model_hmdb_data_loader_bhavna_datasets_singlelabel_cleaned_4_filtered'
# JOB_NAME='full_tuning_B_model_hmdb_data_loader_bhavna_datasets_singlelabel_cleaned_4_filtered_more_climbing_vids'
# JOB_NAME='attentive_probing_B_model_hmdb_data_loader_bhavna_datasets_singlelabel_cleaned_larger_including_negative_category_higher_internal_weight'
# JOB_NAME='full_tuning_B_model_hmdb_data_loader_bhavna_datasets_singlelabel_cleaned_5_combined_cropped'
# JOB_NAME='full_tuning_B_model_hmdb_data_loader_bhavna_datasets_singlelabel_cleaned_5'
# JOB_NAME='full_tuning_B_model_new_internal_vids_and_hmdb_falling_label_fixed'
# JOB_NAME='full_tuning_B_model_new_internal_vids_and_hmdb_equal_class_weights'
# JOB_NAME='full_tuning_B_model_all_relabeled_data_random_cropping_equal_class_weights_less_epochs'
# JOB_NAME='full_tuning_B_model_all_relabeled_data_larger_random_cropping_equal_class_weights_less_epochs'
# JOB_NAME='full_tuning_B_model_all_relabeled_data_vlm_filtered_larger_random_cropping_equal_class_weights_less_epochs'
# JOB_NAME='attentive_probing_B_model_all_relabeled_data_random_cropping_equal_class_weights_less_epochs'
# JOB_NAME='linear_probing_B_model_all_relabeled_data_random_cropping_equal_class_weights_finetuned_stage2_encoder'
# JOB_NAME='full_tuning_S_model_all_relabeled_data_including_youtube_no_motorcycles_cleaned_internal_vids_larger_random_cropping_equal_class_weights'
# JOB_NAME='full_tuning_S_model_all_relabeled_data_including_youtube_larger_random_cropping_6_sec_vids'
# JOB_NAME='full_tuning_B_model_18_yolo_crops_no_youtube_2_climbing_vids'
# JOB_NAME='full_tuning_B_model_18_yolo_crops'
JOB_NAME='attentive_probing_1B_model_8_frames_18_yolo_crops'
# JOB_NAME='attentive_probing_1B_model_16_frames_18_yolo_crops'
# JOB_NAME='full_tuning_B_model_21_yolo_crops'
# JOB_NAME='full_tuning_1B_model_18_yolo_crops'

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

# Command to run
python run_inference_independent.py \
    --model internvideo2_1B_patch14_224 \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'HMDB51' \
    --nb_classes 4 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
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
    --sample_path /home/saumya/internal_vids/test_camera_vids_8_frames/falling \
    # --sample_path /home/saumya/internal_vids/gemini_generated_vids_negative_combined_cropped_longer_v4_16_frames
    # --sample_path /home/saumya/internal_vids/test_camera_vids_8_frames/negative
    # --sample_path /home/saumya/internal_vids/darden_test_vids_combined_cropped_longer_v4/negative
    # --sample_path /home/saumya/internal_vids/darden_test_vids_combined_cropped_longer_v4_16_frames
    # --sample_path /home/saumya/internal_vids/darden_test_vids_combined_cropped_longer_v3/falling
    # --sample_path /home/saumya/internal_vids/test_camera_alerts_8_frames/climbing
    # --sample_path /home/saumya/internal_vids/test_vids_combined_cropped_longer/actively-taking-objects
