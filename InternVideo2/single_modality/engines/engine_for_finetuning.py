#!/usr/bin/env python3
import os
import time
import math
import sys
import json
import numpy as np
from typing import Iterable, Optional
from collections import defaultdict

import torch
from datasets.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
from PIL import Image
import seaborn as sns
import re
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, average_precision_score, precision_recall_curve, precision_score, recall_score

############################################################################
# Helper function for safe barrier (for distributed mode)
############################################################################
def safe_barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

############################################################################
# TRAINING FUNCTIONS
############################################################################
def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    # print(f"[DEBUG] Model outputs: {outputs}")
    loss = criterion(outputs, target)
    return loss, outputs

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    bf16=False,
    internal_loss_scale=1.0,
    multilabel=False
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"Epoch: [{epoch}]"
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        paths = info["path"]
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration

        # Update learning rate and weight decay if schedules are provided
        if (lr_schedule_values is not None or wd_schedule_values is not None) and (data_iter_step % update_freq == 0):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    lr_scale = param_group.get("lr_scale", 1.0)
                    param_group["lr"] = lr_schedule_values[it] * lr_scale
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.bfloat16() if bf16 else samples.half()
            losses, output = train_class_batch(model, samples, targets, criterion)
        else:
            with torch.amp.autocast(device_type='cuda'):
                losses, output = train_class_batch(model, samples, targets, criterion)

        # Scale internal data by loss_scale
        loss_scales = []
        for path in paths:
            if "comb_extracted_tracks" in path or "internal_negative_videos" in path or "internal_positive_videos" in path:
                loss_scales.append(internal_loss_scale)
            else:
                loss_scales.append(1.0)
        loss_scales = torch.tensor(loss_scales).to(device, non_blocking=True)
        if multilabel:
            loss = (losses * loss_scales[:, None]).mean()
        else:
            loss = (losses * loss_scales).mean()
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
            if (data_iter_step + 1) % update_freq == 0 and model_ema is not None:
                model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=((data_iter_step + 1) % update_freq == 0)
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # if mixup_fn is None:
        #     class_acc = (output.argmax(dim=-1) == targets).float().mean()
        # else:
        #     class_acc = None

        metric_logger.update(loss=loss_value)
        # metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr, max_lr_val = 10., 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr_val = max(max_lr_val, group["lr"])
        metric_logger.update(lr=max_lr_val)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            # log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr_val, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

############################################################################
# VALIDATION FUNCTIONS
############################################################################
@torch.no_grad()
def validation_one_epoch(data_loader, model, device, ds=False, bf16=False, output_dir=None):
   return {}

############################################################################
# FINAL TEST FUNCTIONS
############################################################################
@torch.no_grad()
def final_test(data_loader, model, device, file, num_classes, ds=False, bf16=False, train_multilabel=False, eval_multilabel=False, output_dir=None, internal=False):
    """
    Final evaluation after all epochs.
    Overall accuracy (top-1 and top-5) is computed ignoring negative samples (target == -1),
    while precision/recall and Average Precision (AP) are computed using all samples.
    Per-sample predictions are saved and a confusion matrix (ignoring negatives) is generated.
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    all_targets = []
    all_video_ids = []
    all_probs = []

    # Category mapping 
    category_names = {
        0: "climbing",
        1: "fall_floor",
        2: "assault",
        3: "negative" # note that negative category is only used when evaling singlelabel model
    }

    print("category names: ", category_names)
    final_result = []
    for j, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        original_videos, videos, target = batch[0], batch[1], batch[2]
        video_ids = batch[3] if len(batch) >= 4 else ["unknown"] * videos.shape[0]
        segment_indices = batch[4]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            output = model(videos)
        if train_multilabel:
            probs = torch.sigmoid(output)
        else:
            probs = torch.softmax(output, dim=-1)

        all_targets.extend(target.cpu().tolist())
        all_video_ids.extend(video_ids)
        all_probs.append(probs.cpu())

        # Save per-sample predictions (JSON-serialized)
        for i in range(output.size(0)):
            prob_json = json.dumps(probs.data[i].float().cpu().numpy().tolist())
            if eval_multilabel:
                line = f"{video_ids[i]} {segment_indices[i]} {prob_json} {target[i].cpu().numpy()}\n"
                final_result.append(line)
            else:
                line = f"{video_ids[i]} {segment_indices[i]} {prob_json} {int(target[i].cpu().numpy())}\n"
                final_result.append(line)

    with open(file, "w") as f:
        f.write("video_id segment_idx probabilities true_label\n")
        for line in final_result:
            f.write(line)

    # Combine probabilities from all batches
    all_probs = torch.cat(all_probs, dim=0)  # shape [N, C]

    # Average results by segment_idx that correspond to same video
    # Dictionary to store sums of probabilities and counts for averaging
    prob_sums = defaultdict(lambda: torch.zeros(all_probs.shape[1]))  # Shape [C]
    prob_maxes = defaultdict(lambda: torch.zeros(all_probs.shape[1]))  # Shape [C]
    counts = defaultdict(int)
    targets_map = {}

    for video_id, probs, target in zip(all_video_ids, all_probs, all_targets):
        shortened_video_id = video_id
        if eval_multilabel:
            if num_classes > len(target):
                # add negative label to multilabel targets
                target = target + [0] if np.sum(target) > 0 else target + [1]
            targets_map[shortened_video_id] = target
        else:
            if shortened_video_id not in targets_map:
                targets_map[shortened_video_id] = np.zeros(num_classes, dtype=np.float32)
            if target != -1:
                targets_map[shortened_video_id][target] = 1.0
        prob_sums[shortened_video_id] += probs
        if video_id in prob_maxes:
            prob_maxes[shortened_video_id] = torch.max(prob_maxes[shortened_video_id], probs)
        else:
            prob_maxes[shortened_video_id] = probs 
        counts[shortened_video_id] += 1

    print("all_probs original shape ", all_probs.shape)
    print("all targets original len ", len(all_targets))
    # Compute averaged probabilities
    # new_all_probs = torch.stack([prob_sums[vid] / counts[vid] for vid in prob_sums.keys()])
    new_all_probs = torch.stack([prob_maxes[vid] for vid in prob_sums.keys()])
    new_all_targets = [targets_map[vid] for vid in prob_sums.keys()]

    print("all_probs new shape ", new_all_probs.shape)
    print("all targets new len ", len(new_all_targets))

    ############################################################################
    # Compute per-class Average Precision (AP) using all samples (including negatives)
    ############################################################################
    def compute_average_precision(all_probs, all_targets):
        # Convert tensor of probabilities to NumPy array
        all_probs_np = all_probs.cpu().numpy()
        all_targets_np = np.array(all_targets)
        print("all probs np shape ", all_probs_np.shape)
        print("all targets np shape ", all_targets_np.shape)
        C = all_probs_np.shape[1]
        average_ap = 0
        ap_per_class = average_precision_score(all_targets_np, all_probs_np, average=None)
        average_ap = np.mean(ap_per_class)
        class_aps = {c: ap_per_class[c] for c in range(C)}
        print(f"AP for all classes: {class_aps}")
        print("End of per-class AP.\n")

        return average_ap, class_aps

        
    average_ap, class_aps = compute_average_precision(new_all_probs, new_all_targets)
    probs = new_all_probs.cpu().numpy()
    targets_np =  np.array(new_all_targets)

    # Generate confusion matrix heatmap (average probabilities for each combination of true labels)
    target_strings = [''.join(map(str, row.astype(int))) for row in targets_np]

    df = pd.DataFrame({
        'target_combo': target_strings,
        'probs': list(probs)
    })
    grouped = df.groupby('target_combo')['probs']
    avg_probs_combo = grouped.apply(lambda x: np.mean(np.stack(x.values), axis=0)).tolist()
    concurrent_probs_matrix = np.array(avg_probs_combo)

    combo_labels = []
    for combo_str in grouped.groups.keys():
        true_indices = [i for i, char in enumerate(combo_str) if char == '1']
        
        if true_indices:
            label_names = [category_names[i] for i in true_indices]
            combo_labels.append(" & ".join(label_names))
        else:
            combo_labels.append("Negative")

    if concurrent_probs_matrix.size > 0:
        plt.figure(figsize=(12, max(8, len(combo_labels) * 0.5))) # Adjust figure size dynamically for rows
        sns.heatmap(concurrent_probs_matrix, annot=True, fmt=".2f",
                    xticklabels=category_names,
                    yticklabels=pd.Index(combo_labels, name="True Label Combination"), # Use Pandas Index for a label
                    cmap="Blues")
        plt.xlabel("Predicted Class Probability")
        plt.ylabel("True Label Combination")
        plt.title("Multilabel Confusion Heatmap")
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        
        # Save the heatmap
        save_path = f"{output_dir}/internal_confusion_heatmap.png" if internal else f"{output_dir}/confusion_heatmap.png"
        plt.savefig(save_path)
        plt.close() # Close the plot to free memory
        print("Saved confusion heatmap to ", save_path)
    else:
        print("No data to generate the Concurrent Label Confusion Heatmap.")

    # --- Plotting Precision-Recall Curves ---
    all_probs_np = probs
    all_targets_np = np.array(new_all_targets)
    C = all_targets_np.shape[1]
    plt.figure(figsize=(15, 10))
    for i in range(C):
        precision, recall, thresholds = precision_recall_curve(all_targets_np[:, i], all_probs_np[:, i])

        # Plotting the precision-recall curve
        plt.subplot(int(np.ceil(C / 3)), 3, i + 1) # Adjust subplot layout as needed
        plt.plot(recall, precision, label=f'Class {category_names[i]}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {category_names[i]}')
        plt.grid(True)
        plt.legend()

        # Add thresholds to the plot
        for j in range(0, len(thresholds), 10):
            plt.text(recall[j], precision[j], f'{thresholds[j]:.2f}', fontsize=8)


    plt.tight_layout()
    if internal:
        plt.savefig(f"{output_dir}/internal_precision_recall_curves.png")
    else:
        plt.savefig(f"{output_dir}/precision_recall_curves.png")

     # --- Calculate and Print Precision and Recall for a Specific Threshold ---
    print("\n--- Precision and Recall at a Specific Threshold ---")
    ACTIVITY_THRESHOLDS = {
        0: 0.7,
        1: 0.7,
        2: 0.7
    }

    for i in range(C - 1):
        # Convert probabilities to binary predictions based on the threshold
        binary_predictions = (all_probs_np[:, i] >= ACTIVITY_THRESHOLDS[i]).astype(int)

        # Calculate precision and recall for the current class
        # Handle cases where there are no true positives or predicted positives to avoid warnings/errors
        try:
            class_precision = precision_score(all_targets_np[:, i], binary_predictions, zero_division=0)
        except ValueError: # Catches cases where there are no positive samples in true or pred
            class_precision = 0.0 # Or np.nan, depending on desired behavior

        try:
            class_recall = recall_score(all_targets_np[:, i], binary_predictions, zero_division=0)
        except ValueError:
            class_recall = 0.0

        print(f"Class '{category_names[i]}': Precision = {class_precision:.4f}, Recall = {class_recall:.4f}, Threshold = {ACTIVITY_THRESHOLDS[i]}")

    safe_barrier()
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return average_ap, class_aps

############################################################################
# MERGE FUNCTION
############################################################################
def merge(eval_path, num_tasks):
    # not used
    return None, None

############################################################################
# End of file
############################################################################
