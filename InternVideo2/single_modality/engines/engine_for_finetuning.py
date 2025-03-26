# import os
# import time
# import numpy as np
# import math
# import sys
# from typing import Iterable, Optional
# import torch
# from datasets.mixup import Mixup
# from timm.utils import accuracy, ModelEma
# import utils
# from scipy.special import softmax
# import json
# from collections import defaultdict

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

# def train_class_batch(model, samples, target, criterion):
#     outputs = model(samples)
#     print(f"outputs: {outputs}")
#     loss = criterion(outputs, target)
#     return loss, outputs

# def get_loss_scale_for_deepspeed(model):
#     optimizer = model.optimizer
#     return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

# def train_one_epoch(
#         model: torch.nn.Module, criterion: torch.nn.Module,
#         data_loader: Iterable, optimizer: torch.optim.Optimizer,
#         device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
#         model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
#         start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
#         num_training_steps_per_epoch=None, update_freq=None,
#         bf16=False,
#     ):
#     model.train(True)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 1

#     if loss_scaler is None:
#         model.zero_grad()
#         model.micro_steps = 0
#     else:
#         optimizer.zero_grad()

#     for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         step = data_iter_step // update_freq
#         if step >= num_training_steps_per_epoch:
#             continue
#         it = start_steps + step  # global training iteration
#         # Update LR & WD for the current accumulation step
#         if (lr_schedule_values is not None or wd_schedule_values is not None) and data_iter_step % update_freq == 0:
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     lr_scale = param_group.get("lr_scale", 1.0)
#                     param_group["lr"] = lr_schedule_values[it] * lr_scale
#                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                     param_group["weight_decay"] = wd_schedule_values[it]

#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)

#         if loss_scaler is None:
#             samples = samples.bfloat16() if bf16 else samples.half()
#             loss, output = train_class_batch(model, samples, targets, criterion)
#         else:
#             with torch.cuda.amp.autocast():
#                 loss, output = train_class_batch(model, samples, targets, criterion)

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)

#         if loss_scaler is None:
#             loss /= update_freq
#             model.backward(loss)
#             model.step()
#             if (data_iter_step + 1) % update_freq == 0:
#                 if model_ema is not None:
#                     model_ema.update(model)
#             grad_norm = None
#             loss_scale_value = get_loss_scale_for_deepspeed(model)
#         else:
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss /= update_freq
#             grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
#                                     parameters=model.parameters(), create_graph=is_second_order,
#                                     update_grad=(data_iter_step + 1) % update_freq == 0)
#             if (data_iter_step + 1) % update_freq == 0:
#                 optimizer.zero_grad()
#                 if model_ema is not None:
#                     model_ema.update(model)
#             loss_scale_value = loss_scaler.state_dict()["scale"]

#         torch.cuda.synchronize()

#         if mixup_fn is None:
#             class_acc = (output.max(-1)[-1] == targets).float().mean()
#         else:
#             class_acc = None
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(class_acc=class_acc)
#         metric_logger.update(loss_scale=loss_scale_value)
#         min_lr = 10.
#         max_lr = 0.
#         for group in optimizer.param_groups:
#             min_lr = min(min_lr, group["lr"])
#             max_lr = max(max_lr, group["lr"])
#         metric_logger.update(lr=max_lr)
#         metric_logger.update(min_lr=min_lr)
#         weight_decay_value = None
#         for group in optimizer.param_groups:
#             if group["weight_decay"] > 0:
#                 weight_decay_value = group["weight_decay"]
#         metric_logger.update(weight_decay=weight_decay_value)
#         metric_logger.update(grad_norm=grad_norm)

#         if log_writer is not None:
#             log_writer.update(loss=loss_value, head="loss")
#             log_writer.update(class_acc=class_acc, head="loss")
#             log_writer.update(loss_scale=loss_scale_value, head="opt")
#             log_writer.update(lr=max_lr, head="opt")
#             log_writer.update(min_lr=min_lr, head="opt")
#             log_writer.update(weight_decay=weight_decay_value, head="opt")
#             log_writer.update(grad_norm=grad_norm, head="opt")
#             log_writer.set_step()

#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def validation_one_epoch(data_loader, model, device, ds=False, bf16=False, output_dir=None):
#     """
#     Evaluates the model on the validation set, computing overall as well as per-category
#     top-1 and top-5 accuracies. Additionally, for each video in validation, logs the video identifier and 
#     its top-5 predicted categories (with percentages) to a text file, and generates a confusion matrix.
#     """
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import confusion_matrix

#     criterion = torch.nn.CrossEntropyLoss()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Val:'
#     model.eval()

#     # Lists for overall metrics and visualization
#     all_top1 = []
#     all_top5 = []  # Top-5 predictions (list of indices) per video
#     all_targets = []
#     all_video_ids = []
#     all_top5_scores = []  # Softmax percentages for top-5 predictions

#     # Define category names mapping (based on your desired ordering)
#     category_names = {0: "active break-in", 1: "assault", 2: "climbing over fence/gate/wall",
#                       3: "actively taking objects", 4: "running or showing urgency", 5: "fall_floor"}

#     for batch in metric_logger.log_every(data_loader, 10, header):
#         videos = batch[0]
#         target = batch[1]
#         # In validation mode, assume batch[2] contains video identifiers (or paths)
#         if len(batch) >= 3:
#             video_ids = batch[2]
#         else:
#             video_ids = ["unknown"] * videos.shape[0]

#         videos = videos.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         with torch.cuda.amp.autocast():
#             output = model(videos)
#             loss = criterion(output, target)
#         probs = torch.softmax(output, dim=-1)
#         top1_preds = output.argmax(dim=-1)
#         top5_scores, top5_preds = torch.topk(probs, 5, dim=-1)
        
#         all_top1.extend(top1_preds.cpu().tolist())
#         all_targets.extend(target.cpu().tolist())
#         all_video_ids.extend(video_ids)
#         all_top5_scores.extend(top5_scores.cpu().tolist())
#         all_top5.extend(top5_preds.cpu().tolist())

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         batch_size = videos.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

#     # Save per-video predictions to a text file if output_dir is provided
#     details_content = None
#     if output_dir is not None and utils.is_main_process():
#         details_path = os.path.join(output_dir, "validation_details.txt")
#         with open(details_path, "w") as f:
#             for vid, t5_preds, t5_scores in zip(all_video_ids, all_top5, all_top5_scores):
#                 pred_info = []
#                 for pred, score in zip(t5_preds, t5_scores):
#                     pred_info.append(f"{category_names.get(pred, str(pred))} ({score*100:.1f}%)")
#                 f.write(f"Video: {vid} | Top-5: {', '.join(pred_info)}\n")
#         print(f"Saved video prediction details to {details_path}")

#         # Now, read and print out the file's content
#         with open(details_path, "r") as f:
#             details_content = f.read()
#         print("Validation Details:")
#         print(details_content)

#     # Generate confusion matrix and save as an image if output_dir is provided
#     cm = confusion_matrix(all_targets, all_top1)
#     plt.figure(figsize=(10, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix")
#     plt.colorbar()
#     tick_marks = np.arange(len(category_names))
#     plt.xticks(tick_marks, [category_names[i] for i in tick_marks], rotation=45, ha="right")
#     plt.yticks(tick_marks, [category_names[i] for i in tick_marks])
#     plt.ylabel("True Label")
#     plt.xlabel("Predicted Label")
#     plt.tight_layout()
#     if output_dir is not None:
#         cm_path = os.path.join(output_dir, "confusion_matrix.png")
#         plt.savefig(cm_path)
#         print(f"Saved confusion matrix to {cm_path}")
#     plt.close()

#     # Compute per-category stats (top-1 and top-5) as before
#     per_cat_stats = defaultdict(lambda: {"correct_top1": 0, "correct_top5": 0, "count": 0})
#     for true, p1, t5 in zip(all_targets, all_top1, all_top5):
#         per_cat_stats[true]["count"] += 1
#         if p1 == true:
#             per_cat_stats[true]["correct_top1"] += 1
#         if true in t5:
#             per_cat_stats[true]["correct_top5"] += 1

#     print("Per-category accuracies:")
#     for cat in sorted(per_cat_stats.keys()):
#         stats = per_cat_stats[cat]
#         cat_acc1 = stats["correct_top1"] / stats["count"] if stats["count"] > 0 else 0
#         cat_acc5 = stats["correct_top5"] / stats["count"] if stats["count"] > 0 else 0
#         print(f"  Category {cat} ({category_names.get(cat, str(cat))}): Top-1: {cat_acc1*100:.2f}%, Top-5: {cat_acc5*100:.2f}% (n={stats['count']})")

#     # Return overall stats along with the per-video prediction details
#     stats_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     stats_dict["validation_details"] = details_content  # This may be None if output_dir wasn't provided
#     return stats_dict


# @torch.no_grad()
# def final_test(data_loader, model, device, file, ds=False, bf16=False):
#     criterion = torch.nn.CrossEntropyLoss()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#     model.eval()
#     final_result = []
    
#     for batch in metric_logger.log_every(data_loader, 10, header):
#         videos = batch[0]
#         target = batch[1]
#         ids = batch[2]
#         chunk_nb = batch[3]
#         split_nb = batch[4]
#         videos = videos.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         if ds:
#             videos = videos.bfloat16() if bf16 else videos.half()
#             output = model(videos)
#             loss = criterion(output, target)
#         else:
#             with torch.cuda.amp.autocast():
#                 output = model(videos)
#                 loss = criterion(output, target)

#         for i in range(output.size(0)):
#             line = "{} {} {} {} {}\n".format(
#                 ids[i],
#                 str(output.data[i].float().cpu().numpy().tolist()),
#                 str(int(target[i].cpu().numpy())),
#                 str(int(chunk_nb[i].cpu().numpy())),
#                 str(int(split_nb[i].cpu().numpy()))
#             )
#             final_result.append(line)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         batch_size = videos.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

#     with open(file, 'w') as f:
#         f.write("{}, {}\n".format(acc1, acc5))
#         for line in final_result:
#             f.write(line)
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# def merge(eval_path, num_tasks):
#     """
#     Reads prediction files (named "0.txt", "1.txt", …) from eval_path and computes:
#       - Overall top-1 and top-5 accuracy.
#       - Per-category top-1 and top-5 accuracy.
#     """
#     overall_correct_top1 = 0
#     overall_correct_top5 = 0
#     overall_count = 0
#     per_cat_stats = defaultdict(lambda: {"correct_top1": 0, "correct_top5": 0, "count": 0})
    
#     print("Reading individual output files...")
#     for i in range(num_tasks):
#         pred_file = os.path.join(eval_path, f"{i}.txt")
#         if not os.path.exists(pred_file):
#             continue
#         with open(pred_file, "r") as f:
#             lines = f.readlines()
#         # Skip header line
#         for line in lines[1:]:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue
#             try:
#                 probs = json.loads(parts[1])
#             except Exception:
#                 continue
#             true_label = int(parts[2])
#             sorted_idx = sorted(range(len(probs)), key=lambda j: probs[j], reverse=True)
#             top1 = sorted_idx[0]
#             top5 = sorted_idx[:5]
#             overall_count += 1
#             if top1 == true_label:
#                 overall_correct_top1 += 1
#             if true_label in top5:
#                 overall_correct_top5 += 1
#             per_cat_stats[true_label]["count"] += 1
#             if top1 == true_label:
#                 per_cat_stats[true_label]["correct_top1"] += 1
#             if true_label in top5:
#                 per_cat_stats[true_label]["correct_top5"] += 1

#     overall_top1 = overall_correct_top1 / overall_count if overall_count > 0 else 0
#     overall_top5 = overall_correct_top5 / overall_count if overall_count > 0 else 0

#     print("Overall Accuracy: Top-1: {:.2f}%, Top-5: {:.2f}%".format(overall_top1 * 100, overall_top5 * 100))
#     print("Per-category accuracies:")
#     for cat in sorted(per_cat_stats.keys()):
#         stats = per_cat_stats[cat]
#         cat_acc1 = stats["correct_top1"] / stats["count"] if stats["count"] > 0 else 0
#         cat_acc5 = stats["correct_top5"] / stats["count"] if stats["count"] > 0 else 0
#         print(f"  Category {cat}: Top-1: {cat_acc1 * 100:.2f}%, Top-5: {cat_acc5 * 100:.2f}%")
#     return overall_top1, overall_top5


# def compute_video(lst):
#     i, video_id, data, label = lst
#     feat = np.mean(data, axis=0)
#     pred = np.argmax(feat)
#     top1 = 1.0 if int(pred) == int(label) else 0.0
#     top5 = 1.0 if int(label) in np.argsort(-feat)[:5] else 0.0
#     return [pred, top1, top5, int(label)]


######## new code for testing precision and recall
# import os
# import time
# import numpy as np
# import math
# import sys
# from typing import Iterable, Optional
# import torch
# from datasets.mixup import Mixup
# from timm.utils import accuracy, ModelEma
# import utils
# from scipy.special import softmax
# import json
# from collections import defaultdict
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

# def train_class_batch(model, samples, target, criterion):
#     outputs = model(samples)
#     print(f"outputs: {outputs}")
#     loss = criterion(outputs, target)
#     return loss, outputs

# def get_loss_scale_for_deepspeed(model):
#     optimizer = model.optimizer
#     return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
#                     optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
#                     loss_scaler, max_norm: float = 0, model_ema: Optional[ModelEma] = None,
#                     mixup_fn: Optional[Mixup] = None, log_writer=None, start_steps=None,
#                     lr_schedule_values=None, wd_schedule_values=None,
#                     num_training_steps_per_epoch=None, update_freq=None, bf16=False):
#     model.train(True)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(fmt='{value:.6f}'))
#     metric_logger.add_meter('min_lr', utils.SmoothedValue(fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 1
#     if loss_scaler is None:
#         model.zero_grad()
#         model.micro_steps = 0
#     else:
#         optimizer.zero_grad()
#     for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         step = data_iter_step // update_freq
#         if step >= num_training_steps_per_epoch:
#             continue
#         it = start_steps + step
#         if (lr_schedule_values is not None or wd_schedule_values is not None) and data_iter_step % update_freq == 0:
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     lr_scale = param_group.get("lr_scale", 1.0)
#                     param_group["lr"] = lr_schedule_values[it] * lr_scale
#                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                     param_group["weight_decay"] = wd_schedule_values[it]
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)
#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
#         if loss_scaler is None:
#             samples = samples.bfloat16() if bf16 else samples.half()
#             loss, output = train_class_batch(model, samples, targets, criterion)
#         else:
#             with torch.amp.autocast(device_type='cuda'):
#                 loss, output = train_class_batch(model, samples, targets, criterion)
#         loss_value = loss.item()
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
#         if loss_scaler is None:
#             loss /= update_freq
#             model.backward(loss)
#             model.step()
#             if (data_iter_step + 1) % update_freq == 0:
#                 if model_ema is not None:
#                     model_ema.update(model)
#             grad_norm = None
#             loss_scale_value = get_loss_scale_for_deepspeed(model)
#         else:
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss /= update_freq
#             grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
#                                     parameters=model.parameters(), create_graph=is_second_order,
#                                     update_grad=(data_iter_step + 1) % update_freq == 0)
#             if (data_iter_step + 1) % update_freq == 0:
#                 optimizer.zero_grad()
#                 if model_ema is not None:
#                     model_ema.update(model)
#             loss_scale_value = loss_scaler.state_dict()["scale"]
#         torch.cuda.synchronize()
#         if mixup_fn is None:
#             class_acc = (output.max(-1)[-1] == targets).float().mean()
#         else:
#             class_acc = None
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(class_acc=class_acc)
#         metric_logger.update(loss_scale=loss_scale_value)
#         min_lr = 10.
#         max_lr = 0.
#         for group in optimizer.param_groups:
#             min_lr = min(min_lr, group["lr"])
#             max_lr = max(max_lr, group["lr"])
#         metric_logger.update(lr=max_lr)
#         metric_logger.update(min_lr=min_lr)
#         weight_decay_value = None
#         for group in optimizer.param_groups:
#             if group["weight_decay"] > 0:
#                 weight_decay_value = group["weight_decay"]
#         metric_logger.update(weight_decay=weight_decay_value)
#         metric_logger.update(grad_norm=grad_norm)
#         if log_writer is not None:
#             log_writer.update(loss=loss_value, head="loss")
#             log_writer.update(class_acc=class_acc, head="loss")
#             log_writer.update(loss_scale=loss_scale_value, head="opt")
#             log_writer.update(lr=max_lr, head="opt")
#             log_writer.update(min_lr=min_lr, head="opt")
#             log_writer.update(weight_decay=weight_decay_value, head="opt")
#             log_writer.update(grad_norm=grad_norm, head="opt")
#             log_writer.set_step()
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# @torch.no_grad()
# def validation_one_epoch(data_loader, model, device, ds=False, bf16=False, output_dir=None):
#     """
#     Evaluates the model on the validation set.
#     For overall accuracy (top-1, top-5), negative samples (target == -1) are excluded.
#     All samples (including negatives) are used for precision/recall computation.
#     """
#     criterion = torch.nn.CrossEntropyLoss()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Val:'
#     model.eval()
#     all_top1 = []
#     all_top5 = []
#     all_targets = []
#     all_video_ids = []
#     all_top5_scores = []
#     all_probs = []  # For precision/recall calculation
#     category_names = {0: "active break-in", 1: "assault", 2: "climbing over fence/gate/wall",
#                       3: "actively taking objects", 4: "running or showing urgency", 5: "fall_floor"}
#     for batch in metric_logger.log_every(data_loader, 10, header):
#         videos = batch[0]
#         target = batch[1]
#         if len(batch) >= 3:
#             video_ids = batch[2]
#         else:
#             video_ids = ["unknown"] * videos.shape[0]
#         videos = videos.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#         with torch.amp.autocast(device_type='cuda'):
#             output = model(videos)
#         # Compute loss only on valid samples (target != -1)
#         valid_mask = (target != -1)
#         if valid_mask.sum() > 0:
#             loss = criterion(output[valid_mask], target[valid_mask])
#         else:
#             loss = torch.tensor(0.0, device=output.device)
#         probs = torch.softmax(output, dim=-1)
#         top1_preds = output.argmax(dim=-1)
#         top5_scores, top5_preds = torch.topk(probs, 5, dim=-1)
#         all_top1.extend(top1_preds.cpu().tolist())
#         all_targets.extend(target.cpu().tolist())
#         all_video_ids.extend(video_ids)
#         all_top5_scores.extend(top5_scores.cpu().tolist())
#         all_top5.extend(top5_preds.cpu().tolist())
#         all_probs.append(probs.cpu())
#         # For accuracy, use only valid samples:
#         if valid_mask.sum() > 0:
#             valid_outputs = output[valid_mask]
#             valid_targets = target[valid_mask]
#             acc1, acc5 = accuracy(valid_outputs, valid_targets, topk=(1, 5))
#             batch_valid_count = valid_mask.sum().item()
#             metric_logger.meters['acc1'].update(acc1.item(), n=batch_valid_count)
#             metric_logger.meters['acc5'].update(acc5.item(), n=batch_valid_count)
#         metric_logger.update(loss=loss.item())
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
#     details_content = None
#     if output_dir is not None and utils.is_main_process():
#         details_path = os.path.join(output_dir, "validation_details.txt")
#         with open(details_path, "w") as f:
#             for vid, t5_preds, t5_scores in zip(all_video_ids, all_top5, all_top5_scores):
#                 pred_info = []
#                 for pred, score in zip(t5_preds, t5_scores):
#                     pred_info.append(f"{category_names.get(pred, str(pred))} ({score*100:.1f}%)")
#                 f.write(f"Video: {vid} | Top-5: {', '.join(pred_info)}\n")
#         print(f"Saved video prediction details to {details_path}")
#         with open(details_path, "r") as f:
#             details_content = f.read()
#         print("Validation Details:")
#         print(details_content)
#     stats_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     stats_dict["validation_details"] = details_content
#     return stats_dict

# @torch.no_grad()
# def final_test(data_loader, model, device, file, ds=False, bf16=False):
#     """
#     Final evaluation after all epochs.
#     Overall accuracy (top-1, top-5) is computed excluding negative samples (target == -1),
#     while precision/recall are computed using all samples.
#     """
#     criterion = torch.nn.CrossEntropyLoss()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#     model.eval()
#     final_result = []
#     all_top1 = []
#     all_targets = []
#     all_video_ids = []
#     all_top5 = []
#     all_top5_scores = []
#     all_probs = []
#     for batch in metric_logger.log_every(data_loader, 10, header):
#         videos = batch[0]
#         target = batch[1]
#         if len(batch) >= 3:
#             video_ids = batch[2]
#         else:
#             video_ids = ["unknown"] * videos.shape[0]
#         videos = videos.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#         with torch.amp.autocast(device_type='cuda'):
#             output = model(videos)
#         # Compute loss only on valid samples:
#         valid_mask = (target != -1)
#         if valid_mask.sum() > 0:
#             loss = criterion(output[valid_mask], target[valid_mask])
#         else:
#             loss = torch.tensor(0.0, device=output.device)
#         probs = torch.softmax(output, dim=-1)
#         top1_preds = output.argmax(dim=-1)
#         top5_scores, top5_preds = torch.topk(probs, 5, dim=-1)
#         all_top1.extend(top1_preds.cpu().tolist())
#         all_targets.extend(target.cpu().tolist())
#         all_video_ids.extend(video_ids)
#         all_top5_scores.extend(top5_scores.cpu().tolist())
#         all_top5.extend(top5_preds.cpu().tolist())
#         all_probs.append(probs.cpu())
#         for i in range(output.size(0)):
#             line = "{} {} {} {} {}\n".format(
#                 video_ids[i],
#                 str(output.data[i].float().cpu().numpy().tolist()),
#                 str(int(target[i].cpu().numpy())),
#                 "NA", "NA"
#             )
#             final_result.append(line)
#         if valid_mask.sum() > 0:
#             valid_outputs = output[valid_mask]
#             valid_targets = target[valid_mask]
#             acc1, acc5 = accuracy(valid_outputs, valid_targets, topk=(1, 5))
#             batch_valid_count = valid_mask.sum().item()
#             metric_logger.meters['acc1'].update(acc1.item(), n=batch_valid_count)
#             metric_logger.meters['acc5'].update(acc5.item(), n=batch_valid_count)
#         metric_logger.update(loss=loss.item())
#     with open(file, 'w') as f:
#         f.write("{}, {}\n".format(acc1, acc5))
#         for line in final_result:
#             f.write(line)
#     metric_logger.synchronize_between_processes()
#     print('* Test Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
#     all_probs = torch.cat(all_probs, dim=0)  # shape [N, C]
#     category_names = {0: "active break-in", 1: "assault", 2: "climbing over fence/gate/wall",
#                       3: "actively taking objects", 4: "running or showing urgency", 5: "fall_floor"}
#     def compute_precision_recall_with_negatives(all_probs, all_targets, category_names, threshold=0.3):
#         all_probs_np = all_probs.numpy()
#         all_targets_np = np.array(all_targets)
#         N, C = all_probs_np.shape
#         for c in range(C):
#             scores = all_probs_np[:, c]
#             # For precision/recall, include all samples even negatives
#             is_pos = (all_targets_np == c).astype(int)
#             predicted_pos = (scores >= threshold).astype(int)
#             tp = np.sum((predicted_pos == 1) & (is_pos == 1))
#             fp = np.sum((predicted_pos == 1) & (is_pos == 0))
#             fn = np.sum((predicted_pos == 0) & (is_pos == 1))
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#             print(f"[Category {c} - {category_names.get(c, f'Cat {c}')}] Threshold={threshold:.2f}, "
#                   f"TP={tp}, FP={fp}, FN={fn}, Precision={precision*100:.2f}%, Recall={recall*100:.2f}%")
#     print("Precision/Recall on test set (threshold=0.3):")
#     compute_precision_recall_with_negatives(all_probs, all_targets, category_names, threshold=0.1)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# def merge(eval_path, num_tasks):
#     overall_correct_top1 = 0
#     overall_correct_top5 = 0
#     overall_count = 0
#     per_cat_stats = defaultdict(lambda: {"correct_top1": 0, "correct_top5": 0, "count": 0})
#     print("Reading individual output files...")
#     for i in range(num_tasks):
#         pred_file = os.path.join(eval_path, f"{i}.txt")
#         if not os.path.exists(pred_file):
#             continue
#         with open(pred_file, "r") as f:
#             lines = f.readlines()
#         for line in lines[1:]:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue
#             try:
#                 probs = json.loads(parts[1])
#             except Exception:
#                 continue
#             true_label = int(parts[2])
#             sorted_idx = sorted(range(len(probs)), key=lambda j: probs[j], reverse=True)
#             top1 = sorted_idx[0]
#             top5 = sorted_idx[:5]
#             overall_count += 1
#             if top1 == true_label:
#                 overall_correct_top1 += 1
#             if true_label in top5:
#                 overall_correct_top5 += 1
#             per_cat_stats[true_label]["count"] += 1
#             if top1 == true_label:
#                 per_cat_stats[true_label]["correct_top1"] += 1
#             if true_label in top5:
#                 per_cat_stats[true_label]["correct_top5"] += 1
#     overall_top1 = overall_correct_top1 / overall_count if overall_count > 0 else 0
#     overall_top5 = overall_correct_top5 / overall_count if overall_count > 0 else 0
#     print("Overall Accuracy: Top-1: {:.2f}%, Top-5: {:.2f}%".format(overall_top1 * 100, overall_top5 * 100))
#     print("Per-category accuracies:")
#     for cat in sorted(per_cat_stats.keys()):
#         stats = per_cat_stats[cat]
#         cat_acc1 = stats["correct_top1"] / stats["count"] if stats["count"] > 0 else 0
#         cat_acc5 = stats["correct_top5"] / stats["count"] if stats["count"] > 0 else 0
#         print(f"  Category {cat}: Top-1: {cat_acc1 * 100:.2f}%, Top-5: {cat_acc5 * 100:.2f}%")
#     return overall_top1, overall_top5

# def compute_video(lst):
#     i, video_id, data, label = lst
#     feat = np.mean(data, axis=0)
#     pred = np.argmax(feat)
#     top1 = 1.0 if int(pred) == int(label) else 0.0
#     top5 = 1.0 if int(label) in np.argsort(-feat)[:5] else 0.0
#     return [pred, top1, top5, int(label)]


# # ## adding AP
# import os
# import time
# import math
# import sys
# import json
# import numpy as np
# from typing import Iterable, Optional
# from collections import defaultdict

# import torch
# from datasets.mixup import Mixup
# from timm.utils import accuracy, ModelEma
# import utils
# from scipy.special import softmax

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, average_precision_score

# ############################################################################
# # Helper function for safe barrier (for distributed mode)
# ############################################################################
# def safe_barrier():
#     if torch.distributed.is_initialized():
#         torch.distributed.barrier()

# ############################################################################
# # TRAINING FUNCTIONS
# ############################################################################
# def train_class_batch(model, samples, target, criterion):
#     outputs = model(samples)
#     print(f"[DEBUG] Model outputs: {outputs}")  # Debug print
#     loss = criterion(outputs, target)
#     return loss, outputs

# def get_loss_scale_for_deepspeed(model):
#     optimizer = model.optimizer
#     return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

# def train_one_epoch(
#     model: torch.nn.Module,
#     criterion: torch.nn.Module,
#     data_loader: Iterable,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     epoch: int,
#     loss_scaler,
#     max_norm: float = 0,
#     model_ema: Optional[ModelEma] = None,
#     mixup_fn: Optional[Mixup] = None,
#     log_writer=None,
#     start_steps=None,
#     lr_schedule_values=None,
#     wd_schedule_values=None,
#     num_training_steps_per_epoch=None,
#     update_freq=None,
#     bf16=False
# ):
#     model.train(True)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = f"Epoch: [{epoch}]"
#     print_freq = 1

#     if loss_scaler is None:
#         model.zero_grad()
#         model.micro_steps = 0
#     else:
#         optimizer.zero_grad()

#     for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         step = data_iter_step // update_freq
#         if step >= num_training_steps_per_epoch:
#             continue
#         it = start_steps + step  # global training iteration

#         # Update learning rate and weight decay if schedules are provided
#         if (lr_schedule_values is not None or wd_schedule_values is not None) and (data_iter_step % update_freq == 0):
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     lr_scale = param_group.get("lr_scale", 1.0)
#                     param_group["lr"] = lr_schedule_values[it] * lr_scale
#                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                     param_group["weight_decay"] = wd_schedule_values[it]

#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)

#         if loss_scaler is None:
#             samples = samples.bfloat16() if bf16 else samples.half()
#             loss, output = train_class_batch(model, samples, targets, criterion)
#         else:
#             with torch.amp.autocast(device_type='cuda'):
#                 loss, output = train_class_batch(model, samples, targets, criterion)

#         loss_value = loss.item()
#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping training")
#             sys.exit(1)

#         if loss_scaler is None:
#             loss /= update_freq
#             model.backward(loss)
#             model.step()
#             if (data_iter_step + 1) % update_freq == 0 and model_ema is not None:
#                 model_ema.update(model)
#             grad_norm = None
#             loss_scale_value = get_loss_scale_for_deepspeed(model)
#         else:
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss /= update_freq
#             grad_norm = loss_scaler(
#                 loss,
#                 optimizer,
#                 clip_grad=max_norm,
#                 parameters=model.parameters(),
#                 create_graph=is_second_order,
#                 update_grad=((data_iter_step + 1) % update_freq == 0)
#             )
#             if (data_iter_step + 1) % update_freq == 0:
#                 optimizer.zero_grad()
#                 if model_ema is not None:
#                     model_ema.update(model)
#             loss_scale_value = loss_scaler.state_dict()["scale"]

#         torch.cuda.synchronize()

#         if mixup_fn is None:
#             class_acc = (output.argmax(dim=-1) == targets).float().mean()
#         else:
#             class_acc = None

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(class_acc=class_acc)
#         metric_logger.update(loss_scale=loss_scale_value)
#         min_lr, max_lr = 10., 0.
#         for group in optimizer.param_groups:
#             min_lr = min(min_lr, group["lr"])
#             max_lr = max(max_lr, group["lr"])
#         metric_logger.update(lr=max_lr)
#         metric_logger.update(min_lr=min_lr)
#         weight_decay_value = None
#         for group in optimizer.param_groups:
#             if group["weight_decay"] > 0:
#                 weight_decay_value = group["weight_decay"]
#         metric_logger.update(weight_decay=weight_decay_value)
#         metric_logger.update(grad_norm=grad_norm)

#         if log_writer is not None:
#             log_writer.update(loss=loss_value, head="loss")
#             log_writer.update(class_acc=class_acc, head="loss")
#             log_writer.update(loss_scale=loss_scale_value, head="opt")
#             log_writer.update(lr=max_lr, head="opt")
#             log_writer.update(min_lr=min_lr, head="opt")
#             log_writer.update(weight_decay=weight_decay_value, head="opt")
#             log_writer.update(grad_norm=grad_norm, head="opt")
#             log_writer.set_step()

#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ############################################################################
# # VALIDATION FUNCTIONS
# ############################################################################
# @torch.no_grad()
# def validation_one_epoch(data_loader, model, device, ds=False, bf16=False, output_dir=None):
#     """
#     Evaluates the model on the validation set.
#     For overall accuracy (top-1, top-5), negative samples (target == -1) are excluded.
#     All samples (including negatives) are used for precision/recall computation.
#     Additionally, per-category accuracies (as in your original code) are computed.
#     """
#     criterion = torch.nn.CrossEntropyLoss()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Val:"
#     model.eval()

#     all_top1 = []
#     all_top5 = []
#     all_targets = []
#     all_video_ids = []
#     all_top5_scores = []
#     all_probs = []  # For precision/recall and AP calculation

#     category_names = {
#         0: "active break-in",
#         1: "assault",
#         2: "climbing over fence/gate/wall",
#         3: "actively taking objects",
#         4: "running or showing urgency",
#         5: "fall_floor"
#     }

#     for batch in metric_logger.log_every(data_loader, 10, header):
#         videos, target = batch[0], batch[1]
#         video_ids = batch[2] if len(batch) >= 3 else ["unknown"] * videos.shape[0]

#         videos = videos.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         with torch.amp.autocast(device_type='cuda'):
#             output = model(videos)
#             # Compute loss only on valid samples (target != -1)
#             valid_mask = (target != -1)
#             if valid_mask.sum() > 0:
#                 loss = criterion(output[valid_mask], target[valid_mask])
#             else:
#                 loss = torch.tensor(0.0, device=videos.device)

#         probs = torch.softmax(output, dim=-1)
#         top1_preds = output.argmax(dim=-1)
#         top5_scores, top5_preds = torch.topk(probs, 5, dim=-1)

#         all_top1.extend(top1_preds.cpu().tolist())
#         all_targets.extend(target.cpu().tolist())
#         all_video_ids.extend(video_ids)
#         all_top5_scores.extend(top5_scores.cpu().tolist())
#         all_top5.extend(top5_preds.cpu().tolist())
#         all_probs.append(probs.cpu())

#         # For accuracy, update using only valid samples
#         if valid_mask.sum() > 0:
#             valid_outputs = output[valid_mask]
#             valid_targets = target[valid_mask]
#             acc1, acc5 = accuracy(valid_outputs, valid_targets, topk=(1, 5))
#             batch_valid_count = valid_mask.sum().item()
#             metric_logger.meters['acc1'].update(acc1.item(), n=batch_valid_count)
#             metric_logger.meters['acc5'].update(acc5.item(), n=batch_valid_count)
#         metric_logger.update(loss=loss.item())

#     metric_logger.synchronize_between_processes()
#     print("* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}"
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

#     # Save per-video prediction details
#     details_content = None
#     if output_dir is not None and utils.is_main_process():
#         details_path = os.path.join(output_dir, "validation_details.txt")
#         with open(details_path, "w") as f:
#             for vid, t5_preds, t5_scores in zip(all_video_ids, all_top5, all_top5_scores):
#                 pred_info = [f"{category_names.get(pred, str(pred))} ({score*100:.1f}%)"
#                              for pred, score in zip(t5_preds, t5_scores)]
#                 f.write(f"Video: {vid} | Top-5: {', '.join(pred_info)}\n")
#         print(f"[DEBUG] Saved video prediction details to {details_path}")
#         with open(details_path, "r") as f:
#             details_content = f.read()
#         print("Validation Details:")
#         print(details_content)

#     # Generate confusion matrix using only valid (non -1) samples
#     valid_idx = [i for i, lbl in enumerate(all_targets) if lbl != -1]
#     valid_preds = [all_top1[i] for i in valid_idx]
#     valid_labels = [all_targets[i] for i in valid_idx]
#     cm = confusion_matrix(valid_labels, valid_preds)
#     plt.figure(figsize=(10, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix - Validation")
#     plt.colorbar()
#     tick_marks = np.arange(len(category_names))
#     plt.xticks(tick_marks, [category_names[i] for i in tick_marks], rotation=45, ha="right")
#     plt.yticks(tick_marks, [category_names[i] for i in tick_marks])
#     plt.ylabel("True Label")
#     plt.xlabel("Predicted Label")
#     plt.tight_layout()
#     if output_dir is not None and utils.is_main_process():
#         cm_path = os.path.join(output_dir, "confusion_matrix.png")
#         plt.savefig(cm_path)
#         print(f"[DEBUG] Saved confusion matrix to {cm_path}")
#     plt.close()

#     # Compute per-category accuracies (ignoring negatives)
#     per_cat_stats = defaultdict(lambda: {"correct_top1": 0, "correct_top5": 0, "count": 0})
#     for true, p1, t5 in zip(all_targets, all_top1, all_top5):
#         if true == -1:
#             continue
#         per_cat_stats[true]["count"] += 1
#         if p1 == true:
#             per_cat_stats[true]["correct_top1"] += 1
#         if true in t5:
#             per_cat_stats[true]["correct_top5"] += 1

#     print("Per-category accuracies (ignoring negatives):")
#     for cat in sorted(per_cat_stats.keys()):
#         stats = per_cat_stats[cat]
#         cat_acc1 = stats["correct_top1"] / stats["count"] if stats["count"] > 0 else 0
#         cat_acc5 = stats["correct_top5"] / stats["count"] if stats["count"] > 0 else 0
#         print(f"  Category {cat} ({category_names.get(cat, str(cat))}): Top-1: {cat_acc1*100:.2f}%, Top-5: {cat_acc5*100:.2f}% (n={stats['count']})")

#     stats_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     stats_dict["validation_details"] = details_content
#     return stats_dict

# ############################################################################
# # FINAL TEST FUNCTIONS
# ############################################################################
# @torch.no_grad()
# def final_test(data_loader, model, device, file, ds=False, bf16=False):
#     """
#     Final evaluation after all epochs.
#       - Overall accuracy (top-1 and top-5) is computed ignoring negative samples (target == -1)
#       - Precision/Recall (and Average Precision) are computed using all samples
#       - Per-sample predictions are saved for merging, and a confusion matrix (ignoring negatives) is saved.
#     """
#     criterion = torch.nn.CrossEntropyLoss()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"
#     model.eval()

#     final_result = []
#     all_top1 = []
#     all_targets = []
#     all_video_ids = []
#     all_top5 = []
#     all_top5_scores = []
#     all_probs = []

#     # Category mapping for 6 classes
#     category_names = {
#         0: "active break-in",
#         1: "assault",
#         2: "climbing over fence/gate/wall",
#         3: "actively taking objects",
#         4: "running or showing urgency",
#         5: "fall_floor"
#     }

#     for batch in metric_logger.log_every(data_loader, 10, header):
#         videos, target = batch[0], batch[1]
#         video_ids = batch[2] if len(batch) >= 3 else ["unknown"] * videos.shape[0]

#         videos = videos.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         with torch.amp.autocast(device_type='cuda'):
#             output = model(videos)
#         # Compute loss only on valid samples
#         valid_mask = (target != -1)
#         if valid_mask.sum() > 0:
#             loss = criterion(output[valid_mask], target[valid_mask])
#         else:
#             loss = torch.tensor(0.0, device=output.device)

#         probs = torch.softmax(output, dim=-1)
#         top1_preds = output.argmax(dim=-1)
#         top5_scores, top5_preds = torch.topk(probs, 5, dim=-1)

#         all_top1.extend(top1_preds.cpu().tolist())
#         all_targets.extend(target.cpu().tolist())
#         all_video_ids.extend(video_ids)
#         all_top5_scores.extend(top5_scores.cpu().tolist())
#         all_top5.extend(top5_preds.cpu().tolist())
#         all_probs.append(probs.cpu())

#         # Save per-sample predictions for merge (JSON-serialized)
#         for i in range(output.size(0)):
#             prob_json = json.dumps(output.data[i].float().cpu().numpy().tolist())
#             line = f"{video_ids[i]} {prob_json} {int(target[i].cpu().numpy())} NA NA\n"
#             final_result.append(line)

#         if valid_mask.sum() > 0:
#             valid_outputs = output[valid_mask]
#             valid_targets = target[valid_mask]
#             acc1, acc5 = accuracy(valid_outputs, valid_targets, topk=(1, 5))
#             batch_valid_count = valid_mask.sum().item()
#             metric_logger.meters['acc1'].update(acc1.item(), n=batch_valid_count)
#             metric_logger.meters['acc5'].update(acc5.item(), n=batch_valid_count)
#         metric_logger.update(loss=loss.item())

#     with open(file, "w") as f:
#         f.write("video_id probabilities true_label chunk_nb split_nb\n")
#         for line in final_result:
#             f.write(line)

#     metric_logger.synchronize_between_processes()
#     print("* Test Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}"
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
#     # Combine probabilities from all batches
#     all_probs = torch.cat(all_probs, dim=0)  # shape [N, C]

#     # --- Compute Precision and Recall (using all samples, even negatives) ---
#     def compute_precision_recall_with_negatives(all_probs, all_targets, category_names, threshold=0.0001):
#         all_probs_np = all_probs.numpy()
#         all_targets_np = np.array(all_targets)
#         N, C = all_probs_np.shape
#         for c in range(C):
#             scores = all_probs_np[:, c]
#             is_pos = (all_targets_np == c).astype(int)
#             predicted_pos = (scores >= threshold).astype(int)
#             tp = np.sum((predicted_pos == 1) & (is_pos == 1))
#             fp = np.sum((predicted_pos == 1) & (is_pos == 0))
#             fn = np.sum((predicted_pos == 0) & (is_pos == 1))
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#             print(f"[Category {c} - {category_names.get(c, f'Cat {c}')}] Threshold={threshold:.2f}, "
#                   f"TP={tp}, FP={fp}, FN={fn}, Precision={precision*100:.2f}%, Recall={recall*100:.2f}%")
    
#     print("Precision/Recall on test set (threshold=0.5):")
#     compute_precision_recall_with_negatives(all_probs, all_targets, category_names, threshold=0.0001)

#     # --- Compute per-class Average Precision (AP)---
#     def compute_average_precision(all_probs, all_targets, category_names):
#         all_probs_cat = all_probs  # already a tensor of shape [N, C]
#         all_targets_np = np.array(all_targets)
#         # valid_mask = (all_targets_np != -1)
#         if not np.any(valid_mask):
#             print("No valid samples for AP computation.")
#             return
#         all_probs_np = all_probs_cat.cpu().numpy()[valid_mask]
#         all_targets_np = all_targets_np[valid_mask]
#         C = all_probs_np.shape[1]
#         print("Per-class Average Precision (AP):")
#         for c in range(C):
#             is_pos = (all_targets_np == c).astype(int)
#             if np.sum(is_pos) == 0:
#                 ap = 0.0
#             else:
#                 ## zeroing it out before putting into the library 
#                 scores = all_probs_np[:, c]
#                 ap = average_precision_score(is_pos, scores)
#             print(f"  AP for class {c} ({category_names.get(c, f'Cat {c}')}) : {ap:.4f}")
#         print("End of per-class AP.\n")
    
#     compute_average_precision(all_probs, all_targets, category_names)

#     # --- Save confusion matrix using only valid (non -1) samples ---
#     valid_idx = [i for i, lbl in enumerate(all_targets) if lbl != -1]
#     valid_preds = [all_top1[i] for i in valid_idx]
#     valid_labels = [all_targets[i] for i in valid_idx]
#     cm = confusion_matrix(valid_labels, valid_preds)
#     plt.figure(figsize=(10, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix - Final Test")
#     plt.colorbar()
#     tick_marks = np.arange(len(category_names))
#     plt.xticks(tick_marks, [category_names[i] for i in tick_marks], rotation=45, ha="right")
#     plt.yticks(tick_marks, [category_names[i] for i in tick_marks])
#     plt.ylabel("True Label")
#     plt.xlabel("Predicted Label")
#     plt.tight_layout()
#     cm_path = os.path.join(os.path.dirname(file), "test_confusion_matrix.png")
#     plt.savefig(cm_path)
#     print(f"[DEBUG] Saved final test confusion matrix to {cm_path}")
#     plt.close()

#     safe_barrier()
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ############################################################################
# # MERGE FUNCTION
# ############################################################################
# def merge(eval_path, num_tasks):
#     """
#     Reads prediction files (named "0.txt", "1.txt", …) from eval_path and computes:
#       - Overall top-1 and top-5 accuracy.
#       - Per-category top-1 and top-5 accuracy.
#     Debug statements are added to verify parsed lines.
#     """
#     overall_correct_top1 = 0
#     overall_correct_top5 = 0
#     overall_count = 0
#     per_cat_stats = defaultdict(lambda: {"correct_top1": 0, "correct_top5": 0, "count": 0})
    
#     print("[DEBUG] Reading individual output files for merging...")
#     for i in range(num_tasks):
#         pred_file = os.path.join(eval_path, f"{i}.txt")
#         if not os.path.exists(pred_file):
#             print(f"[DEBUG] File {pred_file} not found, skipping.")
#             continue
#         with open(pred_file, "r") as f:
#             lines = f.readlines()
#         if len(lines) == 0:
#             continue
#         # Skip header line
#         for line in lines[1:]:
#             parts = line.strip().split()
#             if len(parts) < 3:
#                 print(f"[DEBUG] Line skipped due to insufficient parts: {line}")
#                 continue
#             try:
#                 probs = json.loads(parts[1])
#             except Exception as e:
#                 print(f"[DEBUG] JSON parse failed for line: {line} with error: {e}")
#                 continue
#             try:
#                 true_label = int(parts[2])
#             except Exception as e:
#                 print(f"[DEBUG] True label parse failed for line: {line} with error: {e}")
#                 continue
#             sorted_idx = sorted(range(len(probs)), key=lambda j: probs[j], reverse=True)
#             top1 = sorted_idx[0]
#             top5 = sorted_idx[:5]
#             overall_count += 1
#             if top1 == true_label:
#                 overall_correct_top1 += 1
#             if true_label in top5:
#                 overall_correct_top5 += 1
#             per_cat_stats[true_label]["count"] += 1
#             if top1 == true_label:
#                 per_cat_stats[true_label]["correct_top1"] += 1
#             if true_label in top5:
#                 per_cat_stats[true_label]["correct_top5"] += 1

#     overall_top1 = overall_correct_top1 / overall_count if overall_count > 0 else 0
#     overall_top5 = overall_correct_top5 / overall_count if overall_count > 0 else 0

#     print(f"[DEBUG] Overall Merged Accuracy: Top-1: {overall_top1*100:.2f}%, Top-5: {overall_top5*100:.2f}%")
#     print("Per-category accuracies (merged):")
#     for cat in sorted(per_cat_stats.keys()):
#         stats = per_cat_stats[cat]
#         cat_acc1 = stats["correct_top1"] / stats["count"] if stats["count"] > 0 else 0
#         cat_acc5 = stats["correct_top5"] / stats["count"] if stats["count"] > 0 else 0
#         print(f"  Category {cat}: Top-1: {cat_acc1*100:.2f}%, Top-5: {cat_acc5*100:.2f}%")
#     return overall_top1, overall_top5

# def compute_video(lst):
#     i, video_id, data, label = lst
#     feat = np.mean(data, axis=0)
#     pred = np.argmax(feat)
#     top1 = 1.0 if int(pred) == int(label) else 0.0
#     top5 = 1.0 if int(label) in np.argsort(-feat)[:5] else 0.0
#     return [pred, top1, top5, int(label)]

# ############################################################################
# # End of file
# ############################################################################


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

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score

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
    print(f"[DEBUG] Model outputs: {outputs}")
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
    bf16=False
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

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
            loss, output = train_class_batch(model, samples, targets, criterion)
        else:
            with torch.amp.autocast(device_type='cuda'):
                loss, output = train_class_batch(model, samples, targets, criterion)

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

        if mixup_fn is None:
            class_acc = (output.argmax(dim=-1) == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
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
            log_writer.update(class_acc=class_acc, head="loss")
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
    """
    Evaluates the model on the validation set.
    For overall accuracy (top-1, top-5), negative samples (target == -1) are excluded.
    All samples (including negatives) are used for precision/recall computation.
    Additionally, per-category accuracies (as in your original code) are computed.
    """
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"
    model.eval()

    all_top1 = []
    all_top5 = []
    all_targets = []
    all_video_ids = []
    all_top5_scores = []
    all_probs = []  # For precision/recall and AP calculation

    category_names = {
        0: "active break-in",
        1: "assault",
        2: "climbing over fence/gate/wall",
        3: "actively taking objects",
        4: "running or showing urgency",
        5: "fall_floor"
    }

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos, target = batch[0], batch[1]
        video_ids = batch[2] if len(batch) >= 3 else ["unknown"] * videos.shape[0]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            output = model(videos)
            # Compute loss only on valid samples (target != -1)
            valid_mask = (target != -1)
            if valid_mask.sum() > 0:
                loss = criterion(output[valid_mask], target[valid_mask])
            else:
                loss = torch.tensor(0.0, device=videos.device)

        probs = torch.softmax(output, dim=-1)
        top1_preds = output.argmax(dim=-1)
        top5_scores, top5_preds = torch.topk(probs, 5, dim=-1)

        all_top1.extend(top1_preds.cpu().tolist())
        all_targets.extend(target.cpu().tolist())
        all_video_ids.extend(video_ids)
        all_top5_scores.extend(top5_scores.cpu().tolist())
        all_top5.extend(top5_preds.cpu().tolist())
        all_probs.append(probs.cpu())

        # For accuracy, update using only valid samples:
        if valid_mask.sum() > 0:
            valid_outputs = output[valid_mask]
            valid_targets = target[valid_mask]
            acc1, acc5 = accuracy(valid_outputs, valid_targets, topk=(1, 5))
            batch_valid_count = valid_mask.sum().item()
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_valid_count)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_valid_count)
        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print("* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}"
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    details_content = None
    if output_dir is not None and utils.is_main_process():
        details_path = os.path.join(output_dir, "validation_details.txt")
        with open(details_path, "w") as f:
            for vid, t5_preds, t5_scores in zip(all_video_ids, all_top5, all_top5_scores):
                pred_info = [f"{category_names.get(pred, str(pred))} ({score*100:.1f}%)"
                             for pred, score in zip(t5_preds, t5_scores)]
                f.write(f"Video: {vid} | Top-5: {', '.join(pred_info)}\n")
        print(f"[DEBUG] Saved video prediction details to {details_path}")
        with open(details_path, "r") as f:
            details_content = f.read()
        # print("Validation Details:")
        # print(details_content)

    # Generate confusion matrix using only valid (non -1) samples
    valid_idx = [i for i, lbl in enumerate(all_targets) if lbl != -1]
    valid_preds = [all_top1[i] for i in valid_idx]
    valid_labels = [all_targets[i] for i in valid_idx]
    cm = confusion_matrix(valid_labels, valid_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Validation")
    plt.colorbar()
    tick_marks = np.arange(len(category_names))
    plt.xticks(tick_marks, [category_names[i] for i in tick_marks], rotation=45, ha="right")
    plt.yticks(tick_marks, [category_names[i] for i in tick_marks])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if output_dir is not None and utils.is_main_process():
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"[DEBUG] Saved confusion matrix to {cm_path}")
    plt.close()

    # Compute per-category accuracies (ignoring negatives)
    per_cat_stats = defaultdict(lambda: {"correct_top1": 0, "correct_top5": 0, "count": 0})
    for true, p1, t5 in zip(all_targets, all_top1, all_top5):
        if true == -1:
            continue
        per_cat_stats[true]["count"] += 1
        if p1 == true:
            per_cat_stats[true]["correct_top1"] += 1
        if true in t5:
            per_cat_stats[true]["correct_top5"] += 1

    print("Per-category accuracies (ignoring negatives):")
    for cat in sorted(per_cat_stats.keys()):
        stats = per_cat_stats[cat]
        cat_acc1 = stats["correct_top1"] / stats["count"] if stats["count"] > 0 else 0
        cat_acc5 = stats["correct_top5"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  Category {cat} ({category_names.get(cat, str(cat))}): Top-1: {cat_acc1*100:.2f}%, Top-5: {cat_acc5*100:.2f}% (n={stats['count']})")

    stats_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats_dict["validation_details"] = details_content
    return stats_dict

############################################################################
# FINAL TEST FUNCTIONS
############################################################################
@torch.no_grad()
def final_test(data_loader, model, device, file, ds=False, bf16=False):
    """
    Final evaluation after all epochs.
    Overall accuracy (top-1 and top-5) is computed ignoring negative samples (target == -1),
    while precision/recall and Average Precision (AP) are computed using all samples.
    Per-sample predictions are saved and a confusion matrix (ignoring negatives) is generated.
    """
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    final_result = []
    all_top1 = []
    all_targets = []
    all_video_ids = []
    all_top5 = []
    all_top5_scores = []
    all_probs = []

    # Category mapping for 6 classes
    category_names = {
        0: "active break-in",
        1: "assault",
        2: "climbing over fence/gate/wall",
        3: "actively taking objects",
        4: "running or showing urgency",
        5: "fall_floor"
    }

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos, target = batch[0], batch[1]
        video_ids = batch[2] if len(batch) >= 3 else ["unknown"] * videos.shape[0]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            output = model(videos)
        # Compute loss only on valid samples:
        valid_mask = (target != -1)
        if valid_mask.sum() > 0:
            loss = criterion(output[valid_mask], target[valid_mask])
        else:
            loss = torch.tensor(0.0, device=output.device)
        probs = torch.softmax(output, dim=-1)
        top1_preds = output.argmax(dim=-1)
        top5_scores, top5_preds = torch.topk(probs, 5, dim=-1)

        all_top1.extend(top1_preds.cpu().tolist())
        all_targets.extend(target.cpu().tolist())
        all_video_ids.extend(video_ids)
        all_top5_scores.extend(top5_scores.cpu().tolist())
        all_top5.extend(top5_preds.cpu().tolist())
        all_probs.append(probs.cpu())

        # Save per-sample predictions (JSON-serialized) for merging
        for i in range(output.size(0)):
            prob_json = json.dumps(output.data[i].float().cpu().numpy().tolist())
            line = f"{video_ids[i]} {prob_json} {int(target[i].cpu().numpy())} NA NA\n"
            final_result.append(line)

        if valid_mask.sum() > 0:
            valid_outputs = output[valid_mask]
            valid_targets = target[valid_mask]
            acc1, acc5 = accuracy(valid_outputs, valid_targets, topk=(1, 5))
            batch_valid_count = valid_mask.sum().item()
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_valid_count)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_valid_count)
        metric_logger.update(loss=loss.item())

    with open(file, "w") as f:
        f.write("video_id probabilities true_label chunk_nb split_nb\n")
        for line in final_result:
            f.write(line)

    metric_logger.synchronize_between_processes()
    print("* Test Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}"
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    # Combine probabilities from all batches
    all_probs = torch.cat(all_probs, dim=0)  # shape [N, C]

    ############################################################################
    # Compute Precision/Recall (using all samples, including negatives)
    ############################################################################
    def compute_precision_recall_with_negatives(all_probs, all_targets, category_names, threshold=0.0001):
        all_probs_np = all_probs.cpu().numpy()
        all_targets_np = np.array(all_targets)
        N, C = all_probs_np.shape
        for c in range(C):
            scores = all_probs_np[:, c]
            # Include all samples for precision/recall computation
            is_pos = (all_targets_np == c).astype(int)
            predicted_pos = (scores >= threshold).astype(int)
            tp = np.sum((predicted_pos == 1) & (is_pos == 1))
            fp = np.sum((predicted_pos == 1) & (is_pos == 0))
            fn = np.sum((predicted_pos == 0) & (is_pos == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            print(f"[Category {c} - {category_names.get(c, f'Cat {c}')}] Threshold={threshold:.2f}, "
                  f"TP={tp}, FP={fp}, FN={fn}, Precision={precision*100:.2f}%, Recall={recall*100:.2f}%")
    
    print("Precision/Recall on test set (threshold=0.5):")
    compute_precision_recall_with_negatives(all_probs, all_targets, category_names, threshold=0.0001)

    ############################################################################
    # Compute per-class Average Precision (AP) using all samples (including negatives)
    ############################################################################
    def compute_average_precision(all_probs, all_targets, category_names):
        # Convert tensor of probabilities to NumPy array
        all_probs_np = all_probs.cpu().numpy()
        all_targets_np = np.array(all_targets)
        C = all_probs_np.shape[1]
        print("Per-class Average Precision (AP):")
        for c in range(C):
            is_pos = (all_targets_np == c).astype(int)
            if np.sum(is_pos) == 0:
                ap = 0.0
            else:
                scores = all_probs_np[:, c]
                ap = average_precision_score(is_pos, scores)
            print(f"  AP for class {c} ({category_names.get(c, f'Cat {c}')}) : {ap:.4f}")
        print("End of per-class AP.\n")
    
    compute_average_precision(all_probs, all_targets, category_names)

    ############################################################################
    # Save confusion matrix using only valid (non -1) samples
    ############################################################################
    valid_idx = [i for i, lbl in enumerate(all_targets) if lbl != -1]
    valid_preds = [all_top1[i] for i in valid_idx]
    valid_labels = [all_targets[i] for i in valid_idx]
    cm = confusion_matrix(valid_labels, valid_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Final Test")
    plt.colorbar()
    tick_marks = np.arange(len(category_names))
    plt.xticks(tick_marks, [category_names[i] for i in tick_marks], rotation=45, ha="right")
    plt.yticks(tick_marks, [category_names[i] for i in tick_marks])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(file), "test_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[DEBUG] Saved final test confusion matrix to {cm_path}")
    plt.close()

    safe_barrier()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

############################################################################
# MERGE FUNCTION
############################################################################
def merge(eval_path, num_tasks):
    """
    Reads prediction files (named "0.txt", "1.txt", …) from eval_path and computes:
      - Overall top-1 and top-5 accuracy.
      - Per-category top-1 and top-5 accuracy.
    Debug statements are included.
    """
    overall_correct_top1 = 0
    overall_correct_top5 = 0
    overall_count = 0
    per_cat_stats = defaultdict(lambda: {"correct_top1": 0, "correct_top5": 0, "count": 0})
    
    print("[DEBUG] Reading individual output files for merging...")
    for i in range(num_tasks):
        pred_file = os.path.join(eval_path, f"{i}.txt")
        if not os.path.exists(pred_file):
            print(f"[DEBUG] File {pred_file} not found, skipping.")
            continue
        with open(pred_file, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        # Skip header line
        for line in lines[1:]:
            parts = json.loads(line)
            print("parts ", parts)
            if len(parts) < 3:
                print(f"[DEBUG] Line skipped due to insufficient parts: {line}")
                continue
            try:
                probs = parts[1]
                # probs = json.loads(parts[1])
            except Exception as e:
                print(f"[DEBUG] JSON parse failed for line: {line} with error: {e}")
                continue
            try:
                true_label = int(parts[2])
            except Exception as e:
                print(f"[DEBUG] True label parse failed for line: {line} with error: {e}")
                continue
            sorted_idx = sorted(range(len(probs)), key=lambda j: probs[j], reverse=True)
            top1 = sorted_idx[0]
            top5 = sorted_idx[:5]
            overall_count += 1
            if top1 == true_label:
                overall_correct_top1 += 1
            if true_label in top5:
                overall_correct_top5 += 1
            per_cat_stats[true_label]["count"] += 1
            if top1 == true_label:
                per_cat_stats[true_label]["correct_top1"] += 1
            if true_label in top5:
                per_cat_stats[true_label]["correct_top5"] += 1
    
    overall_top1 = overall_correct_top1 / overall_count if overall_count > 0 else 0
    overall_top5 = overall_correct_top5 / overall_count if overall_count > 0 else 0

    # print(f"[DEBUG] Overall Merged Accuracy: Top-1: {overall_top1*100:.2f}%, Top-5: {overall_top5*100:.2f}%")
    # print("Per-category accuracies (merged):")
    for cat in sorted(per_cat_stats.keys()):
        stats = per_cat_stats[cat]
        cat_acc1 = stats["correct_top1"] / stats["count"] if stats["count"] > 0 else 0
        cat_acc5 = stats["correct_top5"] / stats["count"] if stats["count"] > 0 else 0
        # print(f"  Category {cat}: Top-1: {cat_acc1*100:.2f}%, Top-5: {cat_acc5*100:.2f}%")
    return overall_top1, overall_top5

def compute_video(lst):
    i, video_id, data, label = lst
    feat = np.mean(data, axis=0)
    pred = np.argmax(feat)
    top1 = 1.0 if int(pred) == int(label) else 0.0
    top5 = 1.0 if int(label) in np.argsort(-feat)[:5] else 0.0
    return [pred, top1, top5, int(label)]

############################################################################
# End of file
############################################################################
