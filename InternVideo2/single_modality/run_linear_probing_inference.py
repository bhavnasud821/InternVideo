#### to resolve final error
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict
import imageio

from datasets.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.nn import BCEWithLogitsLoss
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

# Use the build function that supports the custom dataset branch.
from datasets import build_dataset
from engines.engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
# from engines.engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode
from utils import multiple_samples_collate
import utils
from models import *
from PIL import Image

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['RDMAV_FORK_SAFE'] = '1'

def loadvideo_decord(sample_path, num_frames, crop_size):
        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("Please install decord via 'pip install decord'")
        fname = sample_path
        try:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        except Exception as e:
            print("Video cannot be loaded by decord:", fname)
            return []
        total_frames = len(vr)
        if total_frames == 0:
            return []
        if total_frames < num_frames:
            indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        try:
            buffer = vr.get_batch(indices).asnumpy()
            import cv2
            resized = []
            for frame in buffer:
                frame_resized = cv2.resize(frame, (crop_size, crop_size))
                resized.append(frame_resized)
            buffer = np.stack(resized, axis=0)
        except Exception as e:
            print("Error loading frames from video:", fname)
            return []
        return buffer

def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    # Add all your existing arguments here (for brevity, not all are reprinted below)
    parser.add_argument('--sample_path', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--steps_per_print', default=1, type=int)
    parser.add_argument('--use_ceph_checkpoint', action='store_true', help="whether use ceph to save and load checkpoint, may be some bug now")
    parser.set_defaults(use_ceph_checkpoint=False)
    parser.add_argument('--ceph_checkpoint_prefix', default='', type=str, help='prefix for checkpoint in ceph')
    parser.add_argument('--ckpt_path_split', default='/exp/', type=str, help='string for splitting the ckpt_path')
    parser.add_argument('--multilabel', action='store_true', help="whether to use multilabel sigmoid loss training")

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')
    parser.add_argument('--layer_scale_init_value', default=1e-5, type=float, help="0.1 for base, 1e-5 for large. set 0 to disable LayerScale")
    parser.add_argument('--layerscale_no_force_fp32', action='store_true', help="Not force fp32 for LayerScale")
    parser.set_defaults(layerscale_no_force_fp32=False)
    parser.add_argument('--sep_pos_embed', action='store_true', help="whether use seperable position embedding")
    parser.add_argument('--center_init', action='store_true', help="center initialization for patch embedding")
    parser.add_argument('--orig_t_size', type=int, default=8)

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--head_drop_path', type=float, default=0.0, metavar='PCT', help='Head Drop path rate (default: 0.0)')

    ## Set to true because we don't want a val dataset/dataloader
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--merge_method', type=str, default='proj', help='merge method for features')
    parser.add_argument('--merge_norm', type=str, default='kaiming_BN', help='merge Norm for features')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw")')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the weight decay.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (default: 1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='num of steps to warmup LR')
    parser.add_argument('--open_clip_projector', action='store_true', help="whether open clip projector for training")
    parser.set_defaults(open_clip_projector=False)
    parser.add_argument('--open_block_num', type=int, default=0, help="whether open the last few blocks")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use for training (default: 0)")

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2, help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic', help='Training interpolation (default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--finetune_extra', default='', help='finetune from extra checkpoint')
    parser.add_argument('--delete_head', action='store_true', help='whether delete head')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--checkpoint_num', default=0, type=int, help='number of layers for using checkpoint')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--prefix', default='', type=str, help='prefix for data')
    parser.add_argument('--split', default=' ', type=str, help='split for metadata')
    parser.add_argument('--filename_tmpl', default='img_{:05}.jpg', type=str, help='file template')
    parser.add_argument('--data_path', default='you_data_path', type=str, help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str, help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=400, type=int, help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--use_decord', action='store_true', help='whether use decord to load video, otherwise load image')
    parser.add_argument('--no_use_decord', action='store_false', dest='use_decord')
    parser.set_defaults(use_decord=True)
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--data_set', default='Kinetics', choices=[
        'Kinetics', 'Kinetics_sparse', 'SSV2', 'UCF101', 'HMDB51', 'image_folder', 'mitv1_sparse',
        'ANet', 'HACS', 'ANet_interval', 'HACS_interval', 'MyCustom'
    ], type=str, help='dataset')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--test_best', action='store_true', help='Whether test the best model')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # ADD THIS ARGUMENT for deepspeed
    parser.add_argument('--enable_deepspeed', action='store_true', default=False, help='Enable deepspeed optimization')

    # BF16 and Zero optimization
    parser.add_argument('--bf16', default=False, action='store_false')
    parser.add_argument('--zero_stage', default=0, type=int, help='ZeRO optimizer stage (default: 0)')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_internvideo2_ds_config(args)

    print(args)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("DEBUG: Using device: %s", device)
    
    # Set seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Load the model
    if 'cat' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            num_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            sep_pos_embed=args.sep_pos_embed,
            fc_drop_rate=args.fc_drop_rate,
            drop_path_rate=args.drop_path,
            head_drop_path_rate=args.head_drop_path,
            use_checkpoint=args.use_checkpoint,
            checkpoint_num=args.checkpoint_num,
            init_scale=args.init_scale,
            init_values=args.layer_scale_init_value,
            layerscale_no_force_fp32=args.layerscale_no_force_fp32,
            merge_method=args.merge_method,
            merge_norm=args.merge_norm,
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            num_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            sep_pos_embed=args.sep_pos_embed,
            fc_drop_rate=args.fc_drop_rate,
            drop_path_rate=args.drop_path,
            head_drop_path_rate=args.head_drop_path,
            use_checkpoint=args.use_checkpoint,
            checkpoint_num=args.checkpoint_num,
            init_scale=args.init_scale,
            init_values=args.layer_scale_init_value,
            layerscale_no_force_fp32=args.layer_scale_init_value,
        )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    print("start epoch here is ", args.start_epoch)
    print("finetune is ", args.finetune)
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        if 'head.weight' in checkpoint_model.keys():
            if args.delete_head:
                print("Removing head from pretrained checkpoint")
                del checkpoint_model['head.weight']
                del checkpoint_model['head.bias']
            elif checkpoint_model['head.weight'].shape[0] == 710:
                if args.nb_classes == 400:
                    checkpoint_model['head.weight'] = checkpoint_model['head.weight'][:args.nb_classes]
                    checkpoint_model['head.bias'] = checkpoint_model['head.bias'][:args.nb_classes]
                elif args.nb_classes in [600, 700]:
                    map_path = f'./k710/label_mixto{args.nb_classes}.json'
                    print(f'Load label map from {map_path}')
                    with open(map_path) as f:
                        label_map = json.load(f)
                    checkpoint_model['head.weight'] = checkpoint_model['head.weight'][label_map]
                    checkpoint_model['head.bias'] = checkpoint_model['head.bias'][label_map]
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        
        print("finetune extra is ", args.finetune_extra)
        if args.finetune_extra:
            extra_checkpoint = torch.load(args.finetune_extra, map_location='cpu')
            print("Load extra ckpt from %s" % args.finetune_extra)
            extra_checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in extra_checkpoint:
                    extra_checkpoint_model = extra_checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            for k, v in extra_checkpoint_model.items():
                new_k = k
                if k.startswith('vision_encoder.'):
                    new_k = k.replace('vision_encoder.', '')
                else:
                    print(f"Ignore keys: {k}")
                    continue
                checkpoint_model[new_k] = v

        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            orig_t_size = args.orig_t_size
            new_t_size = args.num_frames * args.num_segments // model.patch_embed.tubelet_size
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // orig_t_size) ** 0.5)
            new_size = int((num_patches // new_t_size) ** 0.5)
            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
                pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
                pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
                pos_embed_checkpoint = new_pos_embed
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        elif 'pos_embed_spatial' in checkpoint_model and 'pos_embed_temporal' in checkpoint_model:
            pos_embed_spatial_checkpoint = checkpoint_model['pos_embed_spatial']
            pos_embed_temporal_checkpoint = checkpoint_model['pos_embed_temporal']
            embedding_size = pos_embed_spatial_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            orig_t_size = pos_embed_temporal_checkpoint.shape[-2]
            new_t_size = args.num_frames // model.patch_embed.tubelet_size
            orig_size = int(pos_embed_spatial_checkpoint.shape[-2] ** 0.5)
            new_size = int((num_patches // new_t_size) ** 0.5)
            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
                tmp_pos_embed = pos_embed_temporal_checkpoint.view(1, orig_t_size, -1, embedding_size)
                tmp_pos_embed = tmp_pos_embed.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                tmp_pos_embed = torch.nn.functional.interpolate(tmp_pos_embed, size=new_t_size, mode='linear')
                tmp_pos_embed = tmp_pos_embed.view(1, -1, embedding_size, new_t_size)
                tmp_pos_embed = tmp_pos_embed.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                checkpoint_model['pos_embed_temporal'] = tmp_pos_embed
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                pos_tokens = pos_embed_spatial_checkpoint
                pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)
                checkpoint_model['pos_embed_spatial'] = pos_tokens

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    print("Freeze backbone for linear probing")
    if '6B' in args.model:
        depth = 48
    else:
        depth = 40  # ViT-g
    block_num_list = [(depth - i - 1) for i in range(args.open_block_num)]
    for name, p in model.named_parameters():
        if name.startswith('patch_embed') or name.startswith('pos_embed') or name.startswith('cls_token'):
            print(f"Freeze {name}")
            p.requires_grad = False
        elif name.startswith('blocks'):
            flag = True
            for num in block_num_list:
                if name.startswith(f'blocks.{num}'):
                    flag = False
                    break
            if flag:
                print(f"Freeze {name}")
                p.requires_grad = False
            else:
                print(f"Unfreeze {name}")
        elif name.startswith('clip_projector') and not args.open_clip_projector:
            print(f"Freeze {name}")
            p.requires_grad = False
        else:
            print(f"Unfreeze {name}")

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    ceph_args = {
        'use_ceph_checkpoint': args.use_ceph_checkpoint,
        'ceph_checkpoint_prefix': args.ceph_checkpoint_prefix,
        'ckpt_path_split': args.ckpt_path_split,
        'local_rank': args.gpu,
    }
    if ceph_args['use_ceph_checkpoint']:
        print("Will automatically upload model on ceph")
        assert ceph_args['ceph_checkpoint_prefix'] != '', "Should set prefix for ceph checkpoint!"

    print("start epoch before auto load model ", args.start_epoch)
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=None, loss_scaler=None, model_ema=model_ema,
        ceph_args=ceph_args,
    )
    print("start epoch after auto load model ", args.start_epoch)



    # Load the sample video
    sample = args.sample_path
    buffer = loadvideo_decord(sample, args.num_frames, args.input_size)
    # Instead of passing the numpy array directly, convert each frame to PIL, apply resize then transform.
    transformed_frames = []

    data_resize = Compose([
        Resize((args.input_size, args.input_size), interpolation=InterpolationMode.BILINEAR)
    ])
    data_transform = Compose([
        ToTensor(),
    ])
    for i in range(buffer.shape[0]):
        pil_img = Image.fromarray(buffer[i])
        resized_img = data_resize(pil_img)  # Apply Resize transform.
        transformed_img = data_transform(resized_img)  # Apply ToTensor transform.
        transformed_frames.append(transformed_img)
    

    # Convert frames to numpy format for visualization
    frames_np = [frame.squeeze().permute(1, 2, 0).cpu().numpy() for frame in transformed_frames]  # Convert [C, H, W] -> [H, W, C]
    frames_np = [(frame * 255).astype(np.uint8) for frame in frames_np]  # Ensure values are in range [0, 255]

    # Save as GIF
    # gif_path = args.sample_path[:-4] + ".gif"
    # imageio.mimsave(gif_path, frames_np, fps=10)  # Adjust fps as needed

    frames = torch.stack(transformed_frames, dim=0)  # [T, C, H, W]
    frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
    videos = frames.unsqueeze(0)
    videos = videos.to(device, non_blocking=True)

    outputs = model(videos)
    print("outputs ", outputs)
    softmax_outputs = torch.softmax(outputs, dim=-1)
    print("softmax outputs ", softmax_outputs)


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)

