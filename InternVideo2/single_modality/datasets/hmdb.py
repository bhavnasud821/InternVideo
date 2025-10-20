import os
import io
import cv2
import numpy as np
import json
import torch
from torchvision import transforms
from PIL import Image
import warnings
from decord import VideoReader, cpu
import random
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, UniformResize, Resize, CenterCrop, Normalize,
    create_random_augment, grayscale, random_short_side_scale_jitter,
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
)
from .volume_transforms import ClipToTensor

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

class Permute:
    def __init__(self, order):
        self.order = order

    def __call__(self, tensor):
        return tensor.permute(*self.order)

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)                 # Frame rate
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    cap.release()
    return fps, frame_count

class HMDBVideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, prefix='', split=' ', mode='train', clip_len=8,
                crop_size=224, short_side_size=256, new_height=256,
                new_width=340, keep_aspect_ratio=True, num_segment=1,
                num_crop=1, test_num_segment=10, test_num_crop=3, filename_tmpl=None, args=None):
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.multilabel = args.train_multilabel if self.mode == 'train' else args.eval_multilabel
        self.aug = False
        self.rand_erase = False
        
        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

        if self.mode == 'train':
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=",")
        self.dataset_samples = list(cleaned.values[:, 0])
        if self.multilabel:
            labels = [np.fromstring(item.strip('[]'), dtype=int, sep=' ') for item in cleaned.iloc[:, 1].to_list()]
            self.label_array = np.array(labels)
        else:
            self.label_array = list(cleaned.values[:, 1])
        self.class_counts = None
        if self.multilabel:
            self.class_counts = torch.tensor(self.label_array, dtype=torch.float).sum(dim=0)
        else:
            self.class_counts = torch.bincount(torch.tensor(self.label_array, dtype=torch.long), minlength=args.nb_classes).float()

        if (mode == 'train'):
            self.data_resize = Compose([
                UniformResize(size=(short_side_size),
                                        interpolation='bilinear')
            ])

        # elif (mode == 'validation'):
        #     self.data_transform = Compose([
        #         Resize(self.short_side_size, interpolation='bilinear'),
        #         CenterCrop(size=(self.crop_size, self.crop_size)),
        #         ClipToTensor(),
        #         Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        #     ])
        elif mode == 'test':
            if self.test_num_crop == 1:
                self.data_resize = Compose([
                    UniformResize(size=(short_side_size), interpolation='bilinear')
                ])
            else:
                self.data_resize = Compose([
                    Resize(size=(short_side_size), interpolation='bilinear')
                ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []

            for ck in range(self.test_num_segment):
                for idx in range(len(self.label_array)):
                    sample_label = self.label_array[idx]
                    self.test_label_array.append(sample_label)
                    self.test_dataset.append(self.dataset_samples[idx])
                    self.test_seg.append(ck)

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            label = self.label_array[index]
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    print("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    label = self.label_array[index]
                    buffer = self.loadvideo_decord(sample)
            negative_video = False
            if self.multilabel and (self.label_array[index].sum() == 0):
                negative_video = True
            elif (not self.multilabel) and (self.label_array[index] == args.nb_classes - 1):
                negative_video = True
            max_padding_ratio = args.max_padding_ratio_negative if negative_video else args.max_padding_ratio_positive
            min_padding_ratio = args.min_padding_ratio_negative if negative_video else args.min_padding_ratio_positive
            buffer = self.random_square_crop_around_people(buffer, sample, min_padding_ratio, max_padding_ratio, use_yolo_crop=args.train_yolo_crops,
                                                           centered=not args.new_spatial_augmentation)
            if args.save_training_images:
                resized_buffer = self.data_resize(buffer)
                # save input image
                rows, cols = 2, 4
                h, w = 224, 224
                grid = Image.new('RGB', (cols * w, rows * h))

                for j in range(8):
                    img = Image.fromarray(resized_buffer[j])
                    grid.paste(img, ((j % cols) * w, (j // cols) * h))
                rand_num = random.randint(0, 1000)
                save_path = f"{args.output_dir}/{self.label_array[index]}/{sample}_{str(rand_num)}_image_grid.png"
                parent_dir = os.path.dirname(save_path)

                os.makedirs(parent_dir, exist_ok=True)
                grid.save(save_path)
            buffer = self._aug_frame(buffer, args)
            if self.multilabel:
                return buffer, torch.tensor(self.label_array[index], dtype=torch.float32), index, {"path": sample}
            else:
                return buffer, torch.tensor(self.label_array[index], dtype=torch.long), index, {"path": sample}

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            args = self.args
            # segment_idx = self.test_seg[index]
            # buffer = self.loadvideo_decord(sample, segment_idx=segment_idx)
            buffer = self.loadvideo_decord(sample)

            if len(buffer) == 0:
                while len(buffer) == 0:
                    print("video {} not correctly loaded during testing".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.test_dataset[index]
                    # segment_idx = 0
                    buffer = self.loadvideo_decord(sample)
            buffer = self.random_square_crop_around_people(buffer, sample, args.test_padding_ratio, args.test_padding_ratio, use_yolo_crop=args.eval_yolo_crops)
            buffer = self.data_resize(buffer)
            if args.save_training_images:
                # save input image
                rows, cols = 2, 4
                h, w = 224, 224
                grid = Image.new('RGB', (cols * w, rows * h))

                for j in range(8):
                    img = Image.fromarray(buffer[j])
                    grid.paste(img, ((j % cols) * w, (j // cols) * h))
                save_path = f"{args.output_dir}/test_videos/{self.label_array[index]}/testing_{sample}_image_grid.png"
                parent_dir = os.path.dirname(save_path)

                os.makedirs(parent_dir, exist_ok=True)
                grid.save(save_path)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)
                transformed_buffer = self.data_transform(buffer)
            return buffer, transformed_buffer, torch.tensor(self.test_label_array[index]), sample, \
                0
            
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        # TODO: figure out if I need to change input_size here, since buffer is not resized
        aug_transform = create_random_augment(
            # input_size=(self.crop_size, self.crop_size),
            input_size=(buffer.shape[1], buffer.shape[2]),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            translate=True
            # translate=not args.new_spatial_augmentation
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 

        # T H W C
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            # [0.08, 1.0],
            [args.spatial_augmentation_min_scale, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False,
            apply_random_crop=True
            # apply_random_crop=not args.new_spatial_augmentation,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def expand_to_square(self, x1, y1, x2, y2, frame_width, frame_height, padding_ratio=0.1):
        """
        Expands a bounding box to a square shape with optional padding,
        clamping to frame boundaries.
        Coordinates are expected to be in pixels.
        """
        # Calculate current width and height
        w = x2 - x1
        h = y2 - y1

        # Handle invalid or empty bounding box
        if w <= 0 or h <= 0:
            return 0, 0, frame_width, frame_height

        # Calculate the desired side length of the square
        # Add padding based on the larger current dimension
        max_side = max(w, h)
        padding_amount = int(padding_ratio * max_side)
        desired_side = max_side + (2 * padding_amount)

        # Calculate center of the original bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate new coordinates before clamping
        new_x1 = center_x - desired_side / 2
        new_y1 = center_y - desired_side / 2
        new_x2 = center_x + desired_side / 2
        new_y2 = center_y + desired_side / 2

        # Clamp coordinates to frame boundaries
        clamped_x1 = max(0, new_x1)
        clamped_y1 = max(0, new_y1)
        clamped_x2 = min(frame_width, new_x2)
        clamped_y2 = min(frame_height, new_y2)

        # Adjust to maintain square aspect ratio after clamping
        actual_width = clamped_x2 - clamped_x1
        actual_height = clamped_y2 - clamped_y1

        if actual_width <= 0 or actual_height <= 0: # Fallback if clamping results in invalid box
            return 0, 0, frame_width, frame_height

        final_side = min(actual_width, actual_height) # Take the smaller of the two clamped dimensions

        # Recalculate based on the final_side and original center, then re-clamp
        # This ensures the final box is square and centered as much as possible
        final_x1 = center_x - final_side / 2
        final_y1 = center_y - final_side / 2
        final_x2 = center_x + final_side / 2
        final_y2 = center_y + final_side / 2

        # Final clamping to ensure it's within bounds after recentering
        final_x1 = int(max(0, final_x1))
        final_y1 = int(max(0, final_y1))
        final_x2 = int(min(frame_width, final_x2))
        final_y2 = int(min(frame_height, final_y2))

        # Ensure the final width/height is at least 1 pixel if possible
        if final_x2 <= final_x1:
            final_x2 = min(final_x1 + 1, frame_width)
            if final_x2 == final_x1: # if we can't even get 1 pixel, then just full frame
                final_x1 = 0
                final_x2 = frame_width
        if final_y2 <= final_y1:
            final_y2 = min(final_y1 + 1, frame_height)
            if final_y2 == final_y1:
                final_y1 = 0
                final_y2 = frame_height
        
        # If after all adjustments, the box is still invalid, return full frame
        if (final_x2 - final_x1) <= 0 or (final_y2 - final_y1) <= 0:
            return 0, 0, frame_width, frame_height

        return final_x1, final_y1, final_x2, final_y2

    def expand_bbox_non_centered(self, x1, y1, x2, y2, frame_width, frame_height, min_padding_ratio, max_padding_ratio):
        """
        Expands a bounding box by a random amount on each side, clamping to frame boundaries. The original bounding box will always be contained.

        Args:
            x1, y1, x2, y2 (int): Coordinates of the original bounding box (inclusive, pixel values).
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.
            min_padding_ratio (float): Minimum ratio by which individual padding amounts can exceed
                                       the original bbox dimension (e.g., 0.1 means up to 10% of bbox_w/h).
            max_padding_ratio (float): Maximum ratio by which individual padding amounts can exceed
                                       the original bbox dimension (e.g., 0.1 means up to 10% of bbox_w/h).

        Returns:
            tuple: (x1_crop, y1_crop, x2_crop, y2_crop) - the coordinates of the new crop.
                   Returns full frame (0,0,frame_width,frame_height) on failure.
        """
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Handle invalid or empty bounding box by returning the full frame
        if bbox_w <= 0 or bbox_h <= 0:
            print("Warning: random_padded_square_crop received an invalid bounding box. Returning full frame.")
            return 0, 0, frame_width, frame_height

        # 1. Generate random padding for each side
        bias_horizontal = random.choice([True, False])

        if bias_horizontal:
            # Bias padding horizontally
            if random.choice([True, False]): # Bias to the left
                pad_left = random.uniform(min_padding_ratio * bbox_w, max_padding_ratio * bbox_w) # More padding
                pad_right = random.uniform(min_padding_ratio * bbox_w, max_padding_ratio * bbox_w / 2) # Less padding
            else: # Bias to the right
                pad_left = random.uniform(min_padding_ratio * bbox_w, max_padding_ratio * bbox_w / 2) # Less padding
                pad_right = random.uniform(min_padding_ratio * bbox_w, max_padding_ratio * bbox_w) # More padding

            # Vertical padding remains somewhat balanced but still random
            pad_top = random.uniform(min_padding_ratio * bbox_h, max_padding_ratio * bbox_h)
            pad_bottom = random.uniform(min_padding_ratio * bbox_h, max_padding_ratio * bbox_h)
        else:
            # Bias padding vertically
            if random.choice([True, False]): # Bias to the top
                pad_top = random.uniform(min_padding_ratio * bbox_h, max_padding_ratio * bbox_h) # More padding
                pad_bottom = random.uniform(min_padding_ratio * bbox_h, max_padding_ratio * bbox_h / 2) # Less padding
            else: # Bias to the bottom
                pad_top = random.uniform(min_padding_ratio * bbox_h, max_padding_ratio * bbox_h / 2) # Less padding
                pad_bottom = random.uniform(min_padding_ratio * bbox_h, max_padding_ratio * bbox_h) # More padding

            # Horizontal padding remains somewhat balanced but still random
            pad_left = random.uniform(min_padding_ratio * bbox_w, max_padding_ratio * bbox_w)
            pad_right = random.uniform(min_padding_ratio * bbox_w, max_padding_ratio * bbox_w)

        # print("chosen paddings, left ", pad_left, " right ", pad_right, " top ", pad_top, " bottom ", pad_bottom)

        # 2. Apply padding and clamp to frame boundaries for initial expanded box
        temp_x1 = max(0.0, x1 - pad_left)
        temp_y1 = max(0.0, y1 - pad_top)
        temp_x2 = min(float(frame_width), x2 + pad_right)
        temp_y2 = min(float(frame_height), y2 + pad_bottom)

        # Calculate current dimensions of the initially padded and clamped box
        current_w = temp_x2 - temp_x1
        current_h = temp_y2 - temp_y1

        # Fallback if initial random padding resulted in an invalid box (e.g., due to extreme clamping)
        if current_w <= 0 or current_h <= 0:
            print("Warning: Initial random padding resulted in zero or negative dimensions after clamping. Returning full frame.")
            return 0, 0, frame_width, frame_height

        # print("temp_x1: ", temp_x1,", temp_y1: ", temp_y1, ", temp_x2: ", temp_x2, ", temp_y2: ", temp_y2, ", frame_width: ", frame_width, ", frame_height: ", frame_height)
        if current_w / current_h < 0.75 or current_w / current_h > 1.33:
            # 3. Adjust to make the box square, expanding the smaller dimension
            # Determine the target side length (the larger of the current dimensions)
            target_side = max(current_w, current_h)

            # Calculate the center of the current padded box
            center_x = (temp_x1 + temp_x2) / 2.0
            center_y = (temp_y1 + temp_y2) / 2.0

            # Attempt to create a square of 'target_side' length, centered on the current box
            final_x1_cand = center_x - target_side / 2.0
            final_y1_cand = center_y - target_side / 2.0
            final_x2_cand = center_x + target_side / 2.0
            final_y2_cand = center_y + target_side / 2.0

            # Clamp these candidate coordinates to ensure they are within frame boundaries
            clamped_x1 = max(0.0, final_x1_cand)
            clamped_y1 = max(0.0, final_y1_cand)
            clamped_x2 = min(float(frame_width), final_x2_cand)
            clamped_y2 = min(float(frame_height), final_y2_cand)

            # print("expanded because of aspect ratio, new coords x1 y1 x2 y2: ", clamped_x1, " ", clamped_y1, " ", clamped_x2, " ", clamped_y2)
            return int(clamped_x1), int(clamped_y1), int(clamped_x2), int(clamped_y2)
        else:
            return int(temp_x1), int(temp_y1), int(temp_x2), int(temp_y2)

    def random_square_crop_around_people(self, buffer, sample, min_padding_ratio, max_padding_ratio, use_yolo_crop=False, centered=True):
        # print("centered is ", centered)
        video_fname = os.path.join(self.prefix, sample)
        if use_yolo_crop:
            if os.path.exists(video_fname.replace(".mp4", "_yolo_crop_info.txt")):
                crop_info_fname = video_fname.replace(".mp4", "_yolo_crop_info.txt")
            else: 
                crop_info_fname = video_fname.replace(".mp4", "_crop_info.txt")
        else:
            crop_info_fname = video_fname.replace(".mp4", "_crop_info.txt")
        with open(crop_info_fname, "r") as f:
            crop_data = json.load(f)
            crop_x1 = crop_data["crop_x1"]
            crop_x2 = crop_data["crop_x2"]
            crop_y1 = crop_data["crop_y1"]
            crop_y2 = crop_data["crop_y2"]
            video_w = crop_data["video_w"]
            video_h = crop_data["video_h"]
            if centered:
                random_padding_ratio = random.uniform(min_padding_ratio, max_padding_ratio)
                x1, y1, x2, y2 = self.expand_to_square(crop_x1, crop_y1, crop_x2, crop_y2, video_w, video_h, padding_ratio=random_padding_ratio)
            else:
                # first make the bbox square without expanding
                x1, y1, x2, y2 = self.expand_to_square(crop_x1, crop_y1, crop_x2, crop_y2, video_w, video_h, padding_ratio=0)
                # then expand randomly in the left, right, top, and bottom directions
                x1, y1, x2, y2 = self.expand_bbox_non_centered(x1, y1, x2, y2, video_w, video_h, min_padding_ratio, max_padding_ratio)
            buffer = buffer[:, y1:y2, x1:x2, :]
        return buffer

    def loadvideo_decord(self, sample):
        """Load video content using Decord"""
        fname = os.path.join(self.prefix, sample)

        try:
            if self.keep_aspect_ratio:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     width=self.new_width,
                                     height=self.new_height,
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                    num_threads=1, ctx=cpu(0))

            if self.mode == 'test':
                total_frames = len(vr)
                all_index = np.linspace(0, total_frames - 1, self.num_segment, dtype=int).tolist()
                buffer = vr.get_batch(all_index).asnumpy()
                return buffer
            elif self.mode == 'validation':
                tick = len(vr) / float(self.num_segment)
                all_index = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)])
                vr.seek(0)
                buffer = vr.get_batch(all_index).asnumpy()
                return buffer

            # handle temporal segments
            start_idx = 0
            sampling_range = len(vr) - start_idx
            average_duration = sampling_range // self.num_segment
            if average_duration > 0:
                all_index = list(np.multiply(list(range(self.num_segment)), average_duration) + 
                                     np.random.randint(average_duration, size=self.num_segment) + 
                                     start_idx)
                # all_index = list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                            # size=self.num_segment))
            elif len(vr) > self.num_segment:
                all_index = list(np.sort(np.random.randint(start_idx, len(vr), size=self.num_segment)))
            else:
                all_index = list(np.zeros((self.num_segment,)))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
    apply_random_crop=True
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            # print("applying random short side scale jitter")
            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        elif apply_random_crop:
            # print("applying random resized crop")
            transform_func = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        else:
            frames = torch.nn.functional.interpolate(frames, size=(crop_size, crop_size), mode='bilinear')

        if random_horizontal_flip:
            # print("applying random horizontal flip")
            frames, _ = horizontal_flip(0.5, frames)
    else:
        # print("applying random short side scale jitter and uniform crop")
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
