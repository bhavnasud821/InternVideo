#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import warnings
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode
import torch
from PIL import Image

class CustomVideoClsDatasetNumeric:
    def __init__(self, anno_path, prefix, split, mode, clip_len, crop_size, short_side_size, new_height, new_width, filename_tmpl, args):
        """
        Initialize the dataset.
        The CSV file should have two columns: video_path,label.
        Negative samples are marked as -1 and are kept as -1.
        """
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.filename_tmpl = filename_tmpl
        self.args = args

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=",")
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])
        self.num_classes = self.label_array.shape[1] if args.multilabel else len(set(self.label_array))
        print(f"Initialized dataset with {len(self.dataset_samples)} samples and {self.num_classes} classes.")

        if self.mode == 'train':
            self.data_transform = None
        elif self.mode == 'validation':
            self.data_transform = Compose([
                Resize((self.short_side_size, self.short_side_size), interpolation=InterpolationMode.BILINEAR),
                CenterCrop((self.crop_size, self.crop_size)),
                ToTensor(),
            ])
        elif self.mode == 'test':
            # For test mode, we define a separate resize and transform.
            self.data_resize = Compose([
                Resize((self.short_side_size, self.short_side_size), interpolation=InterpolationMode.BILINEAR)
            ])
            self.data_transform = Compose([
                ToTensor(),
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.args.test_num_segment):
                for cp in range(self.args.test_num_crop):
                    for idx in range(len(self.label_array)):
                        self.test_seg.append((ck, cp))
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_label_array.append(self.label_array[idx])
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = self.dataset_samples[index]
            # TODO: figure out why this isn't using chunk nb or spatial nb
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(f"Video {sample} not loaded during training.")
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            if self.args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(self.args.num_sample):
                    frames = self._aug_frame(buffer, self.args)  # [T, C, H, W]
                    frames = frames.permute(1, 0, 2, 3)           # [C, T, H, W]
                    frame_list.append(frames)
                    label_list.append(self.label_array[index])
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                frames = self._aug_frame(buffer, self.args)
                frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
                return frames, self.label_array[index], index, {}
        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(f"Video {sample} not loaded during validation.")
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            # For each frame, convert to PIL image, apply the transform, then stack.
            transformed_frames = []
            for i in range(buffer.shape[0]):
                pil_img = Image.fromarray(buffer[i])
                transformed_img = self.data_transform(pil_img)
                transformed_frames.append(transformed_img)
            frames = torch.stack(transformed_frames, dim=0)  # [T, C, H, W]
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
            video_id = os.path.splitext(os.path.basename(sample))[0]
            return frames, self.label_array[index], video_id
        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)
            while len(buffer) == 0:
                warnings.warn(f"Video {sample} not loaded during testing.")
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)
            # Instead of passing the numpy array directly, convert each frame to PIL, apply resize then transform.
            transformed_frames = []
            for i in range(buffer.shape[0]):
                pil_img = Image.fromarray(buffer[i])
                resized_img = self.data_resize(pil_img)  # Apply Resize transform.
                transformed_img = self.data_transform(resized_img)  # Apply ToTensor transform.
                transformed_frames.append(transformed_img)
            frames = torch.stack(transformed_frames, dim=0)  # [T, C, H, W]
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
            video_id = os.path.splitext(os.path.basename(sample))[0]
            return frames, self.test_label_array[index], sample, chunk_nb, split_nb
            # return frames, self.test_label_array[index], video_id, chunk_nb, split_nb
            

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

    def _aug_frame(self, buffer, args):
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        # Convert each frame to a PIL image, then to tensor.
        frames = [to_pil(frame) for frame in buffer]
        frames = [to_tensor(img) for img in frames]
        # Stack frames to get a tensor of shape [T, C, H, W]
        frames = torch.stack(frames)
        return frames

    def loadvideo_decord(self, sample):
        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("Please install decord via 'pip install decord'")
        fname = os.path.join(self.prefix, sample)
        try:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        except Exception as e:
            print("Video cannot be loaded by decord:", fname)
            return []
        total_frames = len(vr)
        if total_frames == 0:
            return []
        if self.mode == 'train':
            if total_frames < self.clip_len:
                indices = list(range(total_frames)) + [total_frames - 1] * (self.clip_len - total_frames)
            else:
                indices = sorted(np.random.choice(total_frames, self.clip_len, replace=False))
        else:
            if total_frames < self.clip_len:
                indices = list(range(total_frames)) + [total_frames - 1] * (self.clip_len - total_frames)
            else:
                # start = (total_frames - self.clip_len) // 2
                # indices = list(range(start, start + self.clip_len))
                indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int).tolist()
        try:
            buffer = vr.get_batch(indices).asnumpy()
            import cv2
            resized = []
            for frame in buffer:
                frame_resized = cv2.resize(frame, (self.crop_size, self.crop_size))
                resized.append(frame_resized)
            buffer = np.stack(resized, axis=0)
        except Exception as e:
            print("Error loading frames from video:", fname)
            return []
        return buffer
