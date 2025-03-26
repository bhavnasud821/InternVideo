# import os
# import io
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# import warnings
# import pandas as pd
# from decord import VideoReader, cpu
# from .video_transforms import Compose, Resize, CenterCrop, Normalize
# from .volume_transforms import ClipToTensor

# from .random_erasing import RandomErasing
# from .video_transforms import (
#     Compose, Resize, CenterCrop, Normalize,
#     create_random_augment, random_short_side_scale_jitter, 
#     random_crop, random_resized_crop_with_shift, random_resized_crop,
#     horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
# )
# from .volume_transforms import ClipToTensor

# try:
#     from petrel_client.client import Client
#     has_client = True
# except ImportError:
#     has_client = False

# class CustomVideoClsDatasetNumeric(Dataset):
#     """
#     Custom video classification dataset that reads a CSV file with two columns:
#       - Column 0: Video file path (relative to a provided prefix)
#       - Column 1: Numeric label (e.g. 0, 1, …, 50)
#     The dataset uses Decord to load the video and samples a fixed number of frames.
#     Mimics the HMDBVideoClsDataset behavior.
#     """
#     def __init__(self, anno_path, prefix='', split=' ', mode='train', clip_len=16,
#                  crop_size=224, short_side_size=256, new_height=256, new_width=340,
#                  keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10,
#                  test_num_crop=3, filename_tmpl='img_{:05}.jpg', args=None):
#         self.anno_path = anno_path
#         self.prefix = prefix
#         self.split = split
#         self.mode = mode
#         self.clip_len = clip_len
#         self.crop_size = crop_size
#         self.short_side_size = short_side_size
#         self.new_height = new_height
#         self.new_width = new_width
#         self.keep_aspect_ratio = keep_aspect_ratio
#         self.num_segment = num_segment
#         self.test_num_segment = test_num_segment
#         self.num_crop = num_crop
#         self.test_num_crop = test_num_crop
#         self.filename_tmpl = filename_tmpl
#         self.args = args
#         self.aug = False
#         self.rand_erase = False

#         self.client = None
#         if has_client:
#             self.client = Client('~/petreloss.conf')

#         if self.mode == 'train':
#             self.aug = True
#             if self.args is not None and hasattr(self.args, 'reprob') and self.args.reprob > 0:
#                 self.rand_erase = True

#         # Ensure Decord is available
#         from decord import VideoReader
#         if VideoReader is None:
#             raise ImportError("Unable to import `decord` which is required to read videos.")

#         # Read CSV – expecting two columns: video_path, numeric label
#         df = pd.read_csv(self.anno_path, header=None, delimiter=self.split)
#         self.dataset_samples = list(df.values[:, 0].astype('str'))
#         self.label_array = [int(x) for x in df.values[:, 1]]
        
#         # Optionally record number of classes
#         self.num_classes = len(set(self.label_array))

#         # Define transforms (similar to HMDBVideoClsDataset)
#         if self.mode == 'train':
#             self.data_transform = None  # (Optionally add training augmentation here)
#         elif self.mode == 'validation':
#             self.data_transform = Compose([
#                 Resize(self.short_side_size, interpolation='bilinear'),
#                 CenterCrop(size=(self.crop_size, self.crop_size)),
#                 ClipToTensor(),
#                 Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225])
#             ])
#         elif self.mode == 'test':
#             self.data_resize = Compose([
#                 Resize(size=(self.short_side_size), interpolation='bilinear')
#             ])
#             self.data_transform = Compose([
#                 ClipToTensor(),
#                 Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225])
#             ])
#             # Mimic HMDB test branch
#             self.test_seg = []
#             self.test_dataset = []
#             self.test_label_array = []
#             for ck in range(self.test_num_segment):
#                 for cp in range(self.test_num_crop):
#                     for idx in range(len(self.label_array)):
#                         self.test_seg.append((ck, cp))
#                         self.test_dataset.append(self.dataset_samples[idx])
#                         self.test_label_array.append(self.label_array[idx])

#     def __getitem__(self, index):
#         if self.mode == 'train':
#             # Mimic HMDB training branch exactly:
#             args = self.args
#             scale_t = 1
#             sample = self.dataset_samples[index]
#             # For this custom dataset, we do not have total_frames info, so we call loadvideo_decord without it.
#             buffer = self.loadvideo_decord(sample)
#             if len(buffer) == 0:
#                 while len(buffer) == 0:
#                     warnings.warn(f"Video {sample} not correctly loaded during training.")
#                     index = np.random.randint(self.__len__())
#                     sample = self.dataset_samples[index]
#                     buffer = self.loadvideo_decord(sample)
#             if args.num_sample > 1:
#                 frame_list = []
#                 label_list = []
#                 index_list = []
#                 for _ in range(args.num_sample):
#                     new_frames = self._aug_frame(buffer, args)
#                     frame_list.append(new_frames)
#                     label_list.append(self.label_array[index])
#                     index_list.append(index)
#                 return frame_list, label_list, index_list, {}
#             else:
#                 buffer = self._aug_frame(buffer, args)
#             return buffer, self.label_array[index], index, {}  # Always return 4 values
#         elif self.mode == 'validation':
#             sample = self.dataset_samples[index]
#             buffer = self.loadvideo_decord(sample)
#             if len(buffer) == 0:
#                 while len(buffer) == 0:
#                     warnings.warn(f"Video {sample} not correctly loaded during validation.")
#                     index = np.random.randint(self.__len__())
#                     sample = self.dataset_samples[index]
#                     buffer = self.loadvideo_decord(sample)
#             buffer = self.data_transform(buffer)
#             video_id = os.path.splitext(os.path.basename(sample))[0]
#             return buffer, self.label_array[index], video_id
#         elif self.mode == 'test':
#             sample = self.test_dataset[index]
#             chunk_nb, split_nb = self.test_seg[index]
#             buffer = self.loadvideo_decord(sample)
#             while len(buffer) == 0:
#                 warnings.warn(f"Video {sample}, temporal {chunk_nb}, spatial {split_nb} not found during testing.")
#                 index = np.random.randint(self.__len__())
#                 sample = self.test_dataset[index]
#                 chunk_nb, split_nb = self.test_seg[index]
#                 buffer = self.loadvideo_decord(sample)
#             buffer = self.data_resize(buffer)
#             if isinstance(buffer, list):
#                 buffer = np.stack(buffer, 0)
#             buffer = self.data_transform(buffer)
#             video_id = os.path.splitext(os.path.basename(sample))[0]
#             return buffer, self.test_label_array[index], video_id, chunk_nb, split_nb

#     def __len__(self):
#         if self.mode != 'test':
#             return len(self.dataset_samples)
#         else:
#             return len(self.test_dataset)

#     def _aug_frame(self, buffer, args):
#         aug_transform = create_random_augment(
#             input_size=(self.crop_size, self.crop_size),
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#         )
#         buffer = [transforms.ToPILImage()(frame) for frame in buffer]
#         buffer = aug_transform(buffer)
#         buffer = [transforms.ToTensor()(img) for img in buffer]
#         buffer = torch.stack(buffer)  # T C H W
#         buffer = buffer.permute(0, 2, 3, 1)  # T H W C
#         buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
#                                   [0.229, 0.224, 0.225])
#         buffer = buffer.permute(3, 0, 1, 2)  # C T H W
#         scl, asp = ([0.08, 1.0], [0.75, 1.3333])
#         buffer = spatial_sampling(
#             buffer,
#             spatial_idx=-1,
#             min_scale=256,
#             max_scale=320,
#             crop_size=self.crop_size,
#             random_horizontal_flip=False if args.data_set == 'SSV2' else True,
#             inverse_uniform_sampling=False,
#             aspect_ratio=asp,
#             scale=scl,
#             motion_shift=False
#         )
#         if self.rand_erase:
#             erase_transform = RandomErasing(
#                 args.reprob,
#                 mode=args.remode,
#                 max_count=args.recount,
#                 num_splits=args.recount,
#                 device="cpu",
#             )
#             buffer = buffer.permute(1, 0, 2, 3)
#             buffer = erase_transform(buffer)
#             buffer = buffer.permute(1, 0, 2, 3)
#         return buffer

#     def loadvideo_decord(self, sample):
#         """Load video content using Decord and sample frames."""
#         fname = os.path.join(self.prefix, sample)
#         try:
#             vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
#         except Exception as e:
#             print("Video cannot be loaded by decord:", fname)
#             return []
#         total_frames = len(vr)
#         if total_frames == 0:
#             return []
#         if self.mode == 'train':
#             if total_frames < self.clip_len:
#                 indices = list(range(total_frames)) + [total_frames - 1] * (self.clip_len - total_frames)
#             else:
#                 indices = sorted(np.random.choice(total_frames, self.clip_len, replace=False))
#         else:
#             if total_frames < self.clip_len:
#                 indices = list(range(total_frames)) + [total_frames - 1] * (self.clip_len - total_frames)
#             else:
#                 start = (total_frames - self.clip_len) // 2
#                 indices = list(range(start, start + self.clip_len))
#         try:
#             ## try to rewrite to get matchign batch (not sure what this info is providing us)
#             buffer = vr.get_batch(indices).asnumpy()  # shape: (clip_len, H, W, C)
#         except Exception as e:
#             print("Error loading frames from video:", fname)
#             return []
#         return buffer

# def spatial_sampling(frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224,
#                      random_horizontal_flip=True, inverse_uniform_sampling=False,
#                      aspect_ratio=None, scale=None, motion_shift=False):
#     assert spatial_idx in [-1, 0, 1, 2]
#     if spatial_idx == -1:
#         if aspect_ratio is None and scale is None:
#             frames, _ = random_short_side_scale_jitter(
#                 images=frames,
#                 min_size=min_scale,
#                 max_size=max_scale,
#                 inverse_uniform_sampling=inverse_uniform_sampling,
#             )
#             frames, _ = random_crop(frames, crop_size)
#         else:
#             transform_func = random_resized_crop_with_shift if motion_shift else random_resized_crop
#             frames = transform_func(
#                 images=frames,
#                 target_height=crop_size,
#                 target_width=crop_size,
#                 scale=scale,
#                 ratio=aspect_ratio,
#             )
#         if random_horizontal_flip:
#             frames, _ = horizontal_flip(0.5, frames)
#     else:
#         assert len({min_scale, max_scale, crop_size}) == 1
#         frames, _ = random_short_side_scale_jitter(frames, min_scale, max_scale)
#         frames, _ = uniform_crop(frames, crop_size, spatial_idx)
#     return frames

# def tensor_normalize(tensor, mean, std):
#     if tensor.dtype == torch.uint8:
#         tensor = tensor.float() / 255.0
#     if isinstance(mean, list):
#         mean = torch.tensor(mean)
#     if isinstance(std, list):
#         std = torch.tensor(std)
#     return (tensor - mean) / std


# ####### code to test precision and recall 
#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import warnings
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode
import torch
from PIL import Image  # Needed for converting numpy arrays to PIL images

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

        df = pd.read_csv(anno_path, header=None, names=["video_path", "label"])
        # Convert label column to int and leave negatives as -1.
        self.label_array = [int(x) for x in df["label"].values]
        self.dataset_samples = list(df["video_path"].astype(str).values)
        self.num_classes = len(set(self.label_array))
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
            return frames, self.test_label_array[index], video_id, chunk_nb, split_nb

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
