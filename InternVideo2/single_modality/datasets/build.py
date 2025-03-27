# import os
# from torchvision import transforms
# from .transforms import *
# from .masking_generator import TubeMaskingGenerator, RandomMaskingGenerator
# from .mae import VideoMAE
# from .mae_multi import VideoMAE_multi
# from .kinetics import VideoClsDataset
# from .kinetics_sparse import VideoClsDataset_sparse
# from .anet import ANetDataset
# from .ssv2 import SSVideoClsDataset, SSRawFrameClsDataset
# from .hmdb import HMDBVideoClsDataset, HMDBRawFrameClsDataset
# # NEW: Import our custom video classification dataset (numeric labels)
# from .custom_video_cls_dataset_numeric import CustomVideoClsDatasetNumeric

# def build_pretraining_dataset(args):
#     transform = DataAugmentationForVideoMAE(args)
#     dataset = VideoMAE(
#         root=None,
#         setting=args.data_path,
#         prefix=args.prefix,
#         split=",",
#         video_ext='mp4',
#         is_color=True,
#         modality='rgb',
#         num_segments=args.num_segments,
#         new_length=args.num_frames,
#         new_step=args.sampling_rate,
#         transform=transform,
#         temporal_jitter=False,
#         video_loader=True,
#         use_decord=args.use_decord,
#         lazy_init=False,
#         num_sample=args.num_sample)
#     print("Data Aug = %s" % str(transform))
#     return dataset

# def build_multi_pretraining_dataset(args):
#     original_flip = args.flip
#     transform = DataAugmentationForVideoMAE(args)
#     args.flip = False
#     transform_ssv2 = DataAugmentationForVideoMAE(args)
#     args.flip = original_flip

#     dataset = VideoMAE_multi(
#         root=None,
#         setting=args.data_path,
#         prefix=args.prefix,
#         split=",",
#         is_color=True,
#         modality='rgb',
#         num_segments=args.num_segments,
#         new_length=args.num_frames,
#         new_step=args.sampling_rate,
#         transform=transform,
#         transform_ssv2=transform_ssv2,
#         temporal_jitter=False,
#         video_loader=True,
#         use_decord=args.use_decord,
#         lazy_init=False,
#         num_sample=args.num_sample)
#     print("Data Aug = %s" % str(transform))
#     print("Data Aug for SSV2 = %s" % str(transform_ssv2))
#     return dataset

# def build_dataset(is_train, test_mode, args):
#     print(f'Use Dataset: {args.data_set}')
#     if args.data_set in [
#             'Kinetics',
#             'Kinetics_sparse',
#             'mitv1_sparse'
#         ]:
#         mode = 'train' if is_train else ('test' if test_mode else 'validation')
#         anno_path = os.path.join(args.data_path, f"{mode}.csv")
#         if 'sparse' in args.data_set:
#             func = VideoClsDataset_sparse
#         else:
#             func = VideoClsDataset

#         dataset = func(
#             anno_path=anno_path,
#             prefix=args.prefix,
#             split=",",
#             mode=mode,
#             clip_len=args.num_frames,
#             frame_sample_rate=args.sampling_rate,
#             num_segment=1,
#             test_num_segment=args.test_num_segment,
#             test_num_crop=args.test_num_crop,
#             num_crop=1 if not test_mode else 3,
#             keep_aspect_ratio=True,
#             crop_size=args.input_size,
#             short_side_size=args.short_side_size,
#             new_height=256,
#             new_width=320,
#             args=args)
        
#         nb_classes = args.nb_classes
    
#     elif args.data_set == 'SSV2':
#         mode = 'train' if is_train else ('test' if test_mode else 'validation')
#         anno_path = os.path.join(args.data_path, f"{mode}.csv")
#         if args.use_decord:
#             func = SSVideoClsDataset
#         else:
#             func = SSRawFrameClsDataset

#         dataset = func(
#             anno_path=anno_path,
#             prefix=args.prefix,
#             split=",",
#             mode=mode,
#             clip_len=1,
#             num_segment=args.num_frames,
#             test_num_segment=args.test_num_segment,
#             test_num_crop=args.test_num_crop,
#             num_crop=1 if not test_mode else 3,
#             keep_aspect_ratio=True,
#             crop_size=args.input_size,
#             short_side_size=args.short_side_size,
#             new_height=256,
#             new_width=320,
#             filename_tmpl=args.filename_tmpl,
#             args=args)
#         nb_classes = 174

#     elif args.data_set == 'UCF101':
#         mode = 'train' if is_train else ('test' if test_mode else 'validation')
#         anno_path = os.path.join(args.data_path, f"{mode}.csv")
#         dataset = VideoClsDataset(
#             anno_path=anno_path,
#             prefix=args.prefix,
#             split=",",
#             mode=mode,
#             clip_len=args.num_frames,
#             frame_sample_rate=args.sampling_rate,
#             num_segment=1,
#             test_num_segment=args.test_num_segment,
#             test_num_crop=args.test_num_crop,
#             num_crop=1 if not test_mode else 3,
#             keep_aspect_ratio=True,
#             crop_size=args.input_size,
#             short_side_size=args.short_side_size,
#             new_height=256,
#             new_width=320,
#             args=args)
#         nb_classes = 101

#     elif args.data_set == 'HMDB51':
#         mode = 'train' if is_train else ('test' if test_mode else 'validation')
#         anno_path = os.path.join(args.data_path, f"{mode}.csv")
#         if args.use_decord:
#             func = HMDBVideoClsDataset
#         else:
#             func = HMDBRawFrameClsDataset

#         dataset = func(
#             anno_path=anno_path,
#             prefix=args.prefix,
#             split=",",
#             mode=mode,
#             clip_len=1,
#             num_segment=args.num_frames,
#             test_num_segment=args.test_num_segment,
#             test_num_crop=args.test_num_crop,
#             num_crop=1 if not test_mode else 3,
#             keep_aspect_ratio=True,
#             crop_size=args.input_size,
#             short_side_size=args.short_side_size,
#             new_height=256,
#             new_width=320,
#             filename_tmpl=args.filename_tmpl,
#             args=args)
#         nb_classes = 51

#     elif args.data_set in ['ANet', 'HACS', 'ANet_interval', 'HACS_interval']:
#         mode = 'train' if is_train else ('test' if test_mode else 'validation')
#         anno_path = os.path.join(args.data_path, f"{mode}.csv")
#         if 'interval' in args.data_set:
#             func = ANetDataset
#         else:
#             func = VideoClsDataset_sparse

#         dataset = func(
#             anno_path=anno_path,
#             prefix=args.prefix,
#             split=",",
#             mode=mode,
#             clip_len=args.num_frames,
#             frame_sample_rate=args.sampling_rate,
#             num_segment=1,
#             test_num_segment=args.test_num_segment,
#             test_num_crop=args.test_num_crop,
#             num_crop=1 if not test_mode else 3,
#             keep_aspect_ratio=True,
#             crop_size=args.input_size,
#             short_side_size=args.short_side_size,
#             new_height=256,
#             new_width=320,
#             args=args)
#         nb_classes = args.nb_classes

#     # NEW branch for our custom video dataset with numeric labels:
#     elif args.data_set == 'MyCustom':
#         mode = 'train' if is_train else ('test' if test_mode else 'validation')
#         anno_path = os.path.join(args.data_path, f"{mode}.csv")
#         dataset = CustomVideoClsDatasetNumeric(
#             anno_path=anno_path,
#             prefix=args.prefix,
#             split=",",
#             mode=mode,
#             clip_len=args.num_frames,
#             crop_size=args.input_size,
#             short_side_size=args.short_side_size,
#             new_height=256,
#             new_width=320,
#             filename_tmpl=args.filename_tmpl,
#             args=args
#         )
#         nb_classes = len(set(dataset.label_array))
#         # nb_classes = 23
#     else:
#         print(f'Unsupported dataset: {args.data_set}')
#         raise NotImplementedError()
#     print("Number of classes:", nb_classes)
#     return dataset, nb_classes


# #### code to test preciaion AND RECALL
import os
from torchvision import transforms
from .transforms import *
from .masking_generator import TubeMaskingGenerator, RandomMaskingGenerator
from .mae import VideoMAE
from .mae_multi import VideoMAE_multi
from .kinetics import VideoClsDataset
from .kinetics_sparse import VideoClsDataset_sparse
from .anet import ANetDataset
from .ssv2 import SSVideoClsDataset, SSRawFrameClsDataset
from .hmdb import HMDBVideoClsDataset, HMDBRawFrameClsDataset
# NEW: Import our custom video classification dataset (numeric labels)
from .custom_video_cls_dataset_numeric import CustomVideoClsDatasetNumeric

def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        prefix=args.prefix,
        split=args.split,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        num_segments=args.num_segments,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=args.use_decord,
        lazy_init=False,
        num_sample=args.num_sample)
    print("Data Aug =", transform)
    return dataset

def build_multi_pretraining_dataset(args):
    original_flip = args.flip
    transform = DataAugmentationForVideoMAE(args)
    args.flip = False
    transform_ssv2 = DataAugmentationForVideoMAE(args)
    args.flip = original_flip

    dataset = VideoMAE_multi(
        root=None,
        setting=args.data_path,
        prefix=args.prefix,
        split=args.split,
        is_color=True,
        modality='rgb',
        num_segments=args.num_segments,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        transform_ssv2=transform_ssv2,
        temporal_jitter=False,
        video_loader=True,
        use_decord=args.use_decord,
        lazy_init=False,
        num_sample=args.num_sample)
    print("Data Aug =", transform)
    print("Data Aug for SSV2 =", transform_ssv2)
    return dataset

def build_dataset(is_train, test_mode, args):
    print(f'Use Dataset: {args.data_set}')
    if args.data_set in ['Kinetics', 'Kinetics_sparse', 'mitv1_sparse']:
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f"{mode}.csv")
        if 'sparse' in args.data_set:
            func = VideoClsDataset_sparse
        else:
            func = VideoClsDataset
        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes
    elif args.data_set == 'SSV2':
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f"{mode}.csv")
        if args.use_decord:
            func = SSVideoClsDataset
        else:
            func = SSRawFrameClsDataset
        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            args=args)
        nb_classes = 174
    elif args.data_set == 'UCF101':
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f"{mode}.csv")
        dataset = VideoClsDataset(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    elif args.data_set == 'HMDB51':
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f"{mode}.csv")
        if args.use_decord:
            func = HMDBVideoClsDataset
        else:
            func = HMDBRawFrameClsDataset
        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            args=args)
        nb_classes = 51 if not args.nb_classes else args.nb_classes
    elif args.data_set in ['ANet', 'HACS', 'ANet_interval', 'HACS_interval']:
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f"{mode}.csv")
        if 'interval' in args.data_set:
            func = ANetDataset
        else:
            func = VideoClsDataset_sparse
        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes
    elif args.data_set == 'MyCustom':
        # mode = 'train' if is_train else ('test' if test_mode else 'validation')
        # anno_path = os.path.join(args.data_path, f"{mode}.csv")
        # dataset = CustomVideoClsDatasetNumeric(
        #     anno_path=anno_path,
        #     prefix=args.prefix,
        #     split=args.split,
        #     mode=mode,
        #     clip_len=args.num_frames,
        #     crop_size=args.input_size,
        #     short_side_size=args.short_side_size,
        #     new_height=256,
        #     new_width=320,
        #     filename_tmpl=args.filename_tmpl,
        #     args=args
        # )
        # # nb_classes = len(set(dataset.label_array))
        # nb_classes = 6
        if is_train:
            mode = 'train'
            anno_path = os.path.join(args.data_path, "train.csv")
        else:
            mode = 'test'
            anno_path = os.path.join(args.data_path, "test.csv")
        dataset = CustomVideoClsDatasetNumeric(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            args=args
        )
        nb_classes = 6
    else:
        print(f'Unsupported dataset: {args.data_set}')
        raise NotImplementedError()
    print("Number of classes:", nb_classes)
    return dataset, nb_classes
