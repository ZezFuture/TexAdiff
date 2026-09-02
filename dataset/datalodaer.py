import cv2
import gc
import math
import random
import itertools
import torch
from torch import nn
from typing import List, Union
from braceexpand import braceexpand
from functools import partial
import webdataset as wds
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import default_collate
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
from torchvision.transforms import ToTensor
from .realesrgan import RealESRGAN_degradation
from utils.img_utils import exists
from utils.img_utils import convert_image_to_fn
import warnings
def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        # Skip if not a dictionary
        if not isinstance(filesample, dict):
            warning_msg = f"Expected dict, got {type(filesample)}"
            warnings.warn(warning_msg)
            continue

        # Safely get fname and value
        fname = filesample.get("fname")
        value = filesample.get("data")

        # Skip if missing required fields
        if fname is None or value is None:
            warning_msg = f"Sample missing 'fname' or 'data': {filesample.keys()}"
            warnings.warn(warning_msg)
            continue
        try:
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
            #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
            #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
            if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value

        except Exception as e:
            warning_msg = f"Error processing sample {fname}: {str(e)}"
            warnings.warn(warning_msg)
            continue

    if valid_sample(current_sample):
        yield current_sample
# def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
#     """Return function over iterator that groups key, value pairs into samples.
#
#     :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
#     lower case (Default value = True)
#     """
#     current_sample = None
#     for filesample in data:
#         assert isinstance(filesample, dict)
#         fname, value = filesample["fname"], filesample["data"]
#         prefix, suffix = keys(fname)
#         if prefix is None:
#             continue
#         if lcase:
#             suffix = suffix.lower()
#         # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
#         #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
#         #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
#         if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
#             if valid_sample(current_sample):
#                 yield current_sample
#             current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
#         if suffixes is None or suffix in suffixes:
#             current_sample[suffix] = value
#     if valid_sample(current_sample):
#         yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    gc.collect()
    torch.cuda.empty_cache()
    return samples


class Text2ImageDataset:
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            shuffle_buffer_size: int = 1000,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            tokenizers=None,
            null_text_ratio: float = 0.1,
            convert_image_to: str = "RGB",
            center_crop: bool = False,
            random_flip: bool = True,
            resize_bak: bool = False,
            crop_size: int = 512,
    ):
        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        degradation = RealESRGAN_degradation('params_realesrgan.yml', device='cpu')

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        preproc = transforms.Compose([
            transforms.Lambda(maybe_convert_fn),
            #transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),

        ])
        # img_preproc = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])

        def transform(example):

            image = preproc(example["image"])

            cropped_img, top_left, size = record_crop_coords(image, crop_size=crop_size, is_center_crop=center_crop)
            example["original_size"] = torch.tensor(size)
            example["crop_coords_top_left"] = torch.tensor(top_left)
            example["target_size"] = torch.tensor([crop_size,crop_size])

            GT_image_t, LR_image_t = degradation.degrade_process(np.asarray(cropped_img) / 255.,
                                                                 resize_bak=resize_bak)

            example["conditioning_pixel_values"] = LR_image_t.squeeze(0)
            example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0

            caption = example['text'] if 'text' in example else ''
            if tokenizers is not None:
                example["text_input_ids"] = tokenize_caption(caption, tokenizers[0], null_text_ratio).squeeze(0)
                example["text_input_ids_2"] = tokenize_caption(caption, tokenizers[1], null_text_ratio).squeeze(0)
            else:
                example["text_input_ids"] = ""
                example["text_input_ids_2"] = ""

            del cropped_img, top_left, size, GT_image_t, LR_image_t, caption
            return example

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=wds.warn_and_continue),
            wds.map(filter_keys({"image", "text"})),
            wds.map(transform),
            wds.to_tuple("pixel_values", "text", "text_input_ids", "text_input_ids_2",
                         "conditioning_pixel_values","original_size","crop_coords_top_left","target_size"),
        ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker #3181
        num_batches = num_worker_batches * num_workers # 3181*4=12724
        num_samples = num_batches * global_batch_size # 305376

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    for sample in samples:
        try:
            for key in required_keys:
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break


key_verifier = wds.filters.pipelinefilter(verify_keys)


def tokenize_caption(caption, tokenizer, null_text_ratio):
    if random.random() < null_text_ratio:
        caption = ""

    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )

    return inputs.input_ids


def rename(filename):
    name = ''.join(random.sample('abcdefghigklmnopqrstuvwxyz', 5))
    return f'{name}_{filename}'


def tarfile_samples(src, handler=wds.handlers.reraise_exception, select_files=None, rename_files=rename):
    streams = wds.tariterators.url_opener(src, handler=handler)
    files = wds.tariterators.tar_file_expander(
        streams, handler=handler, select_files=select_files, rename_files=rename_files
    )
    samples = wds.tariterators.group_by_keys(files, handler=handler)
    return samples


def record_crop_coords(img, crop_size, is_center_crop=True):
    """裁剪图像并返回 (裁剪后图像, (top, left))"""
    width, height = img.size
    if is_center_crop:
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
    else:
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
    cropped_img = img.crop((left, top, left + crop_size, top + crop_size))
    return cropped_img, [top, left], [height, width]


def build_crop_transform(crop_size, center_crop=False):
    def transform_fn(img):
        cropped_images, crop_coords, img_sizes, cropped_sizes = [], [], [], []
        img, coords, img_size = record_crop_coords(img, crop_size, is_center_crop=center_crop)
        cropped_images.append(img)
        crop_coords.append(coords)
        img_sizes.append(img_size)
        examples = {"image": cropped_images, "crop_coords": crop_coords, "img_sizes": img_sizes, }
        return examples

    return transform_fn


tarfile_to_samples = wds.filters.pipelinefilter(tarfile_samples)
