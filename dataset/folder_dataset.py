"""Folder-based image and prompt dataset for TexADiff training."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .realesrgan import RealESRGAN_degradation


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class Text2ImageDataset(Dataset):
    """Read matching images and text prompts from a two-folder dataset.

    Expected layout for text-conditioned training:

        data_root/
            image/
                example.png
            prompt/
                example.txt

    Nested image directories are supported when the prompt directory mirrors
    the same relative layout.
    """

    def __init__(
        self,
        data_root,
        tokenizers,
        null_text_ratio=0.0,
        crop_size=512,
        center_crop=False,
        resize_bak=False,
    ):
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / "image"
        self.prompt_dir = self.data_root / "prompt"
        self.tokenizers = None if tokenizers is None else tuple(tokenizers)
        self.null_text_ratio = null_text_ratio
        self.crop_size = crop_size
        self.center_crop = center_crop
        self.resize_bak = resize_bak

        if self.tokenizers is not None and len(self.tokenizers) != 2:
            raise ValueError("Text2ImageDataset requires the two SDXL tokenizers.")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if self.tokenizers is not None and not self.prompt_dir.is_dir():
            raise FileNotFoundError(f"Prompt directory does not exist: {self.prompt_dir}")

        image_paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No supported images found in {self.image_dir}")

        samples = []
        missing_prompts = []
        for image_path in image_paths:
            if self.tokenizers is None:
                samples.append((image_path, None))
                continue

            relative_prompt = image_path.relative_to(self.image_dir).with_suffix(".txt")
            prompt_path = self.prompt_dir / relative_prompt

            # Also support a flat prompt directory for nested image folders.
            if not prompt_path.is_file():
                flat_prompt_path = self.prompt_dir / f"{image_path.stem}.txt"
                prompt_path = flat_prompt_path if flat_prompt_path.is_file() else prompt_path

            if not prompt_path.is_file():
                missing_prompts.append(prompt_path)
                continue
            samples.append((image_path, prompt_path))

        if missing_prompts:
            examples = "\n".join(str(path) for path in missing_prompts[:5])
            raise FileNotFoundError(
                f"{len(missing_prompts)} images have no matching .txt prompt. "
                f"First missing paths:\n{examples}"
            )

        if not samples:
            raise ValueError("The configured training sample set is empty.")

        self.samples = samples
        self.degradation = RealESRGAN_degradation(
            "params_realesrgan.yml",
            device="cpu",
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, prompt_path = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        cropped_image, crop_coords, original_size = crop_image(
            image,
            crop_size=self.crop_size,
            center_crop=self.center_crop,
        )
        ground_truth, low_resolution = self.degradation.degrade_process(
            np.asarray(cropped_image, dtype=np.float32) / 255.0,
            resize_bak=self.resize_bak,
        )

        sample = {
            "pixel_values": ground_truth * 2.0 - 1.0,
            "conditioning_pixel_values": low_resolution,
            "original_size": torch.tensor(original_size, dtype=torch.long),
            "crop_coords_top_left": torch.tensor(crop_coords, dtype=torch.long),
            "target_size": torch.tensor(
                [self.crop_size, self.crop_size],
                dtype=torch.long,
            ),
        }
        if self.tokenizers is not None:
            caption = prompt_path.read_text(encoding="utf-8").strip()
            if random.random() < self.null_text_ratio:
                caption = ""
            sample["text_input_ids"] = tokenize_caption(
                caption,
                self.tokenizers[0],
            )
            sample["text_input_ids_2"] = tokenize_caption(
                caption,
                self.tokenizers[1],
            )
        return sample


def tokenize_caption(caption, tokenizer):
    inputs = tokenizer(
        caption,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids.squeeze(0)


def crop_image(image, crop_size, center_crop=False):
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Image size {width}x{height} is smaller than crop size {crop_size}."
        )

    if center_crop:
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
    else:
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)

    cropped_image = image.crop(
        (left, top, left + crop_size, top + crop_size)
    )
    return cropped_image, [top, left], [height, width]
