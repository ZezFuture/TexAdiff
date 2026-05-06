#!/usr/bin/env python3
import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from scipy import ndimage
from skimage import measure
from torchvision import transforms

from utils import tools_pipeline
from utils.wavelet_color_fix import wavelet_color_fix
from utils.utils import instantiate_from_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_json_by_image_name(image_path: str, json_dir: str) -> dict:
    image_stem = Path(image_path).stem
    json_path = Path(json_dir) / f"{image_stem}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"找不到对应的 JSON 文件: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_to_same_length(images, prompts):
    if len(images) == len(prompts):
        return images, prompts
    if len(images) == 1:
        return images * len(prompts), prompts
    if len(prompts) == 1:
        return images, prompts * len(images)
    raise ValueError(
        "validation_image 和 validation_prompt 的数量不匹配："
        "必须数量相同，或者其中一个数量为 1。"
    )


def expand_optional_list(values, target_len, name):
    if values is None:
        return [None] * target_len

    if len(values) == target_len:
        return values

    if len(values) == 1:
        return values * target_len

    raise ValueError(f"{name} 的数量必须为 1 或等于图片数量。")


def load_first_stage_models(config_path: str, device: str):
    cfg = OmegaConf.load(config_path)

    psr_model = instantiate_from_config(cfg.model.PSR)
    psr_ckpt = torch.load(cfg.train.sr_path, map_location="cpu")
    psr_key = "params_ema" if "params_ema" in psr_ckpt else "params"
    psr_model.load_state_dict(psr_ckpt[psr_key], strict=True)
    print(f"Loaded PSR model from: {cfg.train.sr_path}")

    rtdm_model = instantiate_from_config(cfg.model.RTDM)
    rtdm_ckpt = torch.load(cfg.train.rtdm_path, map_location="cpu")
    rtdm_model.load_state_dict(rtdm_ckpt, strict=True)
    print(f"Loaded RTDM model from: {cfg.train.rtdm_path}")

    psr_model.eval().to(device)
    rtdm_model.eval().to(device)

    return psr_model, rtdm_model


@torch.no_grad()
def sr_model_inference(img, model, upscale: int, window_size: int = 8):
    _, _, h, w = img.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")

    output = model(img)

    if pad_h > 0 or pad_w > 0:
        out_h = output.shape[-2] - pad_h * upscale
        out_w = output.shape[-1] - pad_w * upscale
        output = output[:, :, :out_h, :out_w]

    return output


def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x[None, None, :, :]
    elif x.dim() == 3:
        x = x[:, None, :, :]
    elif x.dim() != 4:
        raise ValueError(f"不支持的 tensor 维度: {x.shape}")

    return x


def postprocess_rtdm(
        pred: torch.Tensor,
        thresh: float = 0.45,
        morph_kernel_size: int = 5,
        min_area: int = 0,
):
    pred = ensure_bchw(pred)
    pred_np = pred.detach().float().cpu().numpy()  # B,C,H,W

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (morph_kernel_size, morph_kernel_size),
    )

    processed = []

    for rtdm in pred_np[:, 0]:
        binary = (rtdm < thresh).astype(np.uint8)

        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=2)

        binary = ndimage.binary_fill_holes(
            binary > 0,
            structure=np.ones((3, 3)),
        ).astype(np.uint8)

        if min_area > 0:
            labels = measure.label(binary, connectivity=2)
            clean = np.zeros_like(binary, dtype=np.uint8)

            for prop in measure.regionprops(labels):
                if prop.area >= int(min_area):
                    clean[labels == prop.label] = 1

            binary = clean

        processed.append(binary)

    processed = np.stack(processed, axis=0)[:, None, :, :]  # B,1,H,W

    return torch.from_numpy(processed).to(
        device=pred.device,
        dtype=pred.dtype,
    )


def downsample_binary_rtdm(rtdm: torch.Tensor, scale_factor: int = 8):

    rtdm = ensure_bchw(rtdm).float()
    return F.max_pool2d(
        rtdm,
        kernel_size=scale_factor,
        stride=scale_factor,
    )


@torch.no_grad()
def build_rtdm(lr_tensor, psr_model, rtdm_model, args):

    sr_image = sr_model_inference(
        img=lr_tensor,
        model=psr_model,
        upscale=args.upscale,
        window_size=8,
    )

    rtdm = rtdm_model(lr_tensor, sr_image)
    rtdm = postprocess_rtdm(
        pred=rtdm,
        thresh=args.thr,
        morph_kernel_size=5,
        min_area=args.min_area,
    )

    rtdm = downsample_binary_rtdm(rtdm, scale_factor=8)
    return rtdm.float()


def load_lr_image(image_path: str, device: str):
    image_path = Path(image_path)
    pil_img = Image.open(image_path).convert("RGB")
    ori_w, ori_h = pil_img.size

    color_ref = pil_img.resize(
        (ori_w * 4, ori_h * 4),
        resample=Image.Resampling.LANCZOS,
    )
    even_w = ori_w // 2 * 2
    even_h = ori_h // 2 * 2
    pil_img = pil_img.resize((even_w, even_h))

    lr_tensor = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)

    return image_path.stem, lr_tensor, color_ref, (ori_w, ori_h)


def run_inference(args, device="cuda"):
    pipeline = tools_pipeline.get_pipeline(
        args.pretrained_model_name_or_path,
        args.unet_model_name_or_path,
        args.controlnet_model_name_or_path,
        vae_model_name_or_path=args.vae_model_name_or_path,
        lora_path=args.lora_path,
        load_weight_increasement=args.load_weight_increasement,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        revision=args.revision,
        variant=args.variant,
        hf_cache_dir=args.hf_cache_dir,
        use_safetensors=args.use_safetensors,
        device=device,
        args=args,
    )
    psr_model, rtdm_model = load_first_stage_models(
        args.first_stage_model_config, device
    )

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    image_paths, prompts = expand_to_same_length(
        args.validation_image, args.validation_prompt
    )
    negative_prompts = expand_optional_list(
        args.negative_prompt,
        target_len=len(image_paths),
        name="negative_prompt",
    )

    save_dir = Path(args.output_dir) / "eval_img"
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, (image_path, prompt, negative_prompt) in enumerate(
        zip(image_paths, prompts, negative_prompts), start=1
    ):
        image_path = str(image_path)

        if args.caption:
            caption_json = read_json_by_image_name(image_path, args.caption)
            prompt = prompt + caption_json["caption"]
            print(f"Prompt with caption: {prompt}")

        filename, lr_tensor, color_ref, (ori_w, ori_h) = load_lr_image(
            image_path, device
        )

        rtdm = build_rtdm(
            lr_tensor=lr_tensor,
            sr_model=psr_model,
            rtdm_model=rtdm_model,
            args=args,
        ).to(device)

        for sample_idx in range(args.num_validation_images):
            height = lr_tensor.shape[-2] * args.upscale
            width = lr_tensor.shape[-1] * args.upscale

            with torch.autocast("cuda"):
                image = pipeline(
                    image=lr_tensor,
                    mask_image=rtdm,
                    prompt=prompt,
                    controlnet_image=lr_tensor,
                    controlnet_scale=args.controlnet_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=args.guidance_scale,
                ).images[0]

            image = image.resize((ori_w * args.upscale, ori_h * args.upscale))

            if args.color_fix:
                image = wavelet_color_fix(image, color_ref)

            if args.num_validation_images == 1:
                save_name = f"{filename}.png"
            else:
                save_name = f"{filename}_{sample_idx}.png"

            image.save(save_dir / save_name)

        print(f"[{idx}/{len(image_paths)}] saved: {filename}")

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()

    # input / output
    parser.add_argument("--validation_image", type=str, nargs="+", required=True)
    parser.add_argument("--validation_prompt", type=str, nargs="+", required=True)
    parser.add_argument("--negative_prompt", type=str, nargs="+", default=None)
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="controlnet-model")

    # model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--unet_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_name_or_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--first_stage_model_config", type=str, required=True)

    # generation
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--num_validation_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=5)
    parser.add_argument("--seed", type=int, default=None)

    # rtdm
    parser.add_argument("--thr", type=float, default=0.45)
    parser.add_argument("--min_area", type=float, default=0)

    # pipeline options
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--load_weight_increasement", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--color_fix", action="store_true")


    parser.add_argument("--latent_tiled_size", type=int, default=180)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)

    args = parser.parse_args()

    if (
            len(args.validation_image) != 1
            and len(args.validation_prompt) != 1
            and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "必须提供：1 张图配多个 prompt，或 1 个 prompt 配多张图，或图片和 prompt 数量相同。"
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    image_paths = []
    save_dir = Path(args.output_dir) / "eval_img"

    for image_dir in args.validation_image:
        image_dir = Path(image_dir)
        image_paths += [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ]
    image_paths = sorted(image_paths)

    if save_dir.exists():
        done = {p.stem for p in save_dir.iterdir()}
        image_paths = [
            p for p in image_paths
            if p.stem not in done
        ]
    print("Number of processing images:", len(image_paths))

    args.validation_image = [str(p) for p in image_paths]
    run_inference(args)