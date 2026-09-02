#!/usr/bin/env python
# coding=utf-8
"""Local training entry point for the TexADiff SDXL refinement model.

The training objective and model data flow are kept from the original script.
Validation, experiment tracking, and remote upload integrations are excluded.
"""

import argparse
import logging
import random
import re
import shutil
from collections import deque
from pathlib import Path

import diffusers
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from omegaconf import OmegaConf
from packaging import version
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from dataset.folder_dataset import Text2ImageDataset
from models.minicontrolnet import ControlNetModel
from models.unet import UNet2DConditionModel
from utils.cal_mask_lpips_var_torch import LPIPSPseudoLabeler
from utils.utils import instantiate_from_config


logger = get_logger(__name__)

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
CONTROLNET_WEIGHT_NAME = "controlnet.safetensors"
UNET_WEIGHT_NAME = "unet.safetensors"


class LossRecorder:
    """Track a bounded moving average and an exponential moving average."""

    def __init__(self, window=1000, gamma=0.9):
        self.losses = deque(maxlen=window)
        self.gamma = gamma
        self.ema = None

    def add(self, loss):
        self.losses.append(loss)
        self.ema = loss if self.ema is None else self.gamma * self.ema + (1 - self.gamma) * loss

    @property
    def average(self):
        return sum(self.losses) / len(self.losses)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Train the local TexADiff SDXL refinement model with an LPIPS-derived mask."
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    model_group.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None)
    model_group.add_argument("--pretrained_unet_model_name_or_path", type=str, default=None)
    model_group.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    model_group.add_argument("--revision", type=str, default=None)
    model_group.add_argument("--use_safetensors", action="store_true")
    model_group.add_argument(
        "--unet_trainable_param_pattern",
        type=str,
        default=r"mid_block|up_blocks\.2|down_blocks\.[02]",
    )
    model_group.add_argument("--controlnet_scale_factor", type=float, default=1.0)

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Dataset root containing image/ and prompt/ subdirectories.",
    )
    data_group.add_argument("--resolution", type=int, default=512)
    data_group.add_argument("--train_batch_size", type=int, default=4)
    data_group.add_argument("--dataloader_num_workers", type=int, default=4)
    data_group.add_argument("--proportion_empty_prompts", type=float, default=0.0)
    data_group.add_argument("--first_stage_model_config", type=str, required=True)
    data_group.add_argument("--thr_start", type=float, default=0.35)
    data_group.add_argument("--thr_end", type=float, default=0.4)

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--output_dir", type=str, default="controlnet-model")
    training_group.add_argument("--seed", type=int, default=None)
    training_group.add_argument("--max_train_steps", type=int, required=True)
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    training_group.add_argument("--gradient_checkpointing", action="store_true")
    training_group.add_argument("--learning_rate", type=float, default=1e-5)
    training_group.add_argument("--learning_rate_controlnet", type=float, default=1e-4)
    training_group.add_argument("--scale_lr", action="store_true")
    training_group.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    training_group.add_argument("--lr_warmup_steps", type=int, default=500)
    training_group.add_argument("--lr_num_cycles", type=int, default=1)
    training_group.add_argument("--lr_power", type=float, default=1.0)
    training_group.add_argument(
        "--optimizer_type",
        type=str.lower,
        choices=("adamw", "adafactor"),
        default="adafactor",
    )
    training_group.add_argument("--use_8bit_adam", action="store_true")
    training_group.add_argument("--adam_beta1", type=float, default=0.9)
    training_group.add_argument("--adam_beta2", type=float, default=0.999)
    training_group.add_argument("--adam_weight_decay", type=float, default=1e-2)
    training_group.add_argument("--adam_epsilon", type=float, default=1e-8)
    training_group.add_argument("--adafactor_relative_step", action="store_true")
    training_group.add_argument("--adafactor_scale_parameter", action="store_true")
    training_group.add_argument("--adafactor_warmup_init", action="store_true")
    training_group.add_argument("--max_grad_norm", type=float, default=1.0)
    training_group.add_argument("--allow_tf32", action="store_true")
    training_group.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=("no", "fp16", "bf16"),
    )
    training_group.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    training_group.add_argument("--enable_npu_flash_attention", action="store_true")
    training_group.add_argument("--set_grads_to_none", action="store_true")

    checkpoint_group = parser.add_argument_group("checkpointing")
    checkpoint_group.add_argument("--checkpointing_steps", type=int, default=1000)
    checkpoint_group.add_argument("--checkpoints_total_limit", type=int, default=None)

    args = parser.parse_args(input_args)

    if not 0.0 <= args.proportion_empty_prompts <= 1.0:
        parser.error("--proportion_empty_prompts must be in [0, 1].")
    if not 0.0 <= args.thr_start <= args.thr_end <= 1.0:
        parser.error("--thr_start and --thr_end must satisfy 0 <= start <= end <= 1.")
    if args.resolution <= 0 or args.resolution % 8 != 0:
        parser.error("--resolution must be positive and divisible by 8.")
    if args.train_batch_size <= 0:
        parser.error("--train_batch_size must be positive.")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader_num_workers cannot be negative.")
    if args.max_train_steps <= 0:
        parser.error("--max_train_steps must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive.")
    if args.checkpointing_steps <= 0:
        parser.error("--checkpointing_steps must be positive.")
    if args.checkpoints_total_limit is not None and args.checkpoints_total_limit <= 0:
        parser.error("--checkpoints_total_limit must be positive.")
    if args.use_8bit_adam and args.optimizer_type != "adamw":
        parser.error("--use_8bit_adam is only valid with --optimizer_type adamw.")

    return args


def import_text_encoder_class(model_name_or_path, revision, subfolder):
    config = PretrainedConfig.from_pretrained(
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"Unsupported text encoder class: {model_class}")


def downsample_binary_tensor(tensor, scale_factor=8):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError("Mask tensor must have shape (B,H,W) or (B,C,H,W).")

    pooled = F.max_pool2d(
        tensor.float(),
        kernel_size=scale_factor,
        stride=scale_factor,
    )
    return (pooled > 0).to(tensor.dtype)


def load_partial_unet_weights(unet, weight_path):
    incoming_state = load_file(weight_path)
    current_state = unet.state_dict()
    loaded_keys = 0

    with torch.no_grad():
        for name, incoming_value in incoming_state.items():
            if name not in current_state:
                logger.warning(f"Ignoring U-Net key not found in the base model: {name}")
                continue

            target_value = current_state[name]
            if target_value.shape != incoming_value.shape:
                raise ValueError(
                    f"Shape mismatch for U-Net key {name}: "
                    f"expected {tuple(target_value.shape)}, got {tuple(incoming_value.shape)}"
                )

            incoming_value = incoming_value.to(
                device=target_value.device,
                dtype=target_value.dtype,
            )
            target_value.copy_(incoming_value)
            loaded_keys += 1

    logger.info(f"Loaded {loaded_keys} partial U-Net tensors from {weight_path}")


def save_models(
    unet,
    controlnet,
    output_dir,
    unet_pattern,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unet_state = {
        name: value.detach().cpu().contiguous()
        for name, value in unet.state_dict().items()
        if unet_pattern.search(name)
    }

    controlnet_state = {
        name: value.detach().cpu().contiguous()
        for name, value in controlnet.state_dict().items()
    }
    save_file(unet_state, str(output_dir / UNET_WEIGHT_NAME))
    save_file(controlnet_state, str(output_dir / CONTROLNET_WEIGHT_NAME))


def list_checkpoints(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    checkpoints = []
    if not checkpoint_root.exists():
        return checkpoints

    for path in checkpoint_root.iterdir():
        match = CHECKPOINT_PATTERN.match(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints, key=lambda item: item[0])


def prune_checkpoints(checkpoint_root, total_limit):
    if total_limit is None:
        return

    checkpoints = list_checkpoints(checkpoint_root)
    for _, checkpoint_path in checkpoints[:-total_limit]:
        logger.info(f"Removing old checkpoint: {checkpoint_path}")
        shutil.rmtree(checkpoint_path)


def load_first_stage_model(config_path, device):
    config = OmegaConf.load(config_path)
    sr_config = config.model.get("SR")
    if sr_config is None:
        sr_config = config.model.get("PSR")

    sr_path = config.train.get("sr_path")
    if sr_path is None:
        sr_path = config.train.get("psr_path")

    if sr_config is None or sr_path is None:
        raise KeyError(
            "First-stage config must define model.SR/model.PSR and train.sr_path/train.psr_path."
        )

    sr_model = instantiate_from_config(sr_config)
    checkpoint = torch.load(sr_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif isinstance(checkpoint, dict) and "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    sr_model.load_state_dict(state_dict, strict=True)
    sr_model.requires_grad_(False)
    sr_model.eval().to(device)
    logger.info(f"Loaded first-stage SR model from {sr_path}")
    return sr_model


def load_sdxl_components(args):
    text_encoder_class_one = import_text_encoder_class(
        args.pretrained_model_name_or_path,
        args.revision,
        "text_encoder",
    )
    text_encoder_class_two = import_text_encoder_class(
        args.pretrained_model_name_or_path,
        args.revision,
        "text_encoder_2",
    )

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    text_encoder_one = text_encoder_class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_two = text_encoder_class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )

    vae_path = (
        args.pretrained_vae_model_name_or_path
        if args.pretrained_vae_model_name_or_path is not None
        else args.pretrained_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder=None if args.pretrained_vae_model_name_or_path else "vae",
        revision=None if args.pretrained_vae_model_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained_orig(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        use_safetensors=args.use_safetensors,
    )

    return (
        noise_scheduler,
        (tokenizer_one, tokenizer_two),
        (text_encoder_one, text_encoder_two),
        vae,
        unet,
    )


def configure_optional_acceleration(unet, controlnet, args):
    if args.enable_npu_flash_attention:
        if not is_torch_npu_available():
            raise ValueError("NPU flash attention requires torch_npu.")
        unet.enable_npu_flash_attention()

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xFormers is not installed.")

        import xformers

        if version.parse(xformers.__version__) == version.parse("0.0.16"):
            logger.warning("xFormers 0.0.16 is unstable for training; use at least 0.0.17.")
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()


def build_optimizer(args, parameter_groups):
    if args.optimizer_type == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as error:
                raise ImportError(
                    "Install bitsandbytes to use --use_8bit_adam."
                ) from error
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "weight_decay": args.adam_weight_decay,
            "eps": args.adam_epsilon,
        }
    else:
        optimizer_class = transformers.optimization.Adafactor
        optimizer_kwargs = {
            "relative_step": args.adafactor_relative_step,
            "scale_parameter": args.adafactor_scale_parameter,
            "warmup_init": args.adafactor_warmup_init,
        }

    return optimizer_class(parameter_groups, **optimizer_kwargs)


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("MPS does not support bf16 mixed-precision training.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if is_torch_npu_available():
        torch.npu.config.allow_internal_format = False
    if args.seed is not None:
        set_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    if accelerator.is_main_process:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    sr_model = load_first_stage_model(
        args.first_stage_model_config,
        accelerator.device,
    )
    (
        noise_scheduler,
        tokenizers,
        text_encoders,
        vae,
        unet,
    ) = load_sdxl_components(args)

    unet_pattern = re.compile(args.unet_trainable_param_pattern)
    if args.pretrained_unet_model_name_or_path is not None:
        load_partial_unet_weights(
            unet,
            args.pretrained_unet_model_name_or_path,
        )

    controlnet = ControlNetModel()
    if args.controlnet_model_name_or_path is not None:
        incompatible = controlnet.load_state_dict(
            load_file(args.controlnet_model_name_or_path),
            strict=False,
        )
        if incompatible.missing_keys:
            logger.warning(f"Missing ControlNet keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected ControlNet keys: {incompatible.unexpected_keys}")

    vae.requires_grad_(False).eval()
    text_encoders[0].requires_grad_(False).eval()
    text_encoders[1].requires_grad_(False).eval()

    controlnet.requires_grad_(True).train()
    unet.requires_grad_(False).train()
    unet_trainable_parameters = []
    for name, parameter in unet.named_parameters():
        if unet_pattern.search(name):
            parameter.requires_grad_(True)
            unet_trainable_parameters.append(parameter)

    if not unet_trainable_parameters:
        raise ValueError(
            "--unet_trainable_param_pattern did not match any U-Net parameters."
        )

    configure_optional_acceleration(unet, controlnet, args)
    if next(controlnet.parameters()).dtype != torch.float32:
        raise ValueError("ControlNet must start training in float32.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        lr_scale = (
            args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        args.learning_rate *= lr_scale
        args.learning_rate_controlnet *= lr_scale

    parameter_groups = [
        {
            "params": list(controlnet.parameters()),
            "lr": args.learning_rate_controlnet,
        },
        {
            "params": unet_trainable_parameters,
            "lr": args.learning_rate,
        },
    ]
    optimizer = build_optimizer(args, parameter_groups)
    trainable_parameters = [
        parameter
        for group in parameter_groups
        for parameter in group["params"]
    ]

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae_dtype = (
        weight_dtype
        if args.pretrained_vae_model_name_or_path is not None
        else torch.float32
    )
    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoders[0].to(accelerator.device, dtype=weight_dtype)
    text_encoders[1].to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=torch.float32)

    pseudo_labeler = LPIPSPseudoLabeler(
        thr=args.thr_start,
        min_area=int(args.resolution * args.resolution * 0.005),
        dilate_iter=2,
        erode_iter=1,
        thin=False,
        backbone="vgg",
        device=str(accelerator.device),
    )

    train_dataset = Text2ImageDataset(
        data_root=args.train_data_dir,
        tokenizers=tokenizers,
        null_text_ratio=args.proportion_empty_prompts,
        crop_size=args.resolution,
        resize_bak=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.dataloader_num_workers > 0,
        drop_last=True,
    )
    
    if len(train_dataloader) == 0:
        raise ValueError(
            "The dataset must contain at least one complete training batch."
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    (
        unet,
        controlnet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        controlnet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    loss_recorder = LossRecorder(window=1000, gamma=0.9)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Optimization steps = {args.max_train_steps}")
    logger.info(f"  Mask threshold range = [{args.thr_start}, {args.thr_end}]")

    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(unet, controlnet):
                pixel_values = batch["pixel_values"]
                text_input_ids = batch["text_input_ids"]
                text_input_ids_2 = batch["text_input_ids_2"]
                controlnet_image = batch["conditioning_pixel_values"]
                original_size = batch["original_size"]
                crop_coords_top_left = batch["crop_coords_top_left"]
                target_size = batch["target_size"]

                pixel_values = pixel_values.to(accelerator.device)
                controlnet_image = controlnet_image.to(
                    accelerator.device,
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    gt_values = (pixel_values.float() + 1.0) / 2.0
                    sr_image = sr_model(controlnet_image).clamp(0, 1)
                    threshold = random.uniform(args.thr_start, args.thr_end)
                    _, masks = pseudo_labeler(
                        gt_values,
                        sr_image,
                        thr=threshold,
                    )
                    masks = downsample_binary_tensor(masks, 8).float()
                del gt_values, sr_image

                with torch.no_grad():
                    encoded = vae.encode(
                        pixel_values.to(dtype=vae_dtype, non_blocking=True)
                    )
                    latents = (
                        encoded.latent_dist.sample()
                        * vae.config.scaling_factor
                    ).to(weight_dtype)

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(
                    latents.float(),
                    noise.float(),
                    timesteps,
                )

                with torch.no_grad():
                    encoder_output_one = text_encoders[0](
                        text_input_ids.to(accelerator.device),
                        output_hidden_states=True,
                    )
                    encoder_output_two = text_encoders[1](
                        text_input_ids_2.to(accelerator.device),
                        output_hidden_states=True,
                    )
                    text_embeds = torch.cat(
                        (
                            encoder_output_one.hidden_states[-2],
                            encoder_output_two.hidden_states[-2],
                        ),
                        dim=-1,
                    )
                    pooled_text_embeds = encoder_output_two[0]

                add_time_ids = torch.cat(
                    (
                        original_size.to(accelerator.device),
                        crop_coords_top_left.to(accelerator.device),
                        target_size.to(accelerator.device),
                    ),
                    dim=1,
                ).to(dtype=weight_dtype)
                added_conditions = {
                    "text_embeds": pooled_text_embeds,
                    "time_ids": add_time_ids,
                }

                controls = controlnet(
                    controlnet_image,
                    masks,
                    noisy_latents,
                    timesteps,
                )
                controls["scale"] = (
                    controls["scale"] * args.controlnet_scale_factor
                )

                with accelerator.autocast():
                    model_prediction = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeds,
                        added_cond_kwargs=added_conditions,
                        controls=controls,
                        return_dict=False,
                    )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents,
                        noise,
                        timesteps,
                    )
                else:
                    raise ValueError(
                        "Unsupported prediction type: "
                        f"{noise_scheduler.config.prediction_type}"
                    )

                loss_map = F.mse_loss(
                    model_prediction.float(),
                    target.float(),
                    reduction="none",
                )
                loss = loss_map.mean() + (loss_map * masks).mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(
                    set_to_none=args.set_grads_to_none
                )

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = checkpoint_root / f"checkpoint-{global_step}"
                    if accelerator.is_main_process:
                        save_models(
                            accelerator.unwrap_model(unet),
                            accelerator.unwrap_model(controlnet),
                            checkpoint_dir,
                            unet_pattern,
                        )
                        prune_checkpoints(
                            checkpoint_root,
                            args.checkpoints_total_limit,
                        )
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")
                    accelerator.wait_for_everyone()

            scalar_loss = loss.detach().item()
            loss_recorder.add(scalar_loss)
            learning_rates = lr_scheduler.get_last_lr()
            progress_bar.set_postfix(
                loss=scalar_loss,
                loss_avg=loss_recorder.average,
                loss_ema=loss_recorder.ema,
                controlnet_lr=learning_rates[0],
                unet_lr=learning_rates[1],
            )

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = checkpoint_root / "final"
        save_models(
            accelerator.unwrap_model(unet),
            accelerator.unwrap_model(controlnet),
            final_dir,
            unet_pattern,
        )
        logger.info(f"Saved final weights to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
