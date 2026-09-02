#!/usr/bin/env python
# coding=utf-8
"""Train the region-aware mask model with LPIPS-derived pseudo labels."""

import argparse
import logging
import shutil
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.folder_dataset import Text2ImageDataset
from utils.cal_mask_lpips_var_torch import LPIPSPseudoLabeler
from utils.utils import instantiate_from_config


logger = get_logger(__name__)
MASK_WEIGHT_NAME = "rtdmmodel.pth"


class LossRecorder:
    """Track a moving average and an exponential moving average."""

    def __init__(self, window=1000, gamma=0.9):
        self.losses = deque(maxlen=window)
        self.gamma = gamma
        self.ema = None

    def add(self, loss):
        self.losses.append(loss)
        self.ema = (
            loss
            if self.ema is None
            else self.gamma * self.ema + (1 - self.gamma) * loss
        )

    @property
    def average(self):
        return sum(self.losses) / len(self.losses)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Train the local region-aware mask model."
    )

    data_group = parser.add_argument_group("data and models")
    data_group.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Dataset root containing an image/ subdirectory.",
    )
    data_group.add_argument(
        "--first_stage_model_config",
        type=str,
        required=True,
    )
    data_group.add_argument("--resolution", type=int, default=512)
    data_group.add_argument("--train_batch_size", type=int, default=16)
    data_group.add_argument("--dataloader_num_workers", type=int, default=4)

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--output_dir", type=str, default="region-aware-model")
    training_group.add_argument("--seed", type=int, default=None)
    training_group.add_argument("--max_train_steps", type=int, required=True)
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    training_group.add_argument("--learning_rate", type=float, default=1e-5)
    training_group.add_argument("--scale_lr", action="store_true")
    training_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
    )
    training_group.add_argument("--lr_warmup_steps", type=int, default=500)
    training_group.add_argument("--lr_num_cycles", type=int, default=1)
    training_group.add_argument("--lr_power", type=float, default=1.0)
    training_group.add_argument(
        "--optimizer_type",
        type=str.lower,
        choices=("adamw", "adafactor"),
        default="adamw",
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
    training_group.add_argument("--set_grads_to_none", action="store_true")

    checkpoint_group = parser.add_argument_group("checkpointing")
    checkpoint_group.add_argument("--checkpointing_steps", type=int, default=1000)
    checkpoint_group.add_argument("--checkpoints_total_limit", type=int, default=None)

    args = parser.parse_args(input_args)

    if args.resolution <= 0 or args.resolution % 32 != 0:
        parser.error("--resolution must be positive and divisible by 32.")
    if args.train_batch_size <= 0:
        parser.error("--train_batch_size must be positive.")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader_num_workers cannot be negative.")
    if args.max_train_steps <= 0:
        parser.error("--max_train_steps must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive.")
    if args.checkpointing_steps <= 0:
        parser.error("--checkpointing_steps must be positive.")
    if args.checkpoints_total_limit is not None and args.checkpoints_total_limit <= 0:
        parser.error("--checkpoints_total_limit must be positive.")
    if args.use_8bit_adam and args.optimizer_type != "adamw":
        parser.error("--use_8bit_adam is only valid with --optimizer_type adamw.")

    return args


def select_config(config, primary_name, fallback_name):
    selected = config.get(primary_name)
    if selected is None:
        selected = config.get(fallback_name)
    return selected


def load_checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        return checkpoint["params_ema"]
    if isinstance(checkpoint, dict) and "params" in checkpoint:
        return checkpoint["params"]
    return checkpoint


def load_training_models(config_path, device):
    config = OmegaConf.load(config_path)
    sr_config = select_config(config.model, "SR", "PSR")
    mask_config = select_config(config.model, "MASK", "RTDM")
    sr_path = select_config(config.train, "sr_path", "psr_path")

    if sr_config is None or mask_config is None or sr_path is None:
        raise KeyError(
            "The config must define model.SR/model.PSR, "
            "model.MASK/model.RTDM, and train.sr_path/train.psr_path."
        )

    sr_model = instantiate_from_config(sr_config)
    sr_model.load_state_dict(load_checkpoint_state(sr_path), strict=True)
    sr_model.requires_grad_(False).eval().to(device)

    mask_model = instantiate_from_config(mask_config)
    mask_model.requires_grad_(True).train()

    logger.info(f"Loaded first-stage SR model from {sr_path}")
    return sr_model, mask_model


def build_optimizer(args, parameters):
    parameter_group = [{"params": parameters, "lr": args.learning_rate}]

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

    return optimizer_class(parameter_group, **optimizer_kwargs)


def save_model(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    torch.save(state_dict, output_dir / MASK_WEIGHT_NAME)


def list_checkpoints(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        return []

    checkpoints = []
    for path in checkpoint_root.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    return sorted(checkpoints, key=lambda item: item[0])


def prune_checkpoints(checkpoint_root, total_limit):
    if total_limit is None:
        return

    checkpoints = list_checkpoints(checkpoint_root)
    for _, checkpoint_path in checkpoints[:-total_limit]:
        logger.info(f"Removing old checkpoint: {checkpoint_path}")
        shutil.rmtree(checkpoint_path)


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

    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    if accelerator.is_main_process:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    sr_model, mask_model = load_training_models(
        args.first_stage_model_config,
        accelerator.device,
    )

    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    trainable_parameters = list(mask_model.parameters())
    optimizer = build_optimizer(args, trainable_parameters)

    train_dataset = Text2ImageDataset(
        data_root=args.train_data_dir,
        tokenizers=None,
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

    mask_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        mask_model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    pseudo_labeler = LPIPSPseudoLabeler(
        backbone="vgg",
        device=str(accelerator.device),
    )

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

    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    loss_recorder = LossRecorder()

    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(mask_model):
                ground_truth = (
                    batch["pixel_values"].to(accelerator.device).float() + 1.0
                ) / 2.0
                low_resolution = batch["conditioning_pixel_values"].to(
                    accelerator.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                with torch.no_grad():
                    super_resolved = sr_model(low_resolution).clamp(0, 1)
                    heat_map = pseudo_labeler.compute_heat_map(
                        ground_truth,
                        super_resolved,
                    )

                with accelerator.autocast():
                    predicted_heat_map = mask_model(
                        low_resolution,
                        super_resolved,
                    )
                    loss = F.l1_loss(
                        predicted_heat_map.float(),
                        heat_map.float(),
                        reduction="mean",
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = checkpoint_root / f"checkpoint-{global_step}"
                    if accelerator.is_main_process:
                        save_model(
                            accelerator.unwrap_model(mask_model),
                            checkpoint_dir,
                        )
                        prune_checkpoints(
                            checkpoint_root,
                            args.checkpoints_total_limit,
                        )
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")
                    accelerator.wait_for_everyone()

            scalar_loss = loss.detach().item()
            loss_recorder.add(scalar_loss)
            progress_bar.set_postfix(
                loss=scalar_loss,
                loss_avg=loss_recorder.average,
                loss_ema=loss_recorder.ema,
                lr=lr_scheduler.get_last_lr()[0],
            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = checkpoint_root / "final"
        save_model(
            accelerator.unwrap_model(mask_model),
            final_dir,
        )
        logger.info(f"Saved final weights to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
