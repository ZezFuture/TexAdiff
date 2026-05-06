from typing import List, Optional, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def conv_gn_silu(
    in_channels: int,
    out_channels: int,
    norm_groups: int,
    stride: int = 1,
) -> list[nn.Module]:
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        ),
        nn.GroupNorm(norm_groups, out_channels),
        nn.SiLU(),
    ]


class SpatialFeatureTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int):
        super().__init__()

        self.compress = nn.Conv2d(
            in_channels + in_channels // 2,
            out_channels,
            kernel_size=1,
        )

        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.gamma = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.compress(torch.cat([x1, x2], dim=1))

        cond_feat = self.shared(cond)
        gamma = self.gamma(cond_feat)
        beta = self.beta(cond_feat)

        return gamma * x + beta


class ControlNetModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: List[int] = [256, 256],
        out_channels: List[int] = [256, 512],
        groups: List[int] = [4, 8],
        time_embed_dim: int = 256,
        final_out_channels: int = 320,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        act_fn: str = "silu",
    ):
        super().__init__()

        self.time_proj = Timesteps(128, True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(128, time_embed_dim)

        self.image_embedding = nn.Sequential(
            *conv_gn_silu(3, 64, norm_groups=2, stride=1),
            *conv_gn_silu(64, 128, norm_groups=2, stride=1),
            *conv_gn_silu(128, 256, norm_groups=2, stride=2),
        )

        self.mask_embedding = nn.Sequential(
            *conv_gn_silu(1, 32, norm_groups=1, stride=1),
            *conv_gn_silu(32, 64, norm_groups=1, stride=1),
            *conv_gn_silu(64, 128, norm_groups=1, stride=1),
        )

        self.z_embedding = nn.Sequential(
            *conv_gn_silu(4, 64, norm_groups=1, stride=1),
            *conv_gn_silu(64, 128, norm_groups=1, stride=1),
        )

        self.image_res = nn.ModuleList()
        self.mask_res = nn.ModuleList()
        self.z_res = nn.ModuleList()
        self.sft = nn.ModuleList()

        for in_ch, out_ch, group in zip(in_channels, out_channels, groups):
            self.image_res.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=time_embed_dim,
                    groups=group,
                )
            )

            self.mask_res.append(
                ResnetBlock2D(
                    in_channels=in_ch // 2,
                    out_channels=out_ch // 2,
                    temb_channels=time_embed_dim,
                    groups=group // 2,
                )
            )

            self.z_res.append(
                ResnetBlock2D(
                    in_channels=in_ch // 2,
                    out_channels=out_ch // 2,
                    temb_channels=time_embed_dim,
                    groups=group // 2,
                )
            )

            self.sft.append(
                SpatialFeatureTransform(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    cond_channels=out_ch // 2,
                )
            )

        self.mid_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        out_channels[-1],
                        out_channels[-1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.SiLU(),
                    nn.GroupNorm(8, out_channels[-1]),
                    nn.Conv2d(
                        out_channels[-1],
                        out_channels[-1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GroupNorm(8, out_channels[-1]),
                ),
                nn.Conv2d(
                    out_channels[-1],
                    final_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ]
        )

        self.controlnet_mid_block = zero_module(
            nn.Conv2d(
                final_out_channels,
                final_out_channels,
                kernel_size=1,
            )
        )

        self.scale = 1.0

    def _set_gradient_checkpointing(self, module, value: bool = False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_forward_chunking(
        self,
        chunk_size: Optional[int] = None,
        dim: int = 0,
    ) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        chunk_size = chunk_size or 1

        def apply_chunking(module: nn.Module):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                apply_chunking(child)

        for module in self.children():
            apply_chunking(module)

    def get_time_embedding(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            is_mps = sample.device.type == "mps"

            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64

            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)

        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)

        batch_size = sample.shape[0]
        timestep = timestep.expand(batch_size)

        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)

        return self.time_embedding(t_emb)

    def forward(
        self,
        sample: torch.FloatTensor,
        mask: torch.FloatTensor,
        z: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        is_pretrain: bool = False,
    ):
        emb = self.get_time_embedding(sample, timestep)

        sample = self.image_embedding(sample)
        mask = self.mask_embedding(mask)
        z = self.z_embedding(z)

        for image_res, mask_res, z_res, sft in zip(
            self.image_res,
            self.mask_res,
            self.z_res,
            self.sft,
        ):
            sample = image_res(sample, emb)
            mask = mask_res(mask, emb)
            z = z_res(z, emb)
            sample = sft(sample, z, mask)

        sample = self.mid_convs[0](sample) + sample
        sample = self.mid_convs[1](sample)
        sample = self.controlnet_mid_block(sample)

        return {
            "out": sample,
            "scale": self.scale,
        }