# %%
"""
DDPM Model-based U-net

https://arxiv.org/abs/2006.11239.pdf
"""

import einops
import matplotlib.pyplot as plt
import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

from arena.cv.diffusion.diffusion import DiffusionModel
from arena.mintorch.nn.activations import SiLU
from arena.mintorch.nn.attention import SelfAttention2d
from arena.mintorch.nn.encoding import IntSinusoidalPositionalEncoding
from arena.mintorch.nn.normalization import GroupNorm2d

# %%


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, groups: int = 1):
        self.channels = channels
        self.num_heads = num_heads
        self.groups = groups

        super().__init__()

        self.group_norm = GroupNorm2d(groups, channels)
        self.self_attn = SelfAttention2d(channels, num_heads)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x + self.self_attn(self.group_norm(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        t_emb_dim: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 4,
        include_conv1x1: bool = False,
    ) -> None:
        self.c_in = c_in
        self.c_out = c_out
        self.t_emb_dim = t_emb_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups = min(groups, c_out)  # TODO: What is going on here?
        self.include_conv1x1 = include_conv1x1

        super().__init__()

        self.num_steps_block = nn.Sequential(SiLU(), nn.Linear(t_emb_dim, c_out))
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding),
            GroupNorm2d(groups, c_out),
            SiLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=padding),
            GroupNorm2d(groups, c_out),
            nn.SiLU(),
        )

        if include_conv1x1:
            self.conv1x1 = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(
        self,
        x: TensorType["b", "c_in", "h", "w"],
        num_steps: TensorType["b", "emb"],
    ) -> TensorType["b", "c_out", "h", "w"]:
        time_embed = einops.rearrange(
            self.num_steps_block(num_steps), "b c_out -> b c_out 1 1"
        )
        y = self.conv_block_1(x)
        y = self.conv_block_2(y + time_embed)

        if self.include_conv1x1:
            y += self.conv1x1(x)

        return y


class DownBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int | None = None,
        downsample: bool = True,
        t_emb_dim: int = 2,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        groups: int = 4,
    ) -> None:
        self.c_in = c_in
        self.c_out = c_out = c_out or c_in // 2
        self.downsample = downsample
        self.t_emb_dim = t_emb_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        super().__init__()

        self.embed_num_steps = nn.Embedding(2, t_emb_dim)
        self.resid_block_1 = ResidualBlock(c_in, c_out, t_emb_dim, groups=groups)
        self.resid_block_2 = ResidualBlock(c_out, c_out, t_emb_dim, groups=groups)
        self.attn_block = AttentionBlock(c_in)

        if downsample:
            self.conv = nn.Conv2d(
                c_in,
                c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

    def forward(
        self, x: TensorType["b", "c_in", "h", "w"], num_steps: TensorType["b", "emb"]
    ) -> tuple[
        TensorType["b", "c_out", "h/2", "w/2"], TensorType["b", "c_out", "h", "w"]
    ]:
        x = self.resid_block_1(x, num_steps)
        x = self.resid_block_2(x, num_steps)
        x = self.attn_block(x)

        if self.downsample:
            x_downsampled = self.conv(x)
        else:
            x_downsampled = x.clone()

        return x_downsampled, x


class MidBlock(nn.Module):
    def __init__(
        self,
        c: int,
        t_emb_dim: int = 2,
        groups: int = 4,
    ) -> None:
        self.c = c
        self.t_emb_dim = t_emb_dim
        self.groups = groups

        super().__init__()

        self.resid_block_1 = ResidualBlock(c, t_emb_dim, groups=groups)
        self.attn_block = AttentionBlock(c)
        self.resid_block_2 = ResidualBlock(c, t_emb_dim, groups=groups)

    def forward(
        self, x: TensorType["b", "c_mid", "h"], num_steps: TensorType["b", "emb"]
    ) -> TensorType["b", "c_mid", "h"]:

        x = self.resid_block_1(x, num_steps)
        x = self.attn_block(x)
        x = self.resid_block_2(x, num_steps)

        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int | None = None,
        upsample: bool = True,
        t_emb_dim: int = 2,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        groups: int = 4,
    ) -> None:
        self.c_in = c_in
        self.c_out = c_out = c_out or c_in * 2

        self.upsample = upsample
        self.t_emb_dim = t_emb_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        super().__init__()

        self.resid_block_1 = ResidualBlock(c_in, t_emb_dim, groups=groups)
        self.resid_block_2 = ResidualBlock(c_in, t_emb_dim, groups=groups)
        self.attn_block = AttentionBlock(c_in)

        if upsample:
            self.conv = nn.ConvTranspose2d(
                c_in,
                c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

    def forward(
        self,
        x: TensorType["b", "c_out", "h", "w"],
        x_skip: TensorType["b", "c_in", "h", "w"],
        num_steps: TensorType["b", "emb"],
    ) -> TensorType["b", "c_out", "2h", "2w"]:
        x_concat = t.cat([x, x_skip], dim=1)
        x_concat = self.resid_block_1(x_concat, num_steps)
        x_concat = self.resid_block_2(x_concat, num_steps)
        x_concat = self.attn_block(x_concat)

        if self.upsample:
            x = self.conv(x)

        return x


class NumNoiseStepsEncoding(nn.Module):
    def __init__(self, t_emb_dim: int, max_steps: int) -> None:
        self.t_emb_dim = t_emb_dim
        self.max_steps = max_steps
        super().__init__()

        self.embedding = IntSinusoidalPositionalEncoding(max_steps, max_steps)
        self.linear_1 = nn.Linear(max_steps, t_emb_dim)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(t_emb_dim, t_emb_dim)

    def forward(self, num_steps: TensorType["b"]) -> TensorType["b", "emb"]:
        x = self.embedding(num_steps)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)

        return x


class DDPM(DiffusionModel):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        channels: int = 128,
        dim_mults: tuple[int, ...] = (1, 1, 2, 4, 8),
        groups: int = 4,
        max_steps: int = 1000,
    ):
        self.noise_schedule = None
        self.image_shape = image_shape
        self.channels = channels
        self.dim_mults = dim_mults
        self.groups = groups
        self.max_steps = max_steps

        super().__init__()

        self.num_down_blocks = len(dim_mults)
        self.num_up_blocks = len(dim_mults) - 1

        t_emb_dim = 4 * channels
        self.embed_num_steps = NumNoiseStepsEncoding(t_emb_dim, max_steps)

        self.conv_1 = nn.Conv2d(
            image_shape[0], channels, kernel_size=7, stride=1, padding=3
        )

        channels_list = tuple(channels * d for d in dim_mults)
        in_channels_down = (channels,) + channels_list[:-1]
        out_channels_down = channels_list

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    in_channels,
                    out_channels,
                    downsample=(i < self.num_down_blocks - 1),
                    groups=groups,
                    t_emb_dim=t_emb_dim,
                )
                for i, (in_channels, out_channels) in enumerate(
                    zip(in_channels_down, out_channels_down)
                )
            ]
        )

        self.mid_block = MidBlock(
            out_channels_down[-1],
            t_emb_dim=t_emb_dim,
        )

        in_channels_up = channels_list[-1:0:-1]
        out_channels_up = channels_list[-2::-1]

        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    in_channels,
                    out_channels,
                    groups=groups,
                )
                for i, (in_channels, out_channels) in enumerate(
                    zip(in_channels_up, out_channels_up)
                )
            ]
        )

        self.up_block_1 = UpBlock(channels * 4, channels * 2, t_emb_dim=t_emb_dim)
        self.up_block_2 = UpBlock(channels * 2, channels, t_emb_dim=t_emb_dim)

        self.resid_block = ResidualBlock(channels, channels, t_emb_dim=t_emb_dim)
        self.conv_2 = nn.Conv2d(
            channels, self.image_shape[0], kernel_size=1, stride=1, padding=0
        )

    def forward(
        self,
        x: TensorType["b", "c", "h", "w"],
        num_steps: TensorType["b"],
    ) -> TensorType["b", "c", "h", "w"]:

        x = self.conv_1(x)
        t_embed = self.embed_num_steps(num_steps)

        skips = []

        for i, down_block in enumerate(self.down_blocks):
            x, skip = down_block(x, t_embed)

            if i > 0:
                skips.append(skip)

        x = self.mid_block(x, t_embed)

        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skips.pop(), t_embed)

        x = self.resid_block(x, t_embed)
        x = self.conv_2(x)

        return x


# %%
