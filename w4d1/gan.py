# %%
import os
from typing import Optional, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from fancy_einsum import einsum
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

import arena.w4d1.utils as utils
from arena.w0d2.convolutions import (conv1d_minimal, conv2d_minimal, pad1d,
                                     pad2d)

# %%


def conv_transpose1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    kernel_size = weights.shape[2]
    pad_amount = kernel_size - 1
    x_padded = pad1d(x, pad_amount, pad_amount)
    return conv1d_minimal(x_padded, rearrange(weights.flip(-1), "i o k -> o i k"))


utils.test_conv_transpose1d_minimal(conv_transpose1d_minimal)

# %%


def fractional_stride_1d(x, stride: int = 1):
    """Returns a version of x suitable for transposed convolutions, i.e. "spaced out" with zeros between its values.
    This spacing only happens along the last dimension.

    x: shape (batch, in_channels, width)

    Example:
        x = [[[1, 2, 3], [4, 5, 6]]]
        stride = 2
        output = [[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]]
    """
    new_width = x.shape[-1] + (stride - 1) * (x.shape[-1] - 1)
    x_strided = t.zeros(
        size=(*x.shape[:-1], new_width),
        dtype=x.dtype,
        device=x.device,
    )
    x_strided[..., ::stride] = x

    return x_strided


utils.test_fractional_stride_1d(fractional_stride_1d)


def conv_transpose1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    kernel_size = weights.shape[2]
    pad_amount = kernel_size - 1 - padding
    x_strided = fractional_stride_1d(x, stride)
    x_padded = pad1d(x_strided, pad_amount, pad_amount)
    return conv1d_minimal(x_padded, rearrange(weights.flip(-1), "i o k -> o i k"))


utils.test_conv_transpose1d(conv_transpose1d)
# %%

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


def fractional_stride_2d(x, stride_h: int, stride_w: int):
    """
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    """
    height, width = x.shape[-2], x.shape[-1]
    new_width = width + (stride_w - 1) * (width - 1)
    new_height = height + (stride_h - 1) * (height - 1)

    x_strided = t.zeros(
        size=(*x.shape[:-2], new_height, new_width),
        dtype=x.dtype,
        device=x.device,
    )
    x_strided[..., ::stride_h, ::stride_w] = x

    return x_strided


def conv_transpose2d(
    x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0
) -> t.Tensor:
    """Like torch's conv_transpose2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    """
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    _, _, kernel_h, kernel_w = weights.shape

    padding_h = kernel_h - 1 - padding_h
    padding_w = kernel_w - 1 - padding_w

    x_strided = fractional_stride_2d(x, stride_h, stride_w)
    x_padded = pad2d(x_strided, padding_w, padding_w, padding_h, padding_h)

    return conv2d_minimal(
        x_padded, rearrange(weights.flip(-1).flip(-2), "o i h w -> i o h w")
    )


utils.test_conv_transpose2d(conv_transpose2d)

# %%


class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ):
        """
        Same as torch.nn.ConvTranspose2d with bias=False.

        Name your weight field `self.weight` for compatibility with the tests.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        kernel_h, kernel_w = force_pair(kernel_size)

        super().__init__()

        self.weight = nn.Parameter(
            t.zeros(in_channels, out_channels, kernel_h, kernel_w)
        )

        k = self.out_channels * kernel_h * kernel_w
        sf = 1 / np.sqrt(k)
        nn.init.uniform_(self.weight, a=-sf, b=sf)

    def forward(self, x: t.Tensor) -> t.Tensor:
        print(x.shape, self.weight.shape)
        return conv_transpose2d(x, self.weight, self.stride, self.padding)


utils.test_ConvTranspose2d(ConvTranspose2d)

# %%


class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.tanh(x)


utils.test_Tanh(Tanh)

# %%


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        self.neg_slope = negative_slope

        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.where(x > 0, x, x * self.neg_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.neg_slope}"


utils.test_LeakyReLU(LeakyReLU)
# %%


class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.sigmoid(x)


utils.test_Sigmoid(Sigmoid)
# %%


def initialize_weights(model) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        self.latent_dim = latent_dim

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            Rearrange("b (ic h w) -> b ic h w", h=4, w=4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvTranspose2d(64, 1, 4, 2, 1),
            Tanh(),
        )

        initialize_weights(self)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            LeakyReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            LeakyReLU(),
            nn.Conv2d(512, 1024, 4, 1, 1),
            LeakyReLU(),
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(1024 * 4 * 4, 1, bias=False),
            Sigmoid(),
        )

        initialize_weights(self)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x)


class DCGAN(nn.Module):
    def __init__(self, latent_dim: int = 100):
        self.latent_dim = latent_dim

        super().__init__()

        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.generator(x)


# %%

image_size = 64

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = ImageFolder(root="data", transform=transform)

# utils.show_images(trainset, rows=3, cols=5)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
# %%
device = "cuda" if t.cuda.is_available() else "cpu"


@t.inference_mode()
def display_generator_output(netG, latent_dim_size, rows=2, cols=5):

    netG.eval()
    device = next(netG.parameters()).device
    t.manual_seed(0)
    with t.inference_mode():
        noise = t.randn(rows * cols, latent_dim_size).to(device)
        img = netG(noise)
        print(noise.shape, img.shape)
        img_min = img.min(-1, True).values.min(-2, True).values
        img_max = img.max(-1, True).values.max(-2, True).values
        img = (img - img_min) / (img_max - img_min)
        img = utils.pad_width_height(img)
        img = rearrange(img, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=rows)
    if len(img.shape) == 3:
        img = img.squeeze()
    plt.imshow(img)
    plt.show()
    netG.train()


def train_generator_discriminator(
    netG: Generator,
    netD: Discriminator,
    trainloader: DataLoader,
    optG: Optional[t.optim.Optimizer] = None,
    optD: Optional[t.optim.Optimizer] = None,
    epochs: int = 5,
):
    netG.train().to(device)
    netD.train().to(device)

    if optG is None:
        optG = t.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    if optD is None:
        optD = t.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        print("\n\nEpoch\n\n", epoch)

        for i, (x_real, _) in enumerate(trainloader):
            print("\r", i, end="")

            x_real = x_real.to(device)
            z = t.randn(64, netG.latent_dim, device=device)
            netG.zero_grad()

            # Discriminator loop

            optD.zero_grad()
            x_real_eval = netD(x_real)
            x_fake = netG(z)
            x_fake_eval = netD(x_fake.detach())

            lossD = -t.mean(t.log(x_real_eval)) - t.mean(t.log(1 - x_fake_eval))
            lossD.backward()
            optD.step()

            # Generator loop

            optG.zero_grad()
            x_fake_eval = netD(x_fake)
            lossG = -t.mean(t.log(x_fake_eval))
            lossG.backward()
            optG.step()

            if i % 100 == 0:
                display_generator_output(netG, netG.latent_dim, rows=2, cols=6)


# %%

netG = Generator()
netD = Discriminator()

display_generator_output(netG, netG.latent_dim, rows=2, cols=6)
# %%

train_generator_discriminator(netG, netD, trainloader, epochs=5)

# %%
