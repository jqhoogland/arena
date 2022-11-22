# %%

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# from arena.cv.diffusion import utils

# %%


def gradient_images(n_images: int, img_size: tuple[int, int, int]) -> t.Tensor:
    """
    Generate n_images of img_size, each a color gradient
    """
    (C, H, W) = img_size
    corners = t.randint(0, 255, (2, n_images, C))
    xs = t.linspace(0, W / (W + H), W)
    ys = t.linspace(0, H / (W + H), H)
    (x, y) = t.meshgrid(xs, ys, indexing="xy")
    grid = x + y
    grid = grid / grid[-1, -1]
    grid = repeat(grid, "h w -> b c h w", b=n_images, c=C)
    base = repeat(corners[0], "n c -> n c h w", h=H, w=W)
    ranges = repeat(corners[1] - corners[0], "n c -> n c h w", h=H, w=W)
    gradients = base + grid * ranges
    assert gradients.shape == (n_images, C, H, W)
    return gradients / 255


def plot_img(img: t.Tensor, title: Optional[str] = None) -> None:
    img = rearrange(img, "c h w -> h w c")
    plt.imshow(img.numpy())
    if title:
        plt.title(title)
    plt.show()


def normalize_img(img: t.Tensor) -> t.Tensor:
    return img * 2 - 1


def denormalize_img(img: t.Tensor) -> t.Tensor:
    return ((img + 1) / 2).clamp(0, 1)


if __name__ == "__main__":
    print("A few samples from the input distribution: ")
    img_shape = (3, 16, 16)
    n_images = 5
    imgs = gradient_images(n_images, img_shape)
    for i in range(n_images):
        plot_img(imgs[i])

    plot_img(imgs[0], "Original")
    plot_img(normalize_img(imgs[0]), "Normalized")
    plot_img(denormalize_img(normalize_img(imgs[0])), "Denormalized")


# %%


def linear_schedule(
    max_steps: int, min_noise: float = 0.0001, max_noise: float = 0.02
) -> t.Tensor:
    """
    Return the forward process variances as in the paper.

    max_steps: total number of steps of noise addition
    out: shape (step=max_steps, ) the amount of noise at each step
    """

    return t.linspace(min_noise, max_noise, max_steps)


if __name__ == "__main__":
    betas = linear_schedule(max_steps=200)


# %%


def q_forward_slow(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    """Return the input image with num_steps iterations of noise added according to schedule.
    x: shape (channels, height, width)
    betas: shape (T, ) with T >= num_steps

    out: shape (channels, height, width)
    """

    assert betas.shape[0] >= num_steps

    for _, beta in zip(range(num_steps), betas):  # As short as the shorter
        x = t.sqrt(1.0 - beta) * x + t.randn_like(x) * t.sqrt(beta)

    return x


if __name__ == "__main__":
    x = normalize_img(gradient_images(1, (3, 16, 16))[0])
    for n in [1, 10, 50, 200]:
        xt = q_forward_slow(x, n, betas)
        plot_img(denormalize_img(xt), f"Equation 2 after {n} step(s)")
    plot_img(denormalize_img(t.randn_like(xt)), "Random Gaussian noise")


# %%


def q_forward_fast(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    """Equivalent to Equation 2 but without a for loop."""

    assert betas.shape[0] >= num_steps

    alpha_bar = t.prod(1.0 - betas[:num_steps])
    return t.sqrt(alpha_bar) * x + t.randn_like(x) * (1.0 - alpha_bar)


if __name__ == "__main__":
    for n in [1, 10, 50, 200]:
        xt = q_forward_fast(x, n, betas)
        plot_img(denormalize_img(xt), f"Equation 4 after {n} steps")

# %%


class NoiseSchedule(nn.Module):
    betas: t.Tensor
    alphas: t.Tensor
    alpha_bars: t.Tensor

    def __init__(self, max_steps: int, device: Union[t.device, str]) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.device = device

        self.register_buffer("betas", linear_schedule(max_steps).to(device))

    @t.inference_mode()
    def beta(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the beta(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """
        batch_size = 1 if isinstance(num_steps, int) else num_steps.shape[0]

        return self.betas[num_steps].reshape((batch_size, -1))

    @t.inference_mode()
    def alpha(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the alphas(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """

        return 1.0 - self.beta(num_steps)

    @t.inference_mode()
    def alpha_bar(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the alpha_bar(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """

        return t.prod(self.alpha(num_steps), dim=1)

    def __len__(self) -> int:
        return self.max_steps


# %%


def noise_img(
    img: t.Tensor, noise_schedule: NoiseSchedule, max_steps: Optional[int] = None
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Adds a uniform random number of steps of noise to each image in img.

    img: An image tensor of shape (B, C, H, W)
    noise_schedule: The NoiseSchedule to follow
    max_steps: if provided, only perform the first max_steps of the schedule

    Returns a tuple composed of:
    num_steps: an int tensor of shape (B,) of the number of steps of noise added to each image
    noise: the unscaled, standard Gaussian noise added to each image, a tensor of shape (B, C, H, W)
    noised: the final noised image, a tensor of shape (B, C, H, W)
    """

    batch_size = img.shape[0]
    num_steps = t.randint(0, len(noise_schedule), (batch_size,), device=img.device)
    noise = t.randn_like(img)
    noised = t.sqrt(noise_schedule.alpha_bar(num_steps)).reshape(
        (batch_size, 1, 1, 1)
    ) * img + noise * t.sqrt(noise_schedule.alpha(num_steps)).reshape(
        (batch_size, 1, 1, 1)
    )

    return num_steps, noise, noised


if __name__ == "__main__":
    noise_schedule = NoiseSchedule(max_steps=10, device="cpu")
    img = gradient_images(1, (3, 16, 16))
    (num_steps, noise, noised) = noise_img(
        normalize_img(img), noise_schedule, max_steps=10
    )
    plot_img(img[0], "Gradient")
    plot_img(noise[0], "Applied Unscaled Noise")
    plot_img(denormalize_img(noised[0]), "Gradient with Noise Applied")

# %%
