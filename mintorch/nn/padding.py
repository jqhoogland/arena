#%%
import torch as t
from torch import nn

# from arena.mintorch.nn.containers import Module

# %%


def pad1d(x: t.Tensor, left: int, right: int, pad_value=0.0) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    batch_size, n_in_channels, width = x.shape

    x_padded = (
        t.ones((batch_size, n_in_channels, left + width + right), dtype=x.dtype)
        * pad_value
    )
    x_padded[..., left : left + width] = x

    return x_padded


# utils.test_pad1d(pad1d)
# utils.test_pad1d_multi_channel(pad1d)


def pad2d(
    x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value=0.0
) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    batch_size, n_in_channels, height, width = x.shape

    x_padded = (
        t.ones(
            (batch_size, n_in_channels, top + height + bottom, left + width + right),
            dtype=x.dtype,
        )
        * pad_value
    )
    x_padded[..., top : top + height, left : left + width] = x

    return x_padded


# utils.test_pad2d(pad2d)
# utils.test_pad2d_multi_channel(pad2d)
# %%
