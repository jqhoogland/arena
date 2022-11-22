#%%

import numpy as np
import torch as t
from fancy_einsum import einsum

from arena.convnets import utils

# %%


def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    batch_size, n_in_channels, width = x.shape
    batch_stride, in_channels_stride, width_stride = x.stride()  # type: ignore

    _, _n_in_channels, kernel_width = weights.shape
    outer_width = width - kernel_width + 1

    assert (
        n_in_channels == _n_in_channels
    ), f"The number of in channels must match between `x` and `weights`, received {n_in_channels} and {_n_in_channels}"

    x_view = x.as_strided(
        size=(batch_size, n_in_channels, outer_width, kernel_width),
        stride=(batch_stride, in_channels_stride, width_stride, width_stride),
    )

    return einsum(
        "b c_in w_out w_kernel, c_out c_in w_kernel -> b c_out w_out", x_view, weights
    )


utils.test_conv1d_minimal(conv1d_minimal)
# %%


def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """

    batch_size, n_in_channels, height, width = x.shape
    batch_stride, in_channels_stride, height_stride, width_stride = x.stride()  # type: ignore

    _, _n_in_channels, kernel_height, kernel_width = weights.shape
    outer_height = height - kernel_height + 1
    outer_width = width - kernel_width + 1

    assert (
        n_in_channels == _n_in_channels
    ), f"The number of in channels must match between `x` and `weights`, received {n_in_channels} and {_n_in_channels}"

    x_view = x.as_strided(
        size=(
            batch_size,
            n_in_channels,
            outer_height,
            outer_width,
            kernel_height,
            kernel_width,
        ),
        stride=(
            batch_stride,
            in_channels_stride,
            height_stride,
            width_stride,
            height_stride,
            width_stride,
        ),
    )

    return einsum(
        "b c_in h_out w_out h_kernel w_kernel, c_out c_in h_kernel w_kernel -> b c_out h_out w_out",
        x_view,
        weights,
    )


utils.test_conv2d_minimal(conv2d_minimal)
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


utils.test_pad1d(pad1d)
utils.test_pad1d_multi_channel(pad1d)
# %%


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


utils.test_pad2d(pad2d)
utils.test_pad2d_multi_channel(pad2d)
# %%


def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    batch_size, n_in_channels, width = x.shape

    _, _n_in_channels, kernel_width = weights.shape
    outer_width = ((width + 2 * padding - kernel_width) // stride) + 1

    assert (
        n_in_channels == _n_in_channels
    ), f"The number of in channels must match between `x` and `weights`, received {n_in_channels} and {_n_in_channels}"

    x_padded = pad1d(x, padding, padding, 0.0)
    batch_stride, in_channels_stride, width_stride = x_padded.stride()  # type: ignore

    x_view = x_padded.as_strided(
        size=(batch_size, n_in_channels, outer_width, kernel_width),
        stride=(batch_stride, in_channels_stride, width_stride * stride, width_stride),
    )

    return einsum(
        "b c_in w_out w_kernel, c_out c_in w_kernel -> b c_out w_out", x_view, weights
    )


utils.test_conv1d(conv1d)

# %%

IntOrPair = int | tuple[int, int]
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


# %%


def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    """Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    padding_y, padding_x = force_pair(padding)
    stride_y, stride_x = force_pair(stride)

    batch_size, n_in_channels, height, width = x.shape

    _, _n_in_channels, kernel_height, kernel_width = weights.shape
    outer_height = (height + 2 * padding_y - kernel_height) // stride_y + 1
    outer_width = (width + 2 * padding_x - kernel_width) // stride_x + 1

    assert (
        n_in_channels == _n_in_channels
    ), f"The number of in channels must match between `x` and `weights`, received {n_in_channels} and {_n_in_channels}"

    x_padded = pad2d(x, padding_x, padding_x, padding_y, padding_y)
    batch_stride, in_channels_stride, height_stride, width_stride = x_padded.stride()  # type: ignore

    x_view = x_padded.as_strided(
        size=(
            batch_size,
            n_in_channels,
            outer_height,
            outer_width,
            kernel_height,
            kernel_width,
        ),
        stride=(
            batch_stride,
            in_channels_stride,
            height_stride * stride_y,
            width_stride * stride_x,
            height_stride,
            width_stride,
        ),
    )

    return einsum(
        "b c_in h_out w_out h_kernel w_kernel, c_out c_in h_kernel w_kernel -> b c_out h_out w_out",
        x_view,
        weights,
    )


utils.test_conv2d(conv2d)

# %%


def maxpool2d(
    x: t.Tensor,
    kernel_size: IntOrPair,
    stride: IntOrPair | None = None,
    padding: IntOrPair = 0,
) -> t.Tensor:
    """Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    """
    kernel_height, kernel_width = force_pair(kernel_size)
    stride_y, stride_x = (
        force_pair(stride) if stride is not None else (kernel_height, kernel_width)
    )
    padding_y, padding_x = force_pair(padding)

    batch_size, n_channels, height, width = x.shape
    out_height = (height + 2 * padding_y - kernel_height) // stride_y + 1
    out_width = (width + 2 * padding_x - kernel_width) // stride_x + 1

    x_padded = pad2d(x, padding_x, padding_x, padding_y, padding_y, x.min())
    batch_stride, channels_stride, height_stride, width_stride = x_padded.stride()  # type: ignore

    x_view = x_padded.as_strided(
        size=(
            batch_size,
            n_channels,
            out_height,
            out_width,
            kernel_height,
            kernel_width,
        ),
        stride=(
            batch_stride,
            channels_stride,
            height_stride * stride_y,
            width_stride * stride_x,
            height_stride,
            width_stride,
        ),
    )

    return x_view.amax((-2, -1))


utils.test_maxpool2d(maxpool2d)

# %%