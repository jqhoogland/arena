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


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        bias=True,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self.kernel_width, self.kernel_height = force_pair(
            kernel_size
        )
        self.stride = stride
        self.padding = padding

        super().__init__()

        n_in = in_channels * self.kernel_width * self.kernel_height
        self.weight = nn.parameter.Parameter(
            uniform_random(
                (out_channels, in_channels, self.kernel_height, self.kernel_width),
                1.0 / np.sqrt(n_in),
            )
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join(
            map(
                lambda p: f"{p}={getattr(self, p)}",
                ("in_channels", "out_channels", "kernel_size", "stride", "padding"),
            )
        )


utils.test_conv2d_module(Conv2d)
