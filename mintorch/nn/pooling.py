import numpy as np
import torch as t
import torch.nn as nn
from fancy_einsum import einsum

from arena.mintorch.utils import IntOrPair, force_pair

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


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: IntOrPair | None = None,
        padding: IntOrPair = 1,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join(
            map(
                lambda p: f"{p}={getattr(self, p)}",
                ("kernel_size", "stride", "padding"),
            )
        )


utils.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
