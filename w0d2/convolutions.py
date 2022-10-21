#%%

import numpy as np
import torch as t
from fancy_einsum import einsum

import utils

# %%


def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''

    batch_size, n_in_channels, width = x.shape
    batch_stride, in_channels_stride, width_stride = x.stride() # type: ignore
    
    _, _n_in_channels, kernel_width = weights.shape
    outer_width = width - kernel_width + 1

    assert n_in_channels == _n_in_channels, \
        f"The number of in channels must match between `x` and `weights`, received {n_in_channels} and {_n_in_channels}"
  
    x_view = x.as_strided(size=(
        batch_size,
        n_in_channels,
        outer_width,
        kernel_width
    ), stride=(
        batch_stride,
        in_channels_stride,
        width_stride,
        width_stride
    ))

    return einsum("b c_in w_out w_kernel, c_out c_in w_kernel -> b c_out w_out", x_view, weights)

utils.test_conv1d_minimal(conv1d_minimal)
# %%

def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    
    batch_size, n_in_channels, height, width = x.shape
    batch_stride, in_channels_stride, height_stride, width_stride = x.stride() # type: ignore
    
    _, _n_in_channels, kernel_height, kernel_width = weights.shape
    outer_height = height - kernel_height + 1 
    outer_width = width - kernel_width + 1

    assert n_in_channels == _n_in_channels, \
        f"The number of in channels must match between `x` and `weights`, received {n_in_channels} and {_n_in_channels}"
  
    x_view = x.as_strided(size=(
        batch_size,
        n_in_channels,
        outer_height,
        outer_width,
        kernel_height,
        kernel_width
    ), stride=(
        batch_stride,
        in_channels_stride,
        height_stride,
        width_stride,
        height_stride,
        width_stride
    ))

    return einsum("b c_in h_out w_out h_kernel w_kernel, c_out c_in h_kernel w_kernel -> b c_out h_out w_out", x_view, weights)

utils.test_conv2d_minimal(conv2d_minimal)
# %%
