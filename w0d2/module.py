#%%
import numpy as np
import torch as t
import torch.nn as nn

import utils
from convolutions import force_pair, maxpool2d

# %%

IntOrPair = int | tuple[int, int]
Pair = tuple[int, int]

#%%

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: IntOrPair | None = None, padding: IntOrPair = 1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return ", ".join(map(lambda p: f"{p}={getattr(self, p)}", ("kernel_size", "stride", "padding")))

utils.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
# %%

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return (x > 0) * x

utils.test_relu(ReLU)

# %%

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim 
        super().__init__()

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        end = (len(input.shape) - 1 if self.end_dim == -1 else self.end_dim) + 1
        new_shape = (*input.shape[:self.start_dim], -1, *input.shape[end:])
        return input.reshape(new_shape)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"

utils.test_flatten(Flatten)
# %%
