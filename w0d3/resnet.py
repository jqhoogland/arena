#%%
import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import PIL
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import torchvision
from arena.w0d3 import utils
from PIL import Image
from plotly.subplots import make_subplots
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# %%

class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x
# %%

class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features

        self.num_batches_tracked = t.tensor(0, dtype=t.float)

        super().__init__()

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        
        if not self.training:
            x_hat = (x - self.running_mean.reshape((1, -1, 1, 1))) / t.sqrt(self.running_var + self.eps)
        else:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean[0, :, 0, 0]
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var[0, :, 0, 0]

            x_hat = (x - batch_mean) / t.sqrt(batch_var + self.eps)
            self.num_batches_tracked += 1

        return x_hat * self.weight.as_strided(size=x_hat.shape, stride=(0, 1, 0, 0)) \
            + self.bias.as_strided(size=x_hat.shape, stride=(0, 1, 0, 0))

    def extra_repr(self) -> str:
        return f"momentum={self.momentum}, eps={self.eps}, num_features={self.num_features}"

utils.test_batchnorm2d_module(BatchNorm2d)
utils.test_batchnorm2d_forward(BatchNorm2d)
utils.test_batchnorm2d_running_mean(BatchNorm2d)
# %%

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return x.mean(dim=(2, 3))


# %%
