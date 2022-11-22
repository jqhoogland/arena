#%%
import numpy as np
import torch as t
import torch.nn as nn
from fancy_einsum import einsum

from arena.convnets import utils
from arena.convnets.convolutions import conv2d, force_pair, maxpool2d

# %%


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        self.in_features = in_features
        self.out_features = out_features

        super().__init__()

        self.weight = nn.parameter.Parameter(
            (uniform_random((out_features, in_features), 1.0 / np.sqrt(in_features)))
        )
        self.bias = (
            nn.parameter.Parameter(
                (uniform_random((out_features,), 1.0 / np.sqrt(in_features)))
            )
            if bias
            else None
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        bias = self.bias if self.bias is not None else 0.0

        return (
            einsum(
                "b in_features, out_features in_features -> b out_features",
                x,
                self.weight,
            )
            + bias
        )

    def extra_repr(self) -> str:
        bias = self.bias and True
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={bias}"


utils.test_linear_forward(Linear)
utils.test_linear_parameters(Linear)
utils.test_linear_no_bias(Linear)
