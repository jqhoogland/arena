#%%
import numpy as np
import torch as t
import torch.nn as nn
from fancy_einsum import einsum

from arena.convnets import utils
from arena.convnets.convolutions import conv2d, force_pair, maxpool2d

# %%

IntOrPair = int | tuple[int, int]
Pair = tuple[int, int]

#%%


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
# %%


# %%


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim
        super().__init__()

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both."""
        end = (len(input.shape) - 1 if self.end_dim == -1 else self.end_dim) + 1
        new_shape = (*input.shape[: self.start_dim], -1, *input.shape[end:])
        return input.reshape(new_shape)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


utils.test_flatten(Flatten)
# %%


def uniform_random(
    size: t.Size | tuple, min_: float, max_: float | None = None
) -> t.Tensor:
    if max_ is None:
        # Then we sample from -min_ to min_
        min_, max_ = -min_, min_

    return t.rand(size) * (max_ - min_) + min_


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

# %%



class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        super().__init__()

        self.weight = nn.Parameter(t.normal(0., 1., size=(num_embeddings, embedding_dim)))

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''
        return self.weight[x]
        # return t.index_select(self.weight, dim=0, index=x)

    def extra_repr(self) -> str:
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}'

utils.test_embedding(Embedding)

# %%

# %%

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: int | list[int], eps: float = 1e-05, elementwise_affine: bool = True):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        super().__init__()

        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(normalized_shape))
            self.bias = nn.Parameter(t.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: t.Tensor) -> t.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        x_norm = ((x - mean) / t.sqrt(var + self.eps))
    
        if self.elementwise_affine:
            return x_norm * self.weight + self.bias

        return  x_norm

utils.test_layernorm_mean_1d(LayerNorm)
utils.test_layernorm_mean_2d(LayerNorm)
utils.test_layernorm_std(LayerNorm)
utils.test_layernorm_exact(LayerNorm)
utils.test_layernorm_backward(LayerNorm)

# %%

class Dropout(nn.Module):

    def __init__(self, p: float):
        self.p = p

        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training and self.p:
            mask = t.empty(x.shape).bernoulli_(1 - self.p)
            return x * mask * (1 / (1 - self.p))
            
        return x


utils.test_dropout_eval(Dropout)
utils.test_dropout_training(Dropout)

# %%
