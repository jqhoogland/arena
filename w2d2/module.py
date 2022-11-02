# %%
from typing import Union

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from arena.w2d2 import utils

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

class GELU(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Exact
        return 0.5 * x * (1. + t.erf(x / np.sqrt(2.)))

        # Approx 1
        # return x * t.sigmoid(1.702 * x)

        # Approx 2
        # return 0.5 * x * (1. + t.tanh(np.sqrt(2. / np.pi) * (x + 0.044715 * x ** 3)))

utils.plot_gelu(GELU)

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
