#%%
from dataclasses import dataclass

import einops
import numpy as np
import torch as t
from fancy_einsum import einsum
from torch import nn
from torchtyping import TensorType

from arena.mintorch.nn.encoding import TokenSinusoidalPositionalEncoding

# %%


def attention(
    Q: TensorType["b", "s", "h"],
    K: TensorType["b", "s", "h"],
    V: TensorType["b", "s", "h"],
) -> TensorType["b", "s", "h"]:
    """
    Should return the results of self-attention.

    Q: shape (batch, seq_len, d_head)
    K: shape (batch, seq_len, d_head)
    V: shape (batch, seq_len, d_head)

    Return: shape (batch, seq_len, d_head)
    """
    _, _, d_head = Q.shape

    A: TensorType["s", "s"] = t.softmax(
        (einsum("b s h, b t h-> b s t", Q, K) / np.sqrt(d_head)), dim=1
    )

    return A @ V


def test_attention():
    q = t.zeros(5, 10, 10)
    q[0, 0, 0] = 1

    k = t.zeros(5, 10, 10)
    k[0, 0, 0] = 1
    k[0, 1, 0] = 1
    k[0, 2, 0] = 1

    v = t.zeros(5, 10, 10)
    v[0, 0, 0] = 1
    v[0, 1, 0] = 2
    v[0, 2, 0] = 3

    y = attention(q, k, v)

    assert y.shape == (5, 10, 10)
    assert (y[0, :, 0] != 0).all()
    assert (y[0, :, 1:] == 0.0).all()


test_attention()
# %%


def mask(A: TensorType[..., "s", "s"]) -> TensorType[..., "s", "s"]:
    seq_len = A.shape[-1]

    mask = t.triu(t.ones(seq_len, seq_len), diagonal=1).bool()
    return A.masked_fill(mask, -np.inf)


def test_mask():
    x = t.randn(5, 10, 10)
    y = mask(x)

    assert y.shape == x.shape
    assert y[0, 0, 0] == x[0, 0, 0]
    assert y[0, 0, 1] == -np.inf
    assert y[0, 0, 2] == -np.inf
    assert y[0, 1, 0] == x[0, 1, 0]
    assert y[0, 1, 1] == x[0, 1, 1]
    assert y[0, 1, 2] == -np.inf
    assert y[0, 2, 0] == x[0, 2, 0]
    assert y[0, 2, 1] == x[0, 2, 1]
    assert y[0, 2, 2] == x[0, 2, 2]


test_mask()

#%%
def masked_attention(
    Q: TensorType["b", "s", "h"],
    K: TensorType["b", "s", "h"],
    V: TensorType["b", "s", "h"],
) -> TensorType["b", "s", "h"]:
    """
    Should return the results of self-attention.

    Q: shape (batch, seq_len, d_head)
    K: shape (batch, seq_len, d_head)
    V: shape (batch, seq_len, d_head)

    Return: shape (batch, seq_len, d_head)
    """
    _, _, d_head = Q.shape

    A_pre: TensorType["b", "s", "s"] = mask(
        einsum("b s h, b t h-> b s t", Q, K)
    ) / np.sqrt(d_head)

    A: TensorType["b", "s", "s"] = t.softmax(A_pre, dim=-2)

    return A @ V


# TODO: how to test?
# %%


def multihead_masked_attention(
    Q: TensorType["b", "s", "n_heads*headsize"],
    K: TensorType["b", "s", "n_heads*headsize"],
    V: TensorType["b", "s", "n_heads*headsize"],
    num_heads: int,
) -> TensorType["b", "s", "n_heads*headsize"]:
    """
    Should return the results of multihead self-attention.

    Q: shape (batch, seq, n_heads*headsize)
    K: shape (batch, seq, n_heads*headsize)
    V: shape (batch, seq, n_heads*headsize)
    num_heads: int

    Return: shape (batch, seq, n_heads*headsize)
    """
    _Q = einops.rearrange(Q, "b s (n h) -> b n s h", n=num_heads)
    _K = einops.rearrange(K, "b s (n h) -> b n s h", n=num_heads)
    _V = einops.rearrange(V, "b s (n h) -> b n s h", n=num_heads)

    d_head = _Q.shape[-1]

    A_pre: TensorType["b", "n", "s_q", "s_k"] = mask(
        einsum("b n s_q h, b n s_k h -> b n s_q s_k", _Q, _K)
    ) / np.sqrt(d_head)

    A: TensorType["b", "n", "s_q", "s_k"] = t.softmax(A_pre, dim=-1)
    AV: TensorType["b", "n", "s_q", "h"] = einsum(
        "b n s_q s_k, b n s_k h -> b n s_q h", A, _V
    )

    return einops.rearrange(AV, "b n s h -> b s (n h)")


def test_multihead_masked_attention():
    Q = t.linspace(0, 10, 2 * 5 * 4).reshape(2, 5, 4)
    K = t.linspace(5, 20, 2 * 5 * 4).reshape(2, 5, 4)
    V = t.linspace(15, 2, 2 * 5 * 4).reshape(2, 5, 4)
    y = multihead_masked_attention(Q, K, V, num_heads=2)

    print(y)

    assert t.allclose(
        y,
        t.tensor(
            [
                [
                    [15.0000, 14.6667, 14.3333, 14.0000],
                    [13.7668, 13.4335, 13.0346, 12.7012],
                    [12.3451, 12.0117, 11.6705, 11.3372],
                    [11.0013, 10.6679, 10.3337, 10.0004],
                    [9.6668, 9.3335, 9.0000, 8.6667],
                ],
                [
                    [8.3333, 8.0000, 7.6667, 7.3333],
                    [7.0000, 6.6667, 6.3333, 6.0000],
                    [5.6667, 5.3333, 5.0000, 4.6667],
                    [4.3333, 4.0000, 3.6667, 3.3333],
                    [3.0000, 2.6667, 2.3333, 2.0000],
                ],
            ]
        ),
        atol=1e-4,
        rtol=1e-4,
    )


test_multihead_masked_attention()

# %%

# %%


class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        super().__init__()

        self.W_QKV = nn.Linear(hidden_size, hidden_size * 3)
        self.W_O = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: TensorType["b", "s", "h"]) -> TensorType["b", "s", "h"]:
        Q, K, V = self.W_QKV(x).chunk(3, dim=-1)
        return self.W_O(multihead_masked_attention(Q, K, V, self.num_heads))


def test_multihead_masked_attention_2():
    t.manual_seed(420)
    m = MultiheadMaskedAttention(6, 2)
    x = t.linspace(0, 42, 2 * 3 * 6).reshape(2, 3, 6)

    y = m(x)
    print(y)

    assert t.allclose(
        y,
        t.tensor(
            [
                [
                    [-0.7193, 0.4614, 0.4117, -0.5813, 0.2754, -0.5745],
                    [-0.7746, 0.6206, 0.5520, -0.7370, 0.1787, -0.7289],
                    [-1.1632, 1.7392, 1.5775, -1.7907, -0.5079, -1.8103],
                ],
                [
                    [0.0549, -1.9665, -10.8756, -7.1792, 3.4559, 0.9521],
                    [-0.3971, -0.6652, -9.6883, -8.4108, 2.6582, -0.3063],
                    [-0.8686, 0.6920, -8.4500, -9.6953, 1.8262, -1.6189],
                ],
            ]
        ),
        atol=1e-4,
        rtol=1e-4,
    )


test_multihead_masked_attention_2()

# %%


class SelfAttention2d(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, channels: int, num_heads: int = 4):
        """
        Self-Attention with two spatial dimensions.

        channels: the number of channels. Should be divisible by the number of heads.
        """
        assert channels % num_heads == 0

        self.channels = channels
        self.num_heads = num_heads
        self.head_size = channels // num_heads

        super().__init__()

        self.W_QKV = nn.Linear(channels, channels * 3)
        self.W_O = nn.Linear(channels, channels)

    def forward(
        self, x: TensorType["b", "c", "h", "w"]
    ) -> TensorType["b", "c", "h", "w"]:
        """
        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        """
        b, c, h, w = x.shape
        assert c == self.channels

        x_flat = einops.rearrange(x, "b c h w -> b (h w) c")

        Q, K, V = self.W_QKV(x_flat).chunk(3, dim=-1)
        attn = self.W_O(attention(Q, K, V))

        return einops.rearrange(attn, "b (h w) c -> b c h w", h=h, w=w)
