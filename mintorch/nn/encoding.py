#%%
import torch as t
from torch import nn
from torchtyping import TensorType

# from arena.mintorch.nn.containers import Module

# %%


def sinuosoidal_embedding(seq_len: int, embedding_dim: int) -> TensorType["s", "emb"]:
    i = t.arange(seq_len).unsqueeze(1)
    d = t.arange(embedding_dim).unsqueeze(0)

    return t.sin(i / 10000 ** (d / embedding_dim)) * (d % 2 == 0) + t.cos(
        i / 10000 ** ((d - 1) / embedding_dim)
    ) * (d % 2 == 1)


class PositionalEncoding(nn.Module):
    pe: t.Tensor

    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len

        super().__init__()

        self.register_buffer("pe", self.encode(self.max_len, self.d_model))

    def encode(self, seq_len: int, embedding_dim: int) -> TensorType["s", "emb"]:
        raise NotImplementedError

    def forward(self, x: t.Tensor) -> t.Tensor:
        raise NotImplementedError


class TokenSinusoidalPositionalEncoding(PositionalEncoding):
    def encode(self, seq_len: int, embedding_dim: int) -> TensorType["s", "emb"]:
        return sinuosoidal_embedding(seq_len, embedding_dim)

    def forward(self, x: TensorType["b", "s", "emb"]) -> TensorType["b", "s", "emb"]:
        _, seq_len, embedding_dim = x.shape

        return x + self.pe[:seq_len, :embedding_dim].unsqueeze(0)


# def test_positional_encoding():
#     pe = TokenSinusoidalPositionalEncoding(10)
#     x = t.randn(1, 10, 10)
#     y = pe(x)
#     assert y.shape == x.shape
#     assert y[0, 0, 0] == x[0, 0, 0] + pe.pe[0, 0]
#     assert y[0, 0, 1] == x[0, 0, 1] + pe.pe[0, 1]
#     assert y[0, 0, 2] == x[0, 0, 2] + pe.pe[0, 2]


# test_positional_encoding()
# %%
class IntSinusoidalPositionalEncoding(PositionalEncoding):
    def encode(self, seq_len: int, embedding_dim: int) -> TensorType["s", "emb"]:
        return sinuosoidal_embedding(seq_len, embedding_dim)

    def forward(self, x: TensorType["b"]) -> TensorType["b", "emb"]:
        return self.pe[x]


# TODO: Composition over inheritance. This is gross.
