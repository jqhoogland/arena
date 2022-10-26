#%%
from dataclasses import dataclass

import einops
import numpy as np
import torch as t
from arena.w1.encoding import SinusoidalPositionalEncoding
from fancy_einsum import einsum
from torch import nn
from torchtyping import TensorType

# %%

def attention(
    Q: TensorType["batch", "seq_len", "d_head"],
    K: TensorType["batch", "seq_len", "d_head"],
    V: TensorType["batch", "seq_len", "d_head"],
) -> TensorType["batch", "seq_len", "d_head"]:
    '''
    Should return the results of self-attention.

    Q: shape (batch, seq_len, d_head)
    K: shape (batch, seq_len, d_head)
    V: shape (batch, seq_len, d_head)

    Return: shape (batch, seq_len, d_head)
    '''
    _, _, d_head = Q.shape

    A: TensorType["seq_len", "seq_len"] = t.softmax(
        (
            einsum("b s h, b t h-> b s t", Q, K)
            / np.sqrt(d_head)
        ), dim=1
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
    assert (y[0, :, 1:] == 0.).all()

test_attention()
# %%

def mask(A: TensorType[..., "seq_len", "seq_len"]) -> TensorType[..., "seq_len", "seq_len"]:
    seq_len = A.shape[-1]
    return A * t.triu(t.ones(seq_len, seq_len))
    

def test_mask():
    x = t.randn(5, 10, 10)
    y = mask(x)
    assert y.shape == x.shape
    assert y[0, 0, 0] == x[0, 0, 0]
    assert y[0, 0, 1] == x[0, 0, 1]
    assert y[0, 0, 2] == x[0, 0, 2]
    assert y[0, 1, 0] == 0
    assert y[0, 1, 1] == x[0, 1, 1]
    assert y[0, 1, 2] == x[0, 1, 2]
    assert y[0, 2, 0] == 0
    assert y[0, 2, 1] == 0
    assert y[0, 2, 2] == x[0, 2, 2]

test_mask()

#%%
def masked_attention(
    Q: TensorType["batch", "seq_len", "d_head"],
    K: TensorType["batch", "seq_len", "d_head"],
    V: TensorType["batch", "seq_len", "d_head"],

) -> TensorType["batch", "seq_len", "d_head"]:
    '''
    Should return the results of self-attention.

    Q: shape (batch, seq_len, d_head)
    K: shape (batch, seq_len, d_head)
    V: shape (batch, seq_len, d_head)

    Return: shape (batch, seq_len, d_head)
    '''
    _, _, d_head = Q.shape

    A_pre: TensorType["batch", "seq_len", "seq_len"] = mask(
        einsum("b s h, b t h-> b s t", Q, K)
    ) / np.sqrt(d_head)

    A: TensorType["batch", "seq_len", "seq_len"] = t.softmax(A_pre, dim=1)

    return A @ V

# TODO: how to test?
# %%

def multihead_masked_attention(
    Q: TensorType["batch", "seq", "n_heads*headsize"], 
    K: TensorType["batch", "seq", "n_heads*headsize"], 
    V: TensorType["batch", "seq", "n_heads*headsize"],
    num_heads: int
) -> TensorType["batch", "seq", "n_heads*headsize"]:
    '''
    Should return the results of multihead self-attention.

    Q: shape (batch, seq, n_heads*headsize)
    K: shape (batch, seq, n_heads*headsize)
    V: shape (batch, seq, n_heads*headsize)
    num_heads: int

    Return: shape (batch, seq, n_heads*headsize)
    '''
    _, _, d_head = Q.shape

    _Q = einops.rearrange(Q, "batch seq (n_heads headsize) -> batch n_heads seq headsize", n_heads=num_heads)    
    _K = einops.rearrange(K, "batch seq (n_heads headsize) -> batch n_heads seq headsize", n_heads=num_heads)    
    _V = einops.rearrange(V, "batch seq (n_heads headsize) -> batch n_heads seq headsize", n_heads=num_heads)

    A_pre: TensorType["batch", "n_heads", "seq_len", "seq_len"] = mask(
        einsum("b h s c, b h t c -> b h s t", _Q, _K)
    ) / np.sqrt(d_head)

    A: TensorType["batch", "seq_len", "seq_len"] = t.softmax(A_pre, dim=1)

    return einops.rearrange(A @ _V, "batch n_heads seq headsize -> batch seq (n_heads headsize)") 

# %%

class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        super().__init__()

        self.W_QKV = nn.Linear(hidden_size, hidden_size * num_heads * 3)
        self.W_O = nn.Linear(hidden_size * num_heads, hidden_size)

    def forward(self, x: TensorType["batch", "seq", "hidden_size"]) -> TensorType["batch", "seq", "hidden_size"]:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        QKV = self.W_QKV(x)
        Q, K, V = QKV.chunk(3, dim=2)        
        return self.W_O(multihead_masked_attention(Q, K, V, self.num_heads))


def test_multihead_masked_attention():
    x = t.randn(5, 10, 10)
    y = MultiheadMaskedAttention(10, 2)(x)
    assert y.shape == (5, 10, 10)

test_multihead_masked_attention()
# %%

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int = 6
    num_heads: int = 8
    vocab_size: int = 256
    hidden_size: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05

config = TransformerConfig()



# %%

class MLPBlock(nn.Module):

    def __init__(self, hidden_size: int, dropout: float):
        self.hidden_size = hidden_size

        super().__init__()

        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, layer_norm_epsilon: float, dropout: float):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

        super().__init__()

        self.attention = MultiheadMaskedAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = MLPBlock(hidden_size, dropout)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        self.config = config

        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = SinusoidalPositionalEncoding(config.hidden_size, config.max_seq_len)

        self.dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(config.hidden_size, config.num_heads, config.layer_norm_epsilon, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.unembed = nn.Linear(config.hidden_size, config.vocab_size)        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        x = self.ln(x)
        x = self.unembed(x)
        x = self.softmax(x)

        return x


# %%
