#%%
import einops
import numpy as np
import torch as t
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
    _, seq_len, _ = A.shape
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

    A: TensorType["seq_len", "seq_len"] = t.softmax(A_pre, dim=1)

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

    A: TensorType["seq_len", "seq_len"] = t.softmax(A_pre, dim=1)

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
