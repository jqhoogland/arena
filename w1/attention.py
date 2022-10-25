#%%
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
            Q @ K.transpose(-1, -2)
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

def mask(A: TensorType["batch", "seq_len", "seq_len"]) -> TensorType["batch", "seq_len", "d_head"]:
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
        Q @ K.transpose(-1, -2)
    ) / np.sqrt(d_head)

    A: TensorType["seq_len", "seq_len"] = t.softmax(A_pre, dim=1)

    return A @ V

# TODO: how to test?
# %%
