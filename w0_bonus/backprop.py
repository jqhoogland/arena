# %%
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, Union

import numpy as np
from arena.w0_bonus import utils
from einops import repeat

# %%

Arr = np.ndarray
grad_tracking_enabled = True
# %%

@dataclass(frozen=True)
class Recipe:
    '''Extra information necessary to run backpropagation. You don't need to modify this.'''

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."
    args: tuple
    "The input arguments passed to func."
    kwargs: dict[str, Any]
    "Keyword arguments passed to func. To keep things simple today, we aren't going to backpropagate with respect to these."
    parents: dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."

# %%

def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backwards function for f(x) = log(x)

    grad_out: gradient of some loss wrt out
    out: the output of np.log(x)
    x: the input of np.log

    Return: gradient of the given loss wrt x
    '''
    return grad_out / x

utils.test_log_back(log_back)

# %%

def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    '''Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''
    in_shape = broadcasted.shape
    out_shape = original.shape

    if len(in_shape) > len(out_shape):
        broadcasted = broadcasted.sum(axis=tuple(i for i in range(len(in_shape) - len(out_shape))))
        in_shape = broadcasted.shape

    return broadcasted.sum(
        axis=tuple(i for i, (a, b) in enumerate(zip(in_shape, out_shape)) if a != b), 
        keepdims=True
    )

utils.test_unbroadcast(unbroadcast)

# %%

def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    '''Backwards function for x * y wrt argument 0 aka x.'''
    return unbroadcast(grad_out * y, x)

def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    '''Backwards function for x * y wrt argument 1 aka y.'''
    return unbroadcast(grad_out * x, y)

utils.test_multiply_back(multiply_back0, multiply_back1)
utils.test_multiply_back_float(multiply_back0, multiply_back1)

# %%

def forward_and_back(a: Arr, b: Arr, c: Arr) -> tuple[Arr, Arr, Arr]:
    '''
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    '''
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)

    final_grad_out = np.array([1.0])

    grad_g = final_grad_out
    grad_f = log_back(grad_g, g, f) 
    grad_e = multiply_back1(grad_f, f, d, e)
    grad_d = multiply_back0(grad_f, f, d, e)
    grad_c = log_back(grad_e, e, c)
    grad_a = multiply_back0(grad_d, d, a, b)
    grad_b = multiply_back1(grad_d, d, a, b)

    return (
        grad_a,
        grad_b,
        grad_c,
    )

utils.test_forward_and_back(forward_and_back)

# %%

class BackwardFuncLookup:
    def __init__(self) -> None:
        pass

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        pass

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        pass

utils.test_back_func_lookup(BackwardFuncLookup)

BACK_FUNCS = BackwardFuncLookup()
BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)
