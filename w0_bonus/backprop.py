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
        self._lookup: dict[Callable, dict[int, Callable]] = {}

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        self._lookup.setdefault(forward_fn, {})
        self._lookup[forward_fn][arg_position] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self._lookup[forward_fn][arg_position]

utils.test_back_func_lookup(BackwardFuncLookup)

BACK_FUNCS = BackwardFuncLookup()
BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

# %%


class Tensor:
    '''
    A drop-in replacement for torch.Tensor supporting a subset of features.
    '''

    array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    grad: Optional["Tensor"]
    "Backpropagation will accumulate gradients into this field."
    recipe: Optional[Recipe]
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Union[Arr, list], requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __eq__(self, other):
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    @property
    def T(self) -> "Tensor":
        return permute(self)

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def expand(self, new_shape):
        return expand(self, new_shape)

    def permute(self, dims):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        '''Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html'''
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
        return bool(self.item())

def empty(*shape: int) -> Tensor:
    '''Like torch.empty.'''
    return Tensor(np.empty(shape))

def zeros(*shape: int) -> Tensor:
    '''Like torch.zeros.'''
    return Tensor(np.zeros(shape))

def arange(start: int, end: int, step=1) -> Tensor:
    '''Like torch.arange(start, end).'''
    return Tensor(np.arange(start, end, step=step))

def tensor(array: Arr, requires_grad=False) -> Tensor:
    '''Like torch.tensor.'''
    return Tensor(array, requires_grad=requires_grad)

# %%

def log_forward(x: Tensor) -> Tensor:
    y = Tensor(
        np.log(x.array),
        requires_grad=x.requires_grad and grad_tracking_enabled,
    )

    if y.requires_grad:
        y.recipe = Recipe(
            func=np.log,
            args=(x.array,),
            kwargs={},
            parents={0: x},
        )

    return y

log = log_forward
utils.test_log(Tensor, log_forward)
utils.test_log_no_grad(Tensor, log_forward)
a = Tensor([1], requires_grad=True)
grad_tracking_enabled = False
b = log_forward(a)
grad_tracking_enabled = True
assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
assert b.recipe is None, "should not create recipe if grad tracking globally disabled"

# %%

def multiply_forward(a: Union[Tensor, int], b: Union[Tensor, int]) -> Tensor:
    y = Tensor(np.array(0))

    def set_recipe(args, parents):
        if y.requires_grad:
            y.recipe = Recipe(
                func=np.multiply,
                args=args,
                kwargs={},
                parents=parents
            )

    if isinstance(a, int) and isinstance(b, int):
        return Tensor(np.array(a * b))
    
    elif isinstance(a, int): 
        y.array = np.multiply(a, b.array) # type: ignore
        y.requires_grad = b.requires_grad and grad_tracking_enabled # type: ignore
        set_recipe((a, b.array), {1: b}) # type: ignore
            
    elif isinstance(b, int):
        y.array = np.multiply(a.array, b)
        y.requires_grad = a.requires_grad and grad_tracking_enabled
        set_recipe((a.array, b), {0: a})
        
    else:
        y.array = np.multiply(a.array, b.array)
        y.requires_grad = a.requires_grad and b.requires_grad and grad_tracking_enabled
        set_recipe((a.array, b.array), {0: a,  1: b})

    return y

multiply = multiply_forward 
utils.test_multiply(Tensor, multiply_forward) 
utils.test_multiply_no_grad(Tensor, multiply_forward) 
utils.test_multiply_float(Tensor, multiply_forward) 
a = Tensor([2], requires_grad=True) 
b = Tensor([3], requires_grad=True) 
grad_tracking_enabled = False 
b = multiply_forward(a, b) 
grad_tracking_enabled = True 
assert not b.requires_grad, "should not require grad if grad tracking globally disabled" 
assert b.recipe is None, "should not create recipe if grad tracking globally disabled"
# %%

def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    '''
    numpy_func: function. It takes any number of positional arguments, some of which may be NumPy arrays, and any number of keyword arguments which we aren't allowing to be NumPy arrays at present. It returns a single NumPy array.
    is_differentiable: if True, numpy_func is differentiable with respect to some input argument, so we may need to track information in a Recipe. If False, we definitely don't need to track information.

    Return: function. It has the same signature as numpy_func, except wherever there was a NumPy array, this has a Tensor instead.
    '''

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        np_args = tuple(arg.array if isinstance(arg, Tensor) else arg for arg in args)
        y = Tensor(
            numpy_func(*np_args, **kwargs),
            requires_grad = (
                any(isinstance(arg, Tensor) and arg.requires_grad for arg in args) 
                and is_differentiable and grad_tracking_enabled
            ),
        )

        if y.requires_grad:
            y.recipe = Recipe(
                func=numpy_func,
                args=tuple(arg.array if isinstance(arg, Tensor) else arg for arg in args),
                kwargs=kwargs,
                parents={i: arg for i, arg in enumerate(args) if isinstance(arg, Tensor)},
            )

        return y

    return tensor_func

log = wrap_forward_fn(np.log)
multiply = wrap_forward_fn(np.multiply)
utils.test_log(Tensor, log)
utils.test_log_no_grad(Tensor, log)
utils.test_multiply(Tensor, multiply)
utils.test_multiply_no_grad(Tensor, multiply)
utils.test_multiply_float(Tensor, multiply)
# utils.test_sum(wrap_forward_fn, Tensor)
try:
    log(x=Tensor([100]))
except Exception as e:
    print("Got a nice exception as intended:")
    print(e)
else:
    assert False, "Passing tensor by keyword should raise some informative exception."

# %%
