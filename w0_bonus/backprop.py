# %%
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import (Any, Callable, Iterable, Iterator, Optional, Protocol,
                    TypeVar, Union)

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

class Node:
    def __init__(self, *children):
        self.children = list(children)

def get_children(node: Node) -> list[Node]:
    return node.children

T = TypeVar("T")

def topological_sort(node: T, get_children_fn: Callable[[T], list[T]]) -> list[T]:
    '''
    Return a list of node's descendants in reverse topological order from future to past.

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    '''
    
    def dfs(node: T, visited: set[T], stack: list[T], path: set[T] = set()):
        visited.add(node)

        for child in get_children_fn(node):
            if child not in visited:
                dfs(child, visited, stack, path | {node})
            if child in path:
                raise ValueError("Graph is not acyclic")
        stack.append(node)
    
    visited = set()
    stack = []
    dfs(node, visited, stack)
    return stack

utils.test_topological_sort_linked_list(topological_sort)
utils.test_topological_sort_branching(topological_sort)
utils.test_topological_sort_rejoining(topological_sort)
utils.test_topological_sort_cyclic(topological_sort)
# %%

def get_parents(node: Tensor) -> list[Tensor]:
    if node.recipe is None:
        return []
    else:
        return list(node.recipe.parents.values())

def sorted_computational_graph(node: Tensor) -> list[Tensor]:
    '''
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph, in reverse topological order.
    '''
    return topological_sort(node, get_parents)[::-1]

a = Tensor([1], requires_grad=True)
b = Tensor([2], requires_grad=True)
c = Tensor([3], requires_grad=True)
d = a * b
e = c.log()
f = d * e
g = f.log()
name_lookup = {a: "a", b: "b", c: "c", d: "d", e: "e", f: "f", g: "g"}

print([name_lookup[t] for t in sorted_computational_graph(g)])
# Should get something in reverse alphabetical order (or close)

# %%

def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    '''Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node: 
        The rightmost node in the computation graph. 
        If it contains more than one element, end_grad must be provided.
    end_grad: 
        A tensor of the same shape as end_node. 
        Set to 1 if not specified and end_node has only one element.
    '''

    # Get value of end_grad_arr
    end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array

    # Create dict to store gradients
    grads: dict[Tensor, Arr] = {end_node: end_grad_arr}

    # Iterate through the computational graph, using your sorting function
    for node in sorted_computational_graph(end_node):

        # Get the outgradient (recall we need it in our backward functions)
        outgrad = grads.pop(node)
        # We only store the gradients if this node is a leaf (see the is_leaf property of Tensor)
        if node.is_leaf:
            # Add the gradient to this node's grad (need to deal with special case grad=None)
            if node.grad is None:
                node.grad = Tensor(outgrad)
            else:
                node.grad.array += outgrad

        # If node has no recipe, then it has no parents, i.e. the backtracking through computational
        # graph ends here
        if node.recipe is None:
            continue

        # If node has a recipe, then we iterate through parents (which is a dict of {arg_posn: tensor})
        for argnum, parent in node.recipe.parents.items():

            # Get the backward function corresponding to the function that created this node,
            # and the arg posn of this particular parent within that function 
            back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)

            # Use this backward function to calculate the gradient
            in_grad = back_fn(outgrad, node.array, *node.recipe.args, **node.recipe.kwargs)

            # Add the gradient to this node in the dictionary `grads`
            # Note that we only change the grad of the node itself in the code block above
            if grads.get(parent) is None:
                grads[parent] = in_grad
            else:
                grads[parent] += in_grad
# %%

def _argmax(x: Arr, dim=None, keepdim=False):
    '''Like torch.argmax.'''
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))

argmax = wrap_forward_fn(_argmax, is_differentiable=False)
eq = wrap_forward_fn(np.equal, is_differentiable=False)

a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
b = a.argmax()
assert not b.requires_grad
assert b.recipe is None
assert b.item() == 3

# %%

def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backward function for f(x) = -x elementwise.'''
    return np.full_like(x, -1) * grad_out 
    
negative = wrap_forward_fn(np.negative)
BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

utils.test_negative_back(Tensor)
# %%

def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return out * grad_out

exp = wrap_forward_fn(np.exp)
BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

utils.test_exp_back(Tensor)

# %%

def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return grad_out.reshape(x.shape)

reshape = wrap_forward_fn(np.reshape)
BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

utils.test_reshape_back(Tensor)

# %%

def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, tuple(np.argsort(axes)))

BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
permute = wrap_forward_fn(np.transpose)

utils.test_permute_back(Tensor)
# %%

def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out, x)

def _expand(x: Arr, new_shape) -> Arr:
    '''Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    '''
    n_added = len(new_shape) - x.ndim
    shape_non_negative = tuple([x.shape[i - n_added] if s == -1 else s for i, s in enumerate(new_shape)])

    return np.broadcast_to(x, shape_non_negative)

expand = wrap_forward_fn(_expand)
BACK_FUNCS.add_back_func(_expand, 0, expand_back)

utils.test_expand(Tensor)
utils.test_expand_negative_length(Tensor)
# %%

def sum_back(grad_out: Arr | Tensor, out: Arr, x: Arr, dim=None, keepdim=False):
    '''Basic idea: repeat grad_out over the dims along which x was summed'''
    
    # If grad_out is a scalar, we need to make it a tensor (so we can expand it later)
    if not isinstance(grad_out, Arr):
        grad_out = Tensor(grad_out)
    
    # If dim=None, this means we summed over all axes, and we want to repeat back to input shape
    if dim is None:
        dim = list(range(x.ndim))
        
    # If keepdim=False, then we need to add back in dims, so grad_out and x have same number of dims
    if keepdim == False:
        grad_out = np.expand_dims(grad_out, dim)
    
    # Finally, we repeat grad_out along the dims over which x was summed
    return np.broadcast_to(grad_out, x.shape)

def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    '''Like torch.sum, calling np.sum internally.'''
    return np.sum(x, axis=dim, keepdims=keepdim)

sum = wrap_forward_fn(_sum)
BACK_FUNCS.add_back_func(_sum, 0, sum_back)

utils.test_sum_keepdim_false(Tensor)
utils.test_sum_keepdim_true(Tensor)
utils.test_sum_dim_none(Tensor)
# %%

Index = Union[int, tuple[int, ...], tuple[Arr], tuple[Tensor]]

def coerce_index(index: Index) -> Union[int, tuple[int, ...], tuple[Arr]]:
    """
    If index is of type signature `tuple[Tensor]`, converts it to `tuple[Arr]`.
    """
    if isinstance(index, tuple) and set(map(type, index)) == {Tensor}:
        return tuple([i.array for i in index])
    else:
        return index

def _getitem(x: Arr, index: Index) -> Arr:
    '''Like x[index] when x is a torch.Tensor.'''
    return x[coerce_index(index)]

def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    '''Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    '''
    new_grad_out = np.full_like(x, 0)
    np.add.at(new_grad_out, coerce_index(index), grad_out)
    return new_grad_out

getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)

utils.test_getitem_int(Tensor)
utils.test_getitem_tuple(Tensor)
utils.test_getitem_integer_array(Tensor)
utils.test_getitem_integer_tensor(Tensor)

# %%

add = wrap_forward_fn(np.add)
subtract = wrap_forward_fn(np.subtract)
true_divide = wrap_forward_fn(np.true_divide)

# Your code goes here
BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
BACK_FUNCS.add_back_func(np.add, 1, lambda grad_out, out, x, y:  unbroadcast(grad_out, y))
BACK_FUNCS.add_back_func(np.subtract, 0, lambda grad_out, out, x, y:  unbroadcast(grad_out, x))
BACK_FUNCS.add_back_func(np.subtract, 1, lambda grad_out, out, x, y: unbroadcast(-grad_out, y))
BACK_FUNCS.add_back_func(np.true_divide, 0, lambda grad_out, out, x, y: unbroadcast(grad_out/y, x))
BACK_FUNCS.add_back_func(np.true_divide, 1, lambda grad_out, out, x, y: unbroadcast(grad_out * (-x / (y ** 2)), y))

utils.test_add_broadcasted(Tensor)
utils.test_subtract_broadcasted(Tensor)
utils.test_truedivide_broadcasted(Tensor)
# %%


🏠
Home
1️⃣
Introduction
2️⃣
Autograd
3️⃣
More forward and backward methods
4️⃣
Putting everything together
5️⃣
Bonus
Table of Contents
Non-Differentiable Functions
Implementing negative
Implementing exp
Implementing reshape
permute
expand
sum
indexing
Elementwise add, divide, subtract
In-Place Operations
Mixed scalar-tensor operations
In-Place Operations
max
Functional ReLU
2D Matrix Multiply
Filling out the Tensor class with forward and backward methods
Congrats on implementing backprop! The next thing we'll do is write implement a bunch of backward functions that we need to train our model at the end of the day, as well as ones that cover interesting cases.

These should be just like your log_back and multiply_back0, multiplyback1 examples earlier.

Non-Differentiable Functions
For functions like torch.argmax or torch.eq, there's no sensible way to define gradients with respect to the input tensor. For these, we will still use wrap_forward_fn because we still need to unbox the arguments and box the result, but by passing is_differentiable=False we can avoid doing any unnecessary computation.


def _argmax(x: Arr, dim=None, keepdim=False):
    '''Like torch.argmax.'''
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))

argmax = wrap_forward_fn(_argmax, is_differentiable=False)
eq = wrap_forward_fn(np.equal, is_differentiable=False)

a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
b = a.argmax()
assert not b.requires_grad
assert b.recipe is None
assert b.item() == 3
Implementing negative
torch.negative just performs -x elementwise. Make your own version negative using wrap_forward_fn.


def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backward function for f(x) = -x elementwise.'''
    pass

negative = wrap_forward_fn(np.negative)
BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

utils.test_negative_back(Tensor)
Implementing exp
Make your own version of torch.exp. The backward function should express the result in terms of the out parameter - this more efficient than expressing it in terms of x.


def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return out * grad_out

exp = wrap_forward_fn(np.exp)
BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

utils.test_exp_back(Tensor)
Implementing reshape
reshape is a bit more complicated than the many functions we've dealt with so far: there is an additional positional argument new_shape. Since it's not a Tensor, we don't need to think about differentiating with respect to it. Remember, new_shape is the argument that gets passed into the forward function, and we're trying to reverse this operation and return to the shape of the input.

Depending how you wrote wrap_forward_fn and backprop, you might need to go back and adjust them to handle this. Or, you might just have to implement reshape_back and everything will work.

Note that the output is a different shape than the input, but this doesn't introduce any additional complications.


def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    pass

reshape = wrap_forward_fn(np.reshape)
BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

utils.test_reshape_back(Tensor)
permute
In NumPy, the equivalent of torch.permute is called np.transpose, so we will wrap that.

Help - I'm confused about how to implement `permute_back`.

def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    pass

BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
permute = wrap_forward_fn(np.transpose)

utils.test_permute_back(Tensor)
expand
Implement your version of torch.expand.

The backward function should just call unbroadcast.

For the forward function, we will use np.broadcast_to. This function takes in an array and a target shape, and returns a version of the array broadcasted to the target shape using the rules of broadcasting we discussed in an earlier section. For example:


>>> x = np.array([1, 2, 3])
>>> np.broadcast_to(x, (3, 3))
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])

>>> x = np.array([[1], [2], [3]])
>>> np.broadcast_to(x, (3, 3)) # x is already a column; broadcasting is done along rows
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
The reason we can't just use np.broadcast_to and call it a day is that torch.expand supports -1 for a dimension size meaning "don't change the size". For example:


>>> x = torch.tensor([[1], [2], [3]])
>>> x.expand(-1, 3)
tensor([[ 1,  1,  1],
        [ 2,  2,  2],
        [ 3,  3,  3]])
So when implementing _expand, you'll need to be a bit careful when constructing the shape to broadcast to.

Help - I'm not sure how to construct the shape.

def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    pass

def _expand(x: Arr, new_shape) -> Arr:
    '''Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    '''
    pass

expand = wrap_forward_fn(_expand)
BACK_FUNCS.add_back_func(_expand, 0, expand_back)

utils.test_expand(Tensor)
utils.test_expand_negative_length(Tensor)
sum
The output can also be smaller than the input, such as when calling torch.sum. Implement your own torch.sum and sum_back.

Note, if you get weird exceptions that you can't explain, and these exceptions don't even go away when you use the solutions provided, this probably means that your implementation of wrap_forward_fn was wrong in a way which wasn't picked up by the tests. You should return to this function and try to fix it (or just use the solution).


def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    '''Basic idea: repeat grad_out over the dims along which x was summed'''
    pass

def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    '''Like torch.sum, calling np.sum internally.'''
    pass

sum = wrap_forward_fn(_sum)
BACK_FUNCS.add_back_func(_sum, 0, sum_back)

utils.test_sum_keepdim_false(Tensor)
utils.test_sum_keepdim_true(Tensor)
utils.test_sum_dim_none(Tensor)
indexing
In its full generality, indexing a torch.Tensor is really complicated and there are quite a few cases to handle separately.

We only need two cases today:

The index is an integer or tuple of integers.
The index is a tuple of (array or Tensor) representing coordinates. Each array is 1D and of equal length. Some coordinates may be repeated. This is Integer array indexing.
For example, to select the five elements at (0, 0), (1,0), (0, 1), (1, 2), and (0, 0), the index would be the tuple (np.array([0, 1, 0, 1, 0]), np.array([0, 0, 1, 2, 0])).
Note, in _getitem you'll need to deal with one special case: when index is of type signature tuple[Tensor]. If not for this case, return x[index] would suffice for this function. It might help to define a coerce_index function to deal with this particular case; we've provided a docstring for this purpose.

Help - I'm confused about the implementation of getitem_back!

Index = Union[int, tuple[int, ...], tuple[Arr], tuple[Tensor]]

def _getitem(x: Arr, index: Index) -> Arr:
    '''Like x[index] when x is a torch.Tensor.'''
    return x[coerce_index(index)]


def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    '''Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    '''
    new_grad_out = np.full_like(x, 0)
    np.add.at(new_grad_out, coerce_index(index), grad_out)
    return new_grad_out

getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)

utils.test_getitem_int(Tensor)
utils.test_getitem_tuple(Tensor)
utils.test_getitem_integer_array(Tensor)
utils.test_getitem_integer_tensor(Tensor)
elementwise add, subtract, divide
These are exactly analogous to the multiply case. Note that Python and NumPy have the notion of "floor division", which is a truncating integer division as in 7 // 3 = 2. You can ignore floor division: - we only need the usual floating point division which is called "true division".

Use lambda functions to define and register the backward functions each in one line. If you're confused, you can click on the expander below to reveal the first one.

Reveal the first one:

BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))

add = wrap_forward_fn(np.add)
subtract = wrap_forward_fn(np.subtract)
true_divide = wrap_forward_fn(np.true_divide)

# Your code goes here

utils.test_add_broadcasted(Tensor)
utils.test_subtract_broadcasted(Tensor)
utils.test_truedivide_broadcasted(Tensor)

# %%

def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    '''Like torch.add_. Compute x += other * alpha in-place and return tensor.'''
    np.add(x.array, other.array * alpha, out=x.array)
    return x


def safe_example():
    '''This example should work properly.'''
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    a.add_(b)
    c = a * b
    c.sum().backward()
    assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


def unsafe_example():
    '''This example is expected to compute the wrong gradients.'''
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    c = a * b
    a.add_(b)
    c.sum().backward()
    if a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0]):
        print("Grad wrt a is OK!")
    else:
        print("Grad wrt a is WRONG!")
    if b.grad is not None and np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0]):
        print("Grad wrt b is OK!")
    else:
        print("Grad wrt b is WRONG!")
# %%

a = Tensor([0, 1, 2, 3], requires_grad=True)
(a * 2).sum().backward()
b = Tensor([0, 1, 2, 3], requires_grad=True)
(2 * b).sum().backward()
assert a.grad is not None
assert b.grad is not None
assert np.allclose(a.grad.array, b.grad.array)

# %%

def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt x.'''
    return unbroadcast(grad_out * ((x > y) + 0.5 * (x == y)), x)

def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt y.'''
    return unbroadcast(grad_out * ((y > x) + 0.5 * (x == y)), y)

maximum = wrap_forward_fn(np.maximum)

BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)

utils.test_maximum(Tensor)
utils.test_maximum_broadcasted(Tensor)

# %%

def relu(x: Tensor) -> Tensor:
    '''Like torch.nn.function.relu(x, inplace=False).'''
    return maximum(x, 0)

utils.test_relu(Tensor)
# %%

def _matmul2d(x: Arr, y: Arr) -> Arr:
    '''Matrix multiply restricted to the case where both inputs are exactly 2D.'''
    return x @ y

def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return grad_out @ y.T

def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return x.T @ grad_out

matmul = wrap_forward_fn(_matmul2d)

BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)

utils.test_matmul2d(Tensor)
# %%
