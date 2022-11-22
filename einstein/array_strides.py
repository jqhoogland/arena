#%%
from collections import namedtuple

import einops
import numpy as np
import torch as t
from fancy_einsum import einsum

from arena.einstein import utils

# %%

test_input = t.tensor(
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
    dtype=t.float,
)

# %%
TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(output=t.tensor([0, 1, 2, 3]), size=(4,), stride=(1,)),
    TestCase(output=t.tensor([0, 1, 2, 3, 4]), size=(5,), stride=(1,)),
    TestCase(output=t.tensor([0, 5, 10, 15]), size=(4,), stride=(5,)),
    TestCase(output=t.tensor([[0, 1, 2], [5, 6, 7]]), size=(2, 3), stride=(5, 1)),
    TestCase(output=t.tensor([[0, 1, 2], [10, 11, 12]]), size=(2, 3), stride=(10, 1)),
    TestCase(output=t.tensor([[0, 0, 0], [11, 11, 11]]), size=(2, 3), stride=(11, 0)),
    TestCase(output=t.tensor([0, 6, 12, 18]), size=(4,), stride=(6,)),
    TestCase(
        output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]), size=(2, 1, 3), stride=(9, 0, 1)
    ),
    TestCase(
        output=t.tensor(
            [
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                [[[12, 13], [14, 15]], [[16, 17], [18, 19]]],
            ]
        ),
        size=(2, 2, 2, 2),
        stride=(12, 4, 2, 1),
    ),
]
for (i, case) in enumerate(test_cases):
    if (case.size is None) or (case.stride is None):
        print(f"Test {i} failed: attempt missing.")
    else:
        actual = test_input.as_strided(size=case.size, stride=case.stride)
        if (case.output != actual).any():
            print(f"Test {i} failed:")
            print(f"Expected: {case.output}")
            print(f"Actual: {actual}")
        else:
            print(f"Test {i} passed!")

# %%


def as_strided_trace(mat: t.Tensor) -> t.Tensor:
    """
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    """

    # I'm going to assume this wants only rank-2 matrices
    n_cols = mat.shape[1]
    return mat.as_strided(size=(n_cols,), stride=(n_cols + 1,)).sum()


utils.test_trace(as_strided_trace)

# %%


def as_strided_mv(mat: t.Tensor, vec: t.Tensor) -> t.Tensor:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    """

    n_rows, n_cols = mat.shape
    v_stride = vec.stride()

    return (mat * vec.as_strided(size=(n_rows, n_cols), stride=(0, v_stride[0]))).sum(1)


utils.test_mv(as_strided_mv)
utils.test_mv2(as_strided_mv)

# %%


def as_strided_mm(matA: t.Tensor, matB: t.Tensor) -> t.Tensor:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    """
    a_rows, a_cols = matA.shape
    a_stride = matA.stride()  # type: ignore

    _, b_cols = matB.shape  # Assumes shapes are already checked
    b_stride = matB.stride()

    return (
        matA.as_strided(size=(a_rows, a_cols, b_cols), stride=(a_stride[0], a_stride[1], 0))  # type: ignore
        * matB.as_strided(size=(a_rows, a_cols, b_cols), stride=(0, b_stride[0], b_stride[1]))  # type: ignore
    ).sum(1)


utils.test_mm(as_strided_mm)
utils.test_mm2(as_strided_mm)
# %%
