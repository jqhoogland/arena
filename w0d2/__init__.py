#%%
from collections import namedtuple

import numpy as np
import torch as t 
import einops
from fancy_einsum import einsum

# %%

test_input = t.tensor(
    [[0, 1, 2, 3, 4], 
    [5, 6, 7, 8, 9], 
    [10, 11, 12, 13, 14], 
    [15, 16, 17, 18, 19]], dtype=t.float
)

# %%
TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(
        output=t.tensor([0, 1, 2, 3]), 
        size=(4,), 
        stride=(1,)),
    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]), 
        size=(5, ), 
        stride=(1, )),
    TestCase(
        output=t.tensor([0, 5, 10, 15]), 
        size=(4,), 
        stride=(5,)),
    TestCase(
        output=t.tensor([[0, 1, 2], [5, 6, 7]]), 
        size=(2, 3), 
        stride=(5, 1)),
    TestCase(
        output=t.tensor([[0, 1, 2], [10, 11, 12]]), 
        size=(2, 3), 
        stride=(10, 1)),
    TestCase(
        output=t.tensor([[0, 0, 0], [11, 11, 11]]), 
        size=(2, 3),
        stride=(11, 0)),    
    TestCase(
        output=t.tensor([0, 6, 12, 18]), 
        size=(4,), 
        stride=(6,)),
    TestCase(
        output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]), 
        size=(2, 1, 3), 
        stride=(9, 0, 1)),
    TestCase(
        output=t.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]]]]),
        size=(2, 2, 2, 2),
        stride=(12, 4, 2, 1)),
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

