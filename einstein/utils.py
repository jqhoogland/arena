import itertools
from typing import Callable, Optional, Union

import numpy as np
import PIL
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, reduce, repeat
from fancy_einsum import einsum
from IPython.display import HTML, display
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots


def display_array_as_img(img_array):
    """
    Displays a numpy array as an image

    Two options:
        img_array.shape = (height, width) -> interpreted as monochrome
        img_array.shape = (3, height, width) -> interpreted as RGB
    """
    shape = img_array.shape
    assert len(shape) == 2 or (
        shape[0] == 3 and len(shape) == 3
    ), "Incorrect format (see docstring)"

    if len(shape) == 3:
        img_array = rearrange(img_array, "c h w -> h w c")
    height, width = img_array.shape[:2]

    fig = px.imshow(img_array, zmin=0, zmax=255, color_continuous_scale="gray")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict.fromkeys("tblr", 0),
        height=height,
        width=width,
    )
    fig.show(config=dict(displayModeBar=False))


def test_einsum_trace(einsum_trace):
    mat = np.random.randn(3, 3)
    np.testing.assert_almost_equal(einsum_trace(mat), np.trace(mat))
    print("All tests in `test_einsum_trace` passed!")


def test_einsum_mv(einsum_mv):
    mat = np.random.randn(2, 3)
    vec = np.random.randn(3)
    np.testing.assert_almost_equal(einsum_mv(mat, vec), mat @ vec)
    print("All tests in `test_einsum_mv` passed!")


def test_einsum_mm(einsum_mm):
    mat1 = np.random.randn(2, 3)
    mat2 = np.random.randn(3, 4)
    np.testing.assert_almost_equal(einsum_mm(mat1, mat2), mat1 @ mat2)
    print("All tests in `test_einsum_mm` passed!")


def test_einsum_inner(einsum_inner):
    vec1 = np.random.randn(3)
    vec2 = np.random.randn(3)
    np.testing.assert_almost_equal(einsum_inner(vec1, vec2), np.dot(vec1, vec2))
    print("All tests in `test_einsum_inner` passed!")


def test_einsum_outer(einsum_outer):
    vec1 = np.random.randn(3)
    vec2 = np.random.randn(4)
    np.testing.assert_almost_equal(einsum_outer(vec1, vec2), np.outer(vec1, vec2))
    print("All tests in `test_einsum_outer` passed!")


def test_trace(trace_fn):
    for n in range(10):
        assert (
            trace_fn(t.zeros((n, n), dtype=t.long)) == 0
        ), f"Test failed on zero matrix with size ({n}, {n})"
        assert (
            trace_fn(t.eye(n, dtype=t.long)) == n
        ), f"Test failed on identity matrix with size ({n}, {n})"
        x = t.randint(0, 10, (n, n))
        expected = t.trace(x)
        actual = trace_fn(x)
        assert (
            actual == expected
        ), f"Test failed on randmly initialised matrix with size ({n}, {n})"
    print("All tests in `test_trace` passed!")


def test_mv(mv_fn):
    mat = t.randn(3, 4)
    vec = t.randn(4)
    mv_actual = mv_fn(mat, vec)
    mv_expected = mat @ vec
    t.testing.assert_close(mv_actual, mv_expected)
    print("All tests in `test_mv` passed!")


def test_mv2(mv_fn):
    big = t.randn(30)
    mat = big.as_strided(size=(3, 4), stride=(2, 4), storage_offset=8)
    vec = big.as_strided(size=(4,), stride=(3,), storage_offset=8)
    mv_actual = mv_fn(mat, vec)
    mv_expected = mat @ vec
    t.testing.assert_close(mv_actual, mv_expected)
    print("All tests in `test_mv2` passed!")


def test_mm(mm_fn):
    matA = t.randn(3, 4)
    matB = t.randn(4, 5)
    mm_actual = mm_fn(matA, matB)
    mm_expected = matA @ matB
    t.testing.assert_close(mm_actual, mm_expected)
    print("All tests in `test_mm` passed!")


def test_mm2(mm_fn):
    big = t.randn(30)
    matA = big.as_strided(size=(3, 4), stride=(2, 4), storage_offset=8)
    matB = big.as_strided(size=(4, 5), stride=(3, 2), storage_offset=8)
    mm_actual = mm_fn(matA, matB)
    mm_expected = matA @ matB
    t.testing.assert_close(mm_actual, mm_expected)
    print("All tests in `test_mm2` passed!")
