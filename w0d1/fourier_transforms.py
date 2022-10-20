#%%

from typing import Callable, Optional

import ipywidgets as wg
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from fancy_einsum import einsum

import utils

#%%


def DFT_1d(arr: np.ndarray) -> np.ndarray:
    """ """
    n = len(arr)
    coeffs = np.ones((n, n), dtype=np.complex128)

    for i in range(1, len(arr)):
        for j in range(1, len(arr)):
            coeffs[i, j] = np.exp((i + j) * -2.0j * np.pi / n)

    return coeffs @ arr


def test_DFT_func(DFT_1d, x=np.linspace(-1, 1), function=np.square) -> None:
    print("Starting")

    y = function(x)

    y_DFT_actual = DFT_1d(y)
    y_reconstructed_actual = DFT_1d(y_DFT_actual)

    y_DFT_expected = np.fft.fft(y)

    np.testing.assert_allclose(
        y_DFT_actual, y_DFT_expected, atol=1e-10, err_msg="DFT failed"
    )
    np.testing.assert_allclose(
        y_reconstructed_actual, y, atol=1e-10, err_msg="Inverse DFT failed"
    )

    print("Success")


test_DFT_func(DFT_1d)

# %%
