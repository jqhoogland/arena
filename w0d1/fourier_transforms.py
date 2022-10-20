#%%

from typing import Callable, Optional

import ipywidgets as wg
import numpy as np
from numpy.typing import NDArray
import plotly.express as px
import plotly.graph_objs as go
from fancy_einsum import einsum

import utils

#%%

# Discrete FTs

def DFT_1d_v1(arr: NDArray, inverse = False) -> NDArray:
    """
    Returns the DFT of the array `arr`, using the equation above.
    """

    n = len(arr)
    coeffs = np.ones((n, n), dtype=np.complex128)
    
    w_exp_sign = 1 if inverse else -1
    w = np.exp((w_exp_sign * 2.j * np.pi) / n) 

    for i in range(1, len(arr)):
        for j in range(1, len(arr)):
            coeffs[i, j] = w ** (i * j)
    
    if inverse:
        return coeffs @ arr / n
    
    return coeffs @ arr

def DFT_1d_v2(arr: NDArray, inverse = False) -> NDArray:
    """
    Returns the DFT of the array `arr`, using the equation above.
    """

    n = len(arr)
    
    w_exp_sign = 1 if inverse else -1
    exponents = np.outer(np.arange(n), np.arange(n)) * w_exp_sign * (2.j * np.pi / n)
    coeffs = np.exp(exponents)
    
    if inverse:
        return coeffs @ arr / n
    
    return coeffs @ arr


def test_DFT_func_1(DFT_1d, x=np.linspace(-1, 1), function=np.square) -> None:

    y = function(x)

    y_DFT_actual = DFT_1d(y)
    y_reconstructed_actual = DFT_1d(y_DFT_actual, inverse=True)

    y_DFT_expected = np.fft.fft(y)

    np.testing.assert_allclose(y_DFT_actual, y_DFT_expected, atol=1e-10, err_msg="DFT failed")
    np.testing.assert_allclose(y_reconstructed_actual, y, atol=1e-10, err_msg="Inverse DFT failed")

def test_DFT_func_2(DFT_1d):
    # Test a known function.
    x = np.array([1., 2-1j, -1j,-1 + 2j ]).T
    y_DFT_actual = DFT_1d(x)
    y_DFT_pred = np.array([2., -2-2j, -2j, 4+4j]).T
    np.testing.assert_allclose(y_DFT_actual, y_DFT_pred, atol=1e-10, err_msg="DFT failed")

test_DFT_func_1(DFT_1d_v2)
test_DFT_func_2(DFT_1d_v2)

# %%

# Integration helper

from typing import Callable


def integrate_function(func: Callable, x0: float, x1: float, n_samples = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.

    You should use the Left Rectangular Approximation Method (LRAM).
    """
    x = np.linspace(x0, x1, n_samples)
    dx = x[1] - x[0]
    y = func(x) # Assumes already vectorized.
    
    return np.sum(y) * dx

def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Computes the integral of the function x -> func1(x) * func2(x).
    """
    x = np.linspace(x0, x1, n_samples)
    dx = x[1] - x[0]
    y = func1(x) * func2(x) # Assumes already vectorized.
    
    return np.sum(y) * dx   

utils.test_integrate_function(integrate_function)
utils.test_integrate_product(integrate_product)

#%%

# Continuous FTs

def calculate_fourier_series(func: Callable, max_freq: int = 50) \
    -> tuple[tuple[float, NDArray, NDArray], Callable]:
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].

    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """
    a_0 = integrate_function(func, -np.pi, np.pi) / np.pi
    A_n = np.array([
        integrate_product(func, lambda x: np.cos(n * x), -np.pi, np.pi) / np.pi
        for n in range(1, max_freq + 1)
    ])
    B_n = np.array([
        integrate_product(func, lambda x: np.sin(n * x), -np.pi, np.pi) / np.pi
        for n in range(1, max_freq + 1)
    ])

    def func_approx(x):
        return (
            a_0 / 2. 
            + A_n @ np.cos(np.arange(1, max_freq + 1) * x)
            + B_n @ np.sin(np.arange(1, max_freq + 1) * x)
        )

    return (a_0, A_n, B_n), np.vectorize(func_approx)

step_func = lambda x: 1 * (x > 0)
utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)
# %%
