#%%
import ipywidgets as wg
import numpy as np
from matplotlib import pyplot as plt

import utils

#%%

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = np.linspace(-np.pi, np.pi, 2000)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n * x) for n in range(1, NUM_FREQUENCIES + 1)])
x_sin = np.array([np.sin(n * x) for n in range(1, NUM_FREQUENCIES + 1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    y_pred = a_0 / 2 + A_n @ x_cos + B_n @ x_sin

    if step % 100 == 0:
        loss = np.mean((y_pred - y) ** 2)
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    y_grad = 2 * (y_pred - y)

    a_0_grad = y_grad * 1 / 2
    A_n_grads = y_grad @ x_cos.T
    B_n_grads = y_grad @ x_sin.T

    a_0 -= LEARNING_RATE * a_0_grad
    A_n -= LEARNING_RATE * A_n_grads
    B_n -= LEARNING_RATE * B_n_grads

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)

# %%
