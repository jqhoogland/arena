#%%
import ipywidgets as wg
import numpy as np
from numpy import einsum
import torch

from matplotlib import pyplot as plt

import utils

#%%

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-3

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-torch.pi, torch.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n * x) for n in range(1, NUM_FREQUENCIES + 1)])
x_sin = torch.stack([torch.sin(n * x) for n in range(1, NUM_FREQUENCIES + 1)])

a_0 = torch.randn((), device=device, dtype=dtype, requires_grad=True)
A_n = torch.randn(NUM_FREQUENCIES, device=device, dtype=dtype, requires_grad=True)
B_n = torch.randn(NUM_FREQUENCIES, device=device, dtype=dtype, requires_grad=True)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    y_pred = 0.5 * a_0 + A_n @ x_cos + B_n @ x_sin
    # y_pred = 0.5 * a_0 + einsum("freq x, freq -> x", x_cos, A_n) + einsum("freq x, freq -> x", x_sin, B_n)

    loss = torch.mean((y_pred - y).pow(2))

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0.detach().numpy().copy(), A_n.to("cpu").detach().numpy().copy(), B_n.to("cpu").detach().numpy().copy()])
        y_pred_list.append(y_pred.detach())

    loss.backward()

    with torch.no_grad():
        for coeff in [a_0, A_n, B_n]:
            coeff -= LEARNING_RATE * coeff.grad
            coeff.grad = None

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
