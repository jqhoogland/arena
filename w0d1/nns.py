#%%
import ipywidgets as wg
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import einsum
from torch import nn, optim

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
x_all = torch.cat([x_cos, x_sin], dim=0).T

model = nn.Sequential(
    nn.Linear(2 * NUM_FREQUENCIES, 1, bias=True),
    nn.Flatten(0, 1)
)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    y_pred = model(x_all)
    # y_pred = 0.5 * a_0 + einsum("freq x, freq -> x", x_cos, A_n) + einsum("freq x, freq -> x", x_sin, B_n)

    loss = torch.mean((y_pred - y).pow(2))

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        A_n = list(model.parameters())[0].detach().numpy().squeeze()[:NUM_FREQUENCIES]
        B_n = list(model.parameters())[0].detach().numpy().squeeze()[NUM_FREQUENCIES:]
        a_0 = list(model.parameters())[1].item()
        y_pred_list.append(y_pred.cpu().detach().numpy())
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])

    loss.backward()
    optimizer.step()
    model.zero_grad()

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
