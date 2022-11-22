#%%
from typing import Callable

import ipywidgets as wg
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import torch
from matplotlib import pyplot as plt
from numpy import einsum
from torch import nn, optim

from arena.fourier import utils

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

model = nn.Sequential(nn.Linear(2 * NUM_FREQUENCIES, 1, bias=True), nn.Flatten(0, 1))

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


def create_interactive_fourier_graph_widgets(
    calculate_fourier_series: Callable, func: Callable
):

    label = wg.Label("Number of terms in Fourier series: ")

    slider = wg.IntSlider(min=0, max=50, value=0)

    x = np.linspace(-np.pi, np.pi, 1000)
    y = func(x)

    fig = go.FigureWidget(
        data=[
            go.Scatter(x=x, y=y, name="Original function", mode="lines"),
            go.Scatter(x=x, y=y, name="Reconstructed function", mode="lines"),
        ],
        layout=go.Layout(
            title_text=r"Original vs reconstructed",
            template="simple_white",
            margin_t=100,
        ),
    )

    def respond_to_slider(change):
        max_freq = slider.value
        coeffs, func_approx = calculate_fourier_series(func, max_freq)
        fig.data[1].y = np.vectorize(func_approx)(x)

    slider.observe(respond_to_slider)

    respond_to_slider("unimportant text to trigger first response")

    box_layout = wg.Layout(
        border="solid 1px black", padding="20px", margin="20px", width="80%"
    )

    return wg.VBox([wg.HBox([label, slider], layout=box_layout), fig])


def bool_list(i, m):
    l = [True] + [False for j in range(m)]
    l[i] = True
    return l


def create_interactive_fourier_graph(
    calculate_fourier_series: Callable, func: Callable
):

    x = np.linspace(-np.pi, np.pi, 300)
    y = func(x)

    sliders = [
        dict(
            active=0,
            currentvalue_prefix="Max frequency: ",
            pad_t=40,
            steps=[
                dict(method="update", args=[{"visible": bool_list(i, 30)}])
                for i in range(30)
            ],
        )
    ]

    data = [go.Scatter(x=x, y=y, name="Original function", mode="lines")]

    for max_freq in range(30):
        data.append(
            go.Scatter(
                x=x,
                y=calculate_fourier_series(func, max_freq)[1](x),
                name="Reconstructed function",
                mode="lines",
                visible=(max_freq == 0),
            )
        )

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title_text=r"Original vs reconstructed",
            template="simple_white",
            margin_t=100,
            sliders=sliders,
        ),
    )

    return fig


TARGET_FUNC = np.sin
NUM_FREQUENCIES = 4
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6


def get_title_from_coeffs(a_0, A_n, B_n):
    A_n_coeffs = " + ".join(
        [
            f"{a_n:.2f}" + r"\cos{" + (str(n) if n > 1 else "") + " x}"
            for (n, a_n) in enumerate(A_n, 1)
        ]
    )
    B_n_coeffs = " + ".join(
        [
            f"{b_n:.2f}" + r"\sin{" + (str(n) if n > 1 else "") + " x}"
            for (n, b_n) in enumerate(B_n, 1)
        ]
    )
    return r"$y = " + f"{0.5*a_0:.2f}" + " + " + A_n_coeffs + " + " + B_n_coeffs + "$"


def visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list):

    label = wg.Label("Number of steps: ")

    slider = wg.IntSlider(min=0, max=TOTAL_STEPS - 100, value=0, step=100)

    fig = go.FigureWidget(
        data=[go.Scatter(x=x, y=y, mode="lines", marker_color="blue")]
        + [
            go.Scatter(
                x=x,
                y=y_pred_list[i],
                mode="lines",
                marker_color="rgba(100, 100, 100, 0.1)",
            )
            for i in range(len(y_pred_list))
        ],
        layout=go.Layout(
            title_text=r"Original vs reconstructed",
            template="simple_white",
            margin_t=100,
            showlegend=False,
        ),
    )

    def respond_to_slider(change):
        idx = slider.value // 100
        with fig.batch_update():
            fig.update_layout(title_text=get_title_from_coeffs(*coeffs_list[idx]))
            for i in range(len(list(fig.data)) - 1):
                fig.data[i + 1]["marker"]["color"] = (
                    "red" if i == idx else "rgba(100, 100, 100, 0.1)"
                )

    slider.observe(respond_to_slider)

    respond_to_slider("unimportant text to trigger first response")

    box_layout = wg.Layout(
        border="solid 1px black", padding="20px", margin="20px", width="80%"
    )

    return wg.VBox([wg.HBox([label, slider], layout=box_layout), fig])


visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
