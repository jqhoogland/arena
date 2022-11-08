# %%

from typing import Iterable

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from torch import nn, optim

from arena.w3d1 import utils
from arena.w3d1.optimizers import SGD, Adam, RMSprop, rosenbrocks_banana

# %%


class ExponentialLR:
    def __init__(self, optimizer, gamma):
        """Implements ExponentialLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
        """
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self):
        self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f"ExponentialLR(gamma={self.gamma})"


utils.test_ExponentialLR(ExponentialLR, SGD)

# %%


class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        """Implements StepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma

        self.t = 0

    def step(self):
        self.t += 1

        if self.t % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"


utils.test_StepLR(StepLR, SGD)
# %%


class MultiStepLR:
    def __init__(self, optimizer, milestones, gamma=0.1):
        """Implements MultiStepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma

        self.t = 0

    def step(self):
        self.t += 1

        if self.t in self.milestones:
            self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f"MultiStepLR(milestones={self.milestones}, gamma={self.gamma})"


utils.test_MultiStepLR(MultiStepLR, SGD)
# %%


def opt_fn_with_scheduler(
    fn: callable,
    xy: t.Tensor,
    optimizer_class,
    optimizer_kwargs,
    scheduler_class=None,
    scheduler_kwargs=dict(),
    n_iters: int = 100,
):
    """Optimize the a given function starting from the specified point.

    scheduler_class: one of the schedulers you've defined, either ExponentialLR, StepLR or MultiStepLR
    scheduler_kwargs: keyword arguments passed to your optimiser (e.g. gamma)
    """
    assert xy.requires_grad
    optimizer = optimizer_class([xy], **optimizer_kwargs)
    scheduler = scheduler_class(optimizer, **scheduler_kwargs) if scheduler_class else None

    out = [xy.detach().clone()]

    for _ in range(n_iters):
        optimizer.zero_grad()
        loss = fn(*xy)
        loss.backward()
        optimizer.step()
        out.append(xy.detach().clone())

        if scheduler:
            scheduler.step()

    return t.stack(out)

# %%

xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]
optimizers = [
    (SGD, dict(lr=1e-3, momentum=0.98)),
    (SGD, dict(lr=1e-3, momentum=0.98)),
]
schedulers = [
    (), # Empty list stands for no scheduler
    (ExponentialLR, dict(gamma=0.99)),
]

fig = utils.plot_optimization_with_schedulers(opt_fn_with_scheduler, rosenbrocks_banana, xy, optimizers, schedulers, x_range, y_range, show_min=True)

fig.show()

# %%
