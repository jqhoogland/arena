# %%

from typing import Iterable

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from torch import nn, optim

from arena.w3d1 import utils

# %%


def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


x_range = [-2, 2]
y_range = [-1, 3]
fig = utils.plot_fn(rosenbrocks_banana, x_range, y_range, log_scale=True)
fig
# %%


def opt_fn_with_sgd(
    fn: callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100
) -> t.Tensor:
    """
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    """
    assert xy.requires_grad
    optimizer = optim.SGD([xy], lr=lr, momentum=momentum)

    out = [xy.detach().clone()]

    for _ in range(n_iters):
        optimizer.zero_grad()
        loss = fn(*xy)
        loss.backward()
        optimizer.step()
        out.append(xy.detach().clone())

    return t.stack(out)


xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]

fig = utils.plot_optimization_sgd(
    opt_fn_with_sgd,
    rosenbrocks_banana,
    xy,
    x_range,
    y_range,
    lr=0.001,
    momentum=0.98,
    show_min=True,
)

fig.show()
fig
# %%


class SGD:
    params: list

    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.momentum_buffer = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        # Requires that the gradients have already been computed (with `backward()`)

        with t.inference_mode():
            for param, momentum in zip(self.params, self.momentum_buffer):
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported")

                if self.weight_decay:
                    param.grad += self.weight_decay * param

                if self.momentum:
                    if (momentum == 0.0).all():  # (first iteration)
                        momentum.add_(param.grad)
                    else:
                        momentum.mul_(self.momentum).add_(param.grad)

                    param.grad = momentum

                param.add_(-self.lr * param.grad)

    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        return "SGD(lr={}, momentum={}, weight_decay={})".format(
            self.lr, self.momentum, self.weight_decay
        )


utils.test_sgd(SGD)

# %%


class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        """Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.eps = eps

        self.momentum_buffer = [t.zeros_like(param) for param in self.params]
        self.sq_buffer = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        # Requires that the gradients have already been computed (with `backward()`)
        with t.inference_mode():
            for param, momentum, sq in zip(
                self.params, self.momentum_buffer, self.sq_buffer
            ):
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported")

                if self.weight_decay:
                    param.grad += self.weight_decay * param

                sq.mul_(self.alpha).add_((1 - self.alpha) * param.grad**2)

                if self.momentum:
                    momentum.mul_(self.momentum).add_(
                        param.grad / (sq.sqrt() + self.eps)
                    )
                    param.add_(-self.lr * momentum)
                else:
                    param.add_(-self.lr * param.grad / (sq.sqrt() + self.eps))

    def __repr__(self) -> str:
        return "RMSProp(lr={}, momentum={}, weight_decay={}, alpha={}, eps={})".format(
            self.lr, self.momentum, self.weight_decay, self.alpha, self.eps
        )


utils.test_rmsprop(RMSprop)

# %%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        """
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

        self.m_buffer = [t.zeros_like(param) for param in self.params]
        self.v_buffer = [t.zeros_like(param) for param in self.params]
        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        # Requires that the gradients have already been computed (with `backward()`)
        with t.inference_mode():
            for param, m, v in zip(self.params, self.m_buffer, self.v_buffer):
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported")

                if self.weight_decay:
                    param.grad += self.weight_decay * param

                m.mul_(self.betas[0]).add_((1 - self.betas[0]) * param.grad)
                v.mul_(self.betas[1]).add_((1 - self.betas[1]) * (param.grad**2))

                m_hat = m / (1 - (self.betas[0] ** self.t))
                v_hat = v / (1 - (self.betas[1] ** self.t))

                param.add_(-self.lr * m_hat / (v_hat.sqrt() + self.eps))
            
            self.t += 1

    def __repr__(self) -> str:
        return "Adam(lr={}, betas={}, weight_decay={}, alpha={}, eps={})".format(
            self.lr, self.betas, self.weight_decay, self.eps
        )


utils.test_adam(Adam)

# %%
