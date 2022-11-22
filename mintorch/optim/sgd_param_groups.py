# %%

from typing import Iterable

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from torch import nn, optim

from arena.mintorch.optim import utils

# %%


class SGD:
    param_groups: list[dict]

    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter | dict],
        **kwargs,
    ):
        """Implements SGD with momentum_buffer.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        :param kwargs.lr: learning rate
        :param kwargs.momentum_buffer: momentum_buffer
        :param kwargs.weight_decay: weight decay

        """

        kwargs.setdefault("momentum", 0.0)
        kwargs.setdefault("weight_decay", 0.0)

        params = list(params)

        if isinstance(params[0], dict):
            self.param_groups = [
                {
                    **kwargs,
                    **param,
                    "gs": [t.zeros_like(param) for param in param["params"]],
                }
                for param in params
            ]  # type: ignore
        else:
            self.param_groups = [
                {
                    "params": params,
                    **kwargs,
                    "gs": [t.zeros_like(param) for param in params],
                }
            ]

        # Check for duplicates
        for param in self.params:
            if list(self.params).count(param) > 1:
                raise ValueError("Duplicate parameters found")

    @property
    def params(self):
        return [
            param
            for param_group in self.param_groups
            for param in param_group["params"]
        ]

    def zero_grad(self) -> None:
        for param_group in self.param_groups:
            for param in param_group["params"]:
                param.grad = None

    def step(self) -> None:
        # Requires that the gradients have already been computed (with `backward()`)

        with t.inference_mode():
            for param_group in self.param_groups:
                lmbda = param_group["weight_decay"]
                mu = param_group["momentum"]
                lr = param_group["lr"]

                for param, g in zip(param_group["params"], param_group["gs"]):
                    if param.grad is None:
                        continue
                    if param.grad.is_sparse:
                        raise RuntimeError("Sparse gradients are not supported")

                    if lmbda:
                        param.grad += lmbda * param

                    if mu:
                        if (g == 0.0).all():  # (first iteration)
                            g.add_(param.grad)
                        else:
                            g.mul_(mu).add_(param.grad)

                        param.grad = g

                    param.add_(-lr * param.grad)


utils.test_sgd_param_groups(SGD)

# TODO: Something's going wrong here

# %%
