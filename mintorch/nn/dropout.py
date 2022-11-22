import torch as t
from torch import nn

from arena.mintorch.nn.containers import Module

class Dropout(Module):
    def __init__(self, p: float):
        self.p = p

        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training and self.p:
            mask = t.empty(x.shape).bernoulli_(1 - self.p)
            return x * mask * (1 / (1 - self.p))

        return x


# utils.test_dropout_eval(Dropout)
# utils.test_dropout_training(Dropout)

# %%
