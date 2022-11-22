import numpy as np
import torch as t

from arena.torchish.nn.activations import Module


class ReLU(Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return (x > 0) * x


# utils.test_relu(ReLU)


class GELU(Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        # Exact
        return 0.5 * x * (1.0 + t.erf(x / np.sqrt(2.0)))

        # Approx 1
        # return x * t.sigmoid(1.702 * x)

        # Approx 2
        # return 0.5 * x * (1. + t.tanh(np.sqrt(2. / np.pi) * (x + 0.044715 * x ** 3)))


# utils.plot_gelu(GELU)
