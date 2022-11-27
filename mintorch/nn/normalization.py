import einops
import torch as t
from torch import nn

# from arena.mintorch.nn.containers import Module


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        super().__init__()

        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(normalized_shape))
            self.bias = nn.Parameter(t.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: t.Tensor) -> t.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        x_norm = (x - mean) / t.sqrt(var + self.eps)

        if self.elementwise_affine:
            return x_norm * self.weight + self.bias

        return x_norm


# utils.test_layernorm_mean_1d(LayerNorm)
# utils.test_layernorm_mean_2d(LayerNorm)
# utils.test_layernorm_std(LayerNorm)
# utils.test_layernorm_exact(LayerNorm)
# utils.test_layernorm_backward(LayerNorm)


class GroupNorm2d(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine=True,
        device: t.device | str | None = None,
        dtype: t.dtype | None = None,
    ) -> None:
        assert num_channels % num_groups == 0

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.device = device
        self.dtype = dtype

        super().__init__()

        if self.affine:
            self.weight = nn.Parameter(t.ones(num_channels, device=device, dtype=dtype))
            self.bias = nn.Parameter(t.zeros(num_channels, device=device, dtype=dtype))

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = einops.rearrange(x, "b (g c) h w -> b g c h w", g=self.num_groups)

        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        x = (x - mean) / t.sqrt(var + self.eps)
        x = einops.rearrange(x, "b g c h w -> b (g c) h w")

        if self.affine:
            return x * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(
                -1
            ).unsqueeze(-1)

        return x
