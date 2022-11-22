class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim
        super().__init__()

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both."""
        end = (len(input.shape) - 1 if self.end_dim == -1 else self.end_dim) + 1
        new_shape = (*input.shape[: self.start_dim], -1, *input.shape[end:])
        return input.reshape(new_shape)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


utils.test_flatten(Flatten)
# %%


def uniform_random(
    size: t.Size | tuple, min_: float, max_: float | None = None
) -> t.Tensor:
    if max_ is None:
        # Then we sample from -min_ to min_
        min_, max_ = -min_, min_

    return t.rand(size) * (max_ - min_) + min_
