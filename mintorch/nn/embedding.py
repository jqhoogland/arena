import torch as t
from torch import nn

from arena.mintorch.nn.containers import Module


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        super().__init__()

        self.weight = nn.Parameter(
            t.normal(0.0, 1.0, size=(num_embeddings, embedding_dim))
        )

    def forward(self, x: t.LongTensor) -> t.Tensor:
        """For each integer in the input, return that row of the embedding."""
        return self.weight[x]
        # return t.index_select(self.weight, dim=0, index=x)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        )


# utils.test_embedding(Embedding)
