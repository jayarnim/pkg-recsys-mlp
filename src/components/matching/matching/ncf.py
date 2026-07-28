import torch
import torch.nn as nn
from .base import MatchingFunctionLayer
from .registry import register
from ...functions.generator import fc_block


@register("ncf")
class NeuralCollaborativeFilteringLayer(MatchingFunctionLayer):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
        **kwargs,
    ):
        super().__init__()

        kwargs = dict(
            input_dim=embedding_dim*2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        components = list(fc_block(**kwargs))
        self.mlp = nn.Sequential(*components)

    def forward(
        self, 
        *args,
    ) -> torch.Tensor:
        X = torch.cat(
            tensors=args, 
            dim=-1,
        )
        return self.mlp(X)