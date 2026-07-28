import torch
from .base import MatchingFunctionLayer
from .registry import register


@register("mf")
class MatrixFactorizationLayer(MatchingFunctionLayer):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        *args,
    ) -> torch.Tensor:
        return torch.stack(args).prod(dim=0)