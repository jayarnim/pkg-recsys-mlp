import torch
from .base import CombinationLayer
from .registry import register


@register("prod")
class ElementwiseProduct(CombinationLayer):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        *args,
    ) -> torch.Tensor:
        return torch.stack(args).prod(dim=0)