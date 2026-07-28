import torch
from .base import CombinationLayer
from .registry import register


@register("mean")
class ElementwiseMean(CombinationLayer):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        *args,
    ) -> torch.Tensor:
        return torch.stack(args).mean(dim=0)