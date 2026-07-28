import torch
from .base import CombinationLayer
from .registry import register


@register("cat")
class Concatenation(CombinationLayer):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        *args,
    ) -> torch.Tensor:
        return torch.cat(
            tensors=args, 
            dim=-1,
        )