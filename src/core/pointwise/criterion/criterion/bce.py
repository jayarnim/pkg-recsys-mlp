from .base import Criterion
from .registry import register
import torch
import torch.nn.functional as F


@register("bce")
class BinaryCrossEntropy(Criterion):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor,
    ) -> torch.Tensor:
        kwargs = dict(
            input=pred, 
            target=true, 
            reduction='mean',
        )
        return F.binary_cross_entropy_with_logits(**kwargs)