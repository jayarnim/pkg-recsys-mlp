from .base import Criterion
from .registry import register
import torch
import torch.nn.functional as F


@register("bpr")
class BayesianPersonalizedRanking(Criterion):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(
        self, 
        pos: torch.Tensor, 
        neg: torch.Tensor,
    ) -> torch.Tensor:
        diff = pos - neg
        return -F.logsigmoid(diff).mean()