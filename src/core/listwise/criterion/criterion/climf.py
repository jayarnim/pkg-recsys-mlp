from .base import Criterion
from .registry import register
import torch
import torch.nn.functional as F


@register("climf")
class CollaborativeLessIsMoreFiltering(Criterion):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(
        self, 
        pos: torch.Tensor, 
        neg: torch.Tensor,
    ) -> torch.Tensor:
        diff = neg - pos.unsqueeze(1)
        max_pos_term  = F.logsigmoid(pos)
        min_diff_term = F.logsigmoid(-diff).sum(dim=1)
        return -(max_pos_term + min_diff_term).mean()