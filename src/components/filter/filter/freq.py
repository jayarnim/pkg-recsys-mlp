import torch
from .registry import register


@register("freq")
def freq(
    interactions: torch.Tensor,
    max_len: int,
    **kwargs,
) -> torch.Tensor:
    freq_target = interactions.sum(dim=0)

    topk_indices = []

    for row in range(len(interactions)):
        hist_count = int(interactions[row].sum().item())
        
        # padding only
        if hist_count == 0:
            indices = torch.tensor([0], dtype=torch.long)
            topk_indices.append(indices.to(torch.long))
        
        # all
        elif hist_count <= max_len:
            indices = interactions[row].nonzero(as_tuple=True)[0]
            topk_indices.append(indices.to(torch.long))
        
        # top-k selection
        else:
            hist_idx = interactions[row].nonzero(as_tuple=True)[0]
            scores = freq_target[hist_idx]
            vals, indices = torch.topk(scores, k=max_len)
            topk_indices.append(indices.to(torch.long))

    return topk_indices
