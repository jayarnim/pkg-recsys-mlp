import torch
from .registry import register


@register("default")
def default(
    interactions: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    rows, cols = interactions.nonzero(as_tuple=True)

    hist_indices = [
        [] 
        for _ in range(len(interactions))
    ]
    
    for row, col in zip(rows.tolist(), cols.tolist()):
        hist_indices[row].append(col)

    hist_indices = [
        torch.tensor(indices, dtype=torch.long)
        for indices in hist_indices
    ]

    return hist_indices