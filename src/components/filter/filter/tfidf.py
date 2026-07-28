import torch
from sklearn.feature_extraction.text import TfidfTransformer
from .registry import register


@register("tfidf")
def tfidf(
    interactions: torch.Tensor,
    max_len: int,
    **kwargs,
) -> torch.Tensor:
    # compute tfidf
    tfidf = TfidfTransformer(norm=None)
    tfidf_matrix = tfidf.fit_transform(interactions)

    # ndarray -> tensor
    kwargs = dict(
        data=tfidf_matrix.toarray(),
        dtype=torch.float32,
    )
    tfidf_matrix_dense = torch.tensor(**kwargs)

    topk_indices = []

    for row in range(len(tfidf_matrix_dense)):
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
            vals, indices = torch.topk(tfidf_matrix_dense[row], k=max_len)
            topk_indices.append(indices.to(torch.long))

    return topk_indices
