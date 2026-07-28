from . import embedding
from .embedding.registry import EMBEDDING_REGISTRY
from .embedding.base import EmbeddingLayer


def build(
    name: str, 
    **kwargs,
) -> EmbeddingLayer:
    cls = EMBEDDING_REGISTRY[name]
    return cls(**kwargs)