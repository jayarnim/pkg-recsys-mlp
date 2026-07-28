from . import matching
from .matching.registry import MATCHING_REGISTRY
from .matching.base import MatchingFunctionLayer


def build(
    name: str, 
    **kwargs,
) -> MatchingFunctionLayer:
    cls = MATCHING_REGISTRY[name]
    return cls(**kwargs)