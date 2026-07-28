from . import combination
from .combination.registry import COMB_REGISTRY
from .combination.base import CombinationLayer


def build(
    name: str,
    **kwargs,
) -> CombinationLayer:
    cls = COMB_REGISTRY[name]
    return cls(**kwargs)