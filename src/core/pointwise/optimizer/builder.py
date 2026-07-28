from . import optimizer
from .optimizer.registry import OPTIMIZER_REGISTRY
from ..config.config.optimizer import OptimizerCfg
from collections.abc import Iterator
from torch.nn import Parameter
from torch.optim import Optimizer


def build(
    params: Iterator[Parameter], 
    cfg: OptimizerCfg,
) -> Optimizer:
    kwargs = dict(
        params=params, 
        **cfg.params,
    )
    cls = OPTIMIZER_REGISTRY[cfg.name]
    return cls(**kwargs)