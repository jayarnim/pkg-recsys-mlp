from .elbo import build as build_elbo
from .optimizer import build as build_optimizer

from .engine import Engine
import torch.nn as nn
from ...config.config.trainer import TrnCfg


def build(
    model: nn.Module, 
    cfg: TrnCfg,
) -> Engine:
    kwargs = dict(
        params=model.parameters(),
        cfg=cfg.optimizer,
    )
    optimizer = build_optimizer(**kwargs)

    kwargs = dict(
        cfg=cfg.elbo,
    )
    elbo = build_elbo(**kwargs)

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        elbo=elbo,
    )
    return Engine(**kwargs)