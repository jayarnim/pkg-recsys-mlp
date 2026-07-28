from .trn import build as build_trn
from .val import build as build_val
from .callbacks.earlystopping import EarlyStopping
from .callbacks.logger import Logger
from .callbacks.checkpointer import Checkpointer
from .trainer import Trainer
from ..config.config.trainer import TrainerCfg
from msr.constants import *
import torch.nn as nn


def build(
    model: nn.Module, 
    cfg: TrainerCfg,
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
    rating_col: str=DEFAULT_RATING_COL,
    prediction_col: str=DEFAULT_PREDICTION_COL,
) -> Trainer:
    kwargs = dict(
        model=model,
        cfg=cfg.trn,
    )
    trn = build_trn(**kwargs)

    kwargs = dict(
        model=model,
        cfg=cfg.val,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        prediction_col=prediction_col,
    )
    val = build_val(**kwargs)

    callbacks = [
        EarlyStopping(**vars(cfg.early_stopping)),
        Logger(),
        Checkpointer(),
    ]

    kwargs = dict(
        model=model,
        trn=trn,
        val=val,
        callbacks=callbacks,
        num_epochs=cfg.num_epochs,
    )
    return Trainer(**kwargs)