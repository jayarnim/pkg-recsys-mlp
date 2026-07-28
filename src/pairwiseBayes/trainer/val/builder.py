from .predictor import Predictor
from .calculator import Calculator
from .engine import Engine
import torch.nn as nn
from ...config.config.trainer import ValCfg
from msr.constants import *


def build(
    model: nn.Module, 
    cfg: ValCfg,
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
    rating_col: str=DEFAULT_RATING_COL,
    prediction_col: str=DEFAULT_PREDICTION_COL,
) -> Engine:
    predictor = Predictor(
        model=model,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        prediction_col=prediction_col,
    )

    calculator = Calculator(
        **vars(cfg.metric),
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        prediction_col=prediction_col,
    )

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
    )
    return Engine(**kwargs)