from .predictor import Predictor
from .calculator import Calculator
from .evaluator import Evaluator
import torch.nn as nn
from msr.constants import *


def build(
    model: nn.Module, 
    cutoff: list[int],
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
    rating_col: str=DEFAULT_RATING_COL,
    prediction_col: str=DEFAULT_PREDICTION_COL,
) -> Evaluator:
    kwargs = dict(
        model=model,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        prediction_col=prediction_col,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        cutoff=cutoff,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        prediction_col=prediction_col,
    )
    calculator = Calculator(**kwargs)

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
    )
    return Evaluator(**kwargs)