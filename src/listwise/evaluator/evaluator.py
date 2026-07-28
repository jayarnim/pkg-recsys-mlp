import pandas as pd
import torch
from core.listwise.dataloader import DataLoader
from .predictor import Predictor
from .calculator import Calculator


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator(object):
    def __init__(
        self, 
        predictor: Predictor, 
        calculator: Calculator,
    ):
        super().__init__()
        self.predictor = predictor
        self.calculator = calculator

    def __call__(
        self, 
        dataloader: DataLoader,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        kwargs = dict(
            dataloader=dataloader,
        )
        result = self.predictor(**kwargs)

        kwargs = dict(
            result=result,
        )
        metrics_sheet = self.calculator(**kwargs)

        return result, metrics_sheet