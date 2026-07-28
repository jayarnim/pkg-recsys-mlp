import torch
from ..state import State
from .predictor import Predictor
from .calculator import Calculator
from core.listwise.dataloader import DataLoader


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
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
        state: State,
    ) -> None:
        kwargs = dict(
            dataloader=dataloader,
            state=state,
        )
        result = self.predictor(**kwargs)
        
        kwargs = dict(
            result=result,
            state=state,
        )
        self.calculator(**kwargs)