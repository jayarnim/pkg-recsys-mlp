from abc import ABC, abstractmethod
from functools import wraps
from .output import BayesModelOutput
import torch
import torch.nn as nn


class BayesModel(nn.Module, ABC):
    def __init__(self, kwargs):
        super().__init__()

        init_args = kwargs.copy()
        init_args.pop("self", None)
        init_args.pop("__class__", None)

        self.init_args = init_args

    def __init_subclass__(cls):
        super().__init_subclass__()

        predict = cls.predict

        @wraps(predict)
        def wrapped(self, *args, **kwargs):
            out = predict(self, *args, **kwargs)

            if not isinstance(out, BayesModelOutput):
                raise TypeError(
                    f"{cls.__name__}.predict() must return "
                    f"BayesModelOutput, got {type(out).__name__}."
                )

            return out

        cls.predict = wrapped

    @abstractmethod
    def forward(
        self, 
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, 
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> BayesModelOutput:
        raise NotImplementedError