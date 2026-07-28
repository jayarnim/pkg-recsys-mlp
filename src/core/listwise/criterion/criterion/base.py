from abc import ABC, abstractmethod
import torch


class Criterion(ABC):
    @abstractmethod
    def __call__(
        self, 
        pos: torch.Tensor, 
        neg: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError