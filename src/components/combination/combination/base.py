from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class CombinationLayer(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, 
        *args,
    ) -> torch.Tensor:
        raise NotImplementedError