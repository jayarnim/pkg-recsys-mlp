from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class MatchingFunctionLayer(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
