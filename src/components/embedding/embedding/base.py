from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, 
        user_info: torch.Tensor, 
        item_info: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError