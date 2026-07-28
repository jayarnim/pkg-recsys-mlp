from typing import Literal
import torch
import torch.nn as nn


class Feature(nn.Module):
    def __init__(
        self,
        name: str,
        entity: Literal["user", "item"],
        data: torch.Tensor,
    ):
        super().__init__()
        self.name = name
        self.entity = entity
        self.register_buffer(
            name="data",
            tensor=data,
        )

    def __getitem__(
        self,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.data[idx]