from typing import Literal
import torch
import torch.nn as nn
from .feature import Feature


class FeatureStore(nn.Module):
    def __init__(self):
        super().__init__()
        self.user = nn.ModuleDict()
        self.item = nn.ModuleDict()

    def forward(
        self,
        user_idx: torch.Tensor=None,
        item_idx: torch.Tensor=None,
    ) -> dict[str, torch.Tensor]:
        features = {}

        if user_idx is not None:
            features.update({
                name: feature[user_idx]
                for name, feature in self.user.items()
            })

        if item_idx is not None:
            features.update({
                name: feature[item_idx]
                for name, feature in self.item.items()
            })

        return features

    def register(
        self,
        name: str,
        entity: Literal["user", "item"],
        data: torch.Tensor,
    ) -> None:
        feature = Feature(
            name=name,
            entity=entity,
            data=data,
        )

        if entity=="user":
            self.user[name] = feature
        elif entity=="item":
            self.item[name] = feature
        else:
            raise ValueError(f"Unknown entity: {entity}")