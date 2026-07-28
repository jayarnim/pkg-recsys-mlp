import torch
import torch.nn as nn
from .base import EmbeddingLayer
from .registry import register


@register("history")
class HistoryEmbeddingLayer(EmbeddingLayer):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
    ):
        super().__init__()
        kwargs = dict(
            in_features=num_items,
            out_features=embedding_dim,
            bias=False,
        )
        user_projection = nn.Linear(**kwargs)

        kwargs = dict(
            in_features=num_users,
            out_features=embedding_dim,
            bias=False,
        )
        item_projection = nn.Linear(**kwargs)

        components = dict(
            user=user_projection,
            item=item_projection,
        )
        self.projection = nn.ModuleDict(components)

    def forward(
        self, 
        user_interaction: torch.Tensor, 
        item_interaction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_count = user_interaction.sum(dim=1, keepdim=True).clamp(min=1).sqrt()
        item_count = item_interaction.sum(dim=1, keepdim=True).clamp(min=1).sqrt()

        user_emb = self.projection["user"](user_interaction) / user_count
        item_emb = self.projection["item"](item_interaction) / item_count

        return user_emb, item_emb