import torch
import torch.nn as nn
from .base import EmbeddingLayer
from .registry import register


@register("idx")
class IDXEmbeddingLayer(EmbeddingLayer):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
    ):
        super().__init__()

        PADDING_IDX = 0

        kwargs = dict(
            num_embeddings=num_users+2, 
            embedding_dim=embedding_dim,
            padding_idx=PADDING_IDX,
        )
        user_embedding = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=num_items+2, 
            embedding_dim=embedding_dim,
            padding_idx=PADDING_IDX,
        )
        item_embedding = nn.Embedding(**kwargs)

        components = dict(
            user=user_embedding,
            item=item_embedding,
        )
        self.embedding = nn.ModuleDict(components)

        self.init_embeddings()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.embedding["user"](user_idx)
        item_emb = self.embedding["item"](item_idx)
        return user_emb, item_emb

    def init_embeddings(self):
        for name, emb in self.embedding.items():
            kwargs = dict(
                tensor=emb.weight, 
                mean=0.0, 
                std=0.01,
            )
            nn.init.normal_(**kwargs)