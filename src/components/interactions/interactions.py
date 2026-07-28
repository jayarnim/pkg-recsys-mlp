import torch
import torch.nn as nn
import pandas as pd
from msr.constants import *
from .mat import user_item_mat


class Interactions(nn.Module):
    def __init__(
        self, 
        df: pd.DataFrame, 
        num_users: int,
        num_items: int,
        user_col: str=DEFAULT_USER_COL,
        item_col: str=DEFAULT_ITEM_COL,
    ):
        super().__init__()

        mat = user_item_mat(
            df=df,
            num_users=num_users,
            num_items=num_items,
            user_col=user_col,
            item_col=item_col,
        )
        self.register_buffer(
            name="user_mat", 
            tensor=mat.float(),
        )
        self.register_buffer(
            name="item_mat", 
            tensor=mat.T.contiguous().float(),
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # USER ==========
        # SEARCH
        user_slice = self.user_mat[user_idx]
        # VALID IDX FILTERING
        VALID = item_idx >= 2
        # GENERATE BATCH IDX
        batch_idx = torch.arange(user_idx.size(0), device=user_idx.device)
        # TARGET IDX MASKING
        user_slice[batch_idx[VALID], item_idx[VALID]] = 0.0
        # DROP PADDING IDX
        user_slice = user_slice[:, 2:]

        # ITEM ==========
        # SEARCH
        item_slice = self.item_mat[item_idx]
        # VALID IDX FILTERING
        VALID = user_idx >= 2
        # GENERATE BATCH IDX
        batch_idx = torch.arange(item_idx.size(0), device=item_idx.device)
        # TARGET IDX MASKING
        item_slice[batch_idx[VALID], user_idx[VALID]] = 0.0
        # DROP PADDING IDX
        item_slice = item_slice[:, 2:]

        return user_slice, item_slice
