from typing import Literal
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from msr.constants import *
from .mat import user_item_mat
from .filter import FILTER_REGISTRY


class Histories(nn.Module):
    def __init__(
        self, 
        df: pd.DataFrame, 
        num_users: int,
        num_items: int,
        anchor: Literal["user", "item"], 
        filter: str="default",
        max_len: int=None,
        user_col: str=DEFAULT_USER_COL,
        item_col: str=DEFAULT_ITEM_COL,
    ):
        super().__init__()

        # USER- ITEM INTERACTION MATRIX
        mat = user_item_mat(
            df=df,
            num_users=num_users,
            num_items=num_items,
            user_col=user_col,
            item_col=item_col,
        )

        # MATRIX -> RAGGED LIST
        histories = FILTER_REGISTRY[filter](
            interactions=(
                mat
                if anchor=="user"
                else mat.T
            ),
            max_len=max_len,
        )

        # RAGGED LIST -> PADDED TENSOR
        histories_padded = pad_sequence(
            sequences=histories, 
            batch_first=True, 
            padding_value=0,
        )

        if max_len:
            if histories_padded.size(1) < max_len:
                histories_padded = F.pad(
                    input=histories_padded,
                    pad=(0, max_len - histories_padded.size(1)),
                    value=0,
                )

        # BUFFER
        self.register_buffer(
            name="histories", 
            tensor=histories_padded,
        )        

    def forward(
        self,
        anchor_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        PADDING_IDX = 0
        OOV_IDX = 1

        # SEARCH ANCHOR HISTORY
        hist_slice = self.histories[anchor_idx, :]
        # MASK TO CURRENT TARGET IDX FROM HISTORY
        marking_target_idx = hist_slice==target_idx.unsqueeze(1)
        # MASK TO PADDING OR OOV
        marking_padding_idx = hist_slice==PADDING_IDX
        # MASK TO OOV
        marking_oov_idx = hist_slice==OOV_IDX
        # FINAL MASK
        mask = marking_target_idx | marking_padding_idx | marking_oov_idx
        return hist_slice, mask