import torch
import pandas as pd
from msr.constants import *


def user_item_mat(
    df: pd.DataFrame, 
    num_users: int,
    num_items: int,
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
) -> torch.Tensor:
    kwargs = dict(
        size=(num_users+2, num_items+2),
        dtype=torch.int32,
    )
    user_item_matrix = torch.zeros(**kwargs)

    kwargs = dict(
        data=df[user_col].values, 
        dtype=torch.long,
    )
    user_indices = torch.tensor(**kwargs)
    
    kwargs = dict(
        data=df[item_col].values, 
        dtype=torch.long,
    )
    item_indices = torch.tensor(**kwargs)

    user_item_matrix[user_indices, item_indices] = 1

    return user_item_matrix