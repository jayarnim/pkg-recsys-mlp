from typing import Literal
import pandas as pd
from sklearn.model_selection import train_test_split
from msr.python_splitters import python_stratified_split
from msr.constants import *


def stratified_split(
    df: pd.DataFrame,
    ratio: dict[str, float], 
    seed: int,
    min_rating: int=1,
    filter_by: Literal["user", "item"]="user",
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    ratio_type  = list(ratio.keys())
    ratio_vals = list(ratio.values())

    # trn_val_tst -> [trn, val, tst]
    kwargs = dict(
        data=df,
        ratio=ratio_vals,
        seed=seed,
        min_rating=min_rating,
        filter_by=filter_by,
        col_user=user_col,
        col_item=item_col,
    )
    split = python_stratified_split(**kwargs)

    return dict(zip(ratio_type, split))
