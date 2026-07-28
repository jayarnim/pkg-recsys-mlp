import pandas as pd
from msr.constants import *


def unobserved_items(
    df: pd.DataFrame, 
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
) -> dict[int, list]:
    user_list = sorted(df[user_col].unique())
    item_list = sorted(df[item_col].unique())

    obs = {
        user: set(df.loc[df[user_col]==user, item_col].tolist())
        for user in user_list
    }

    unobs = {
        user: list(set(item_list) - obs[user])
        for user in user_list
    }

    return unobs