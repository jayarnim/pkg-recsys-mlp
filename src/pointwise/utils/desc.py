import pandas as pd
from msr.constants import *


def main(
    df: pd.DataFrame, 
    percentile: float=0.9,
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
) -> None:
    user_counts = df[user_col].value_counts()
    item_counts = df[item_col].value_counts()

    N_USERS = df[user_col].nunique()
    N_ITEMS = df[item_col].nunique()
    TOTAL_INTERACTION = len(df)
    DENSITY = df.shape[0] / (N_USERS * N_ITEMS)
    MAX_USER_INTERACTION = user_counts.max()
    MAX_ITEM_INTERACTION = item_counts.max()
    TOP_PERCENTAILE_USER_INTERACTION = user_counts.quantile(percentile)
    TOP_PERCENTAILE_ITEM_INTERACTION = item_counts.quantile(percentile)

    print(
        f"number of user: {N_USERS}",
        f"number of item: {N_ITEMS}",
        f"total interaction: {TOTAL_INTERACTION}",
        f"interaction density: {DENSITY * 100:.4f} %",
        f"max interaction of user: {MAX_USER_INTERACTION}",
        f"max interaction of item: {MAX_ITEM_INTERACTION}",
        f"top {(1-percentile) * 100:.1f} % interaction of user: {TOP_PERCENTAILE_USER_INTERACTION:.1f}",
        f"top {(1-percentile) * 100:.1f} % interaction of item: {TOP_PERCENTAILE_ITEM_INTERACTION:.1f}",
        f"mean interaction of user: {TOTAL_INTERACTION // N_USERS}",
        f"mean interaction of item: {TOTAL_INTERACTION // N_ITEMS}",
        sep="\n",
    )
