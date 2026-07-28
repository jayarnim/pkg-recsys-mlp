import pandas as pd
from sklearn.preprocessing import LabelEncoder
from msr.constants import *


def main(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
) -> pd.DataFrame:
    ORIGIN_COLS = [user_col, item_col]
    RENAME_COLS = [DEFAULT_USER_COL, DEFAULT_ITEM_COL]

    RENAMES = dict(zip(ORIGIN_COLS, RENAME_COLS))
    
    df = df[ORIGIN_COLS]
    df = df.rename(columns=RENAMES)

    for col in RENAME_COLS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col]) + 2

    return df