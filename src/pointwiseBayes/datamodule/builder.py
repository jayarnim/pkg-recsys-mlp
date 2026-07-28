from .unobs import unobserved_items
from .split import stratified_split
from .dataloader import build as build_dataloader
from .dataloader import DataLoader
from ..config.config.datamodule import DataModuleCfg
from msr.constants import *
import pandas as pd


def build(
    df: pd.DataFrame, 
    cfg: DataModuleCfg, 
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
) -> dict[str, DataLoader]:
    split = stratified_split(
        df=df,
        user_col=user_col,
        item_col=item_col,
        **vars(cfg.split),
    )

    unobs = unobserved_items(
        df=df,
        user_col=user_col,
        item_col=item_col,
    )

    MAPPING = {
        "trn": "opt",
        "val": "msr",
        "tst": "msr",
    }

    return {
        k: build_dataloader(
            df=v,
            unobs=unobs,
            cfg=cfg.dataloader[MAPPING[k]],
            user_col=user_col,
            item_col=item_col,
        )
        for k, v in split.items()
    }, split["trn"]