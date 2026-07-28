from .dataset import PointwiseDataset
from .collate import collate_fn
from .sampler import NegativeSampler
from torch.utils.data import DataLoader
import pandas as pd
from ..config.config.dataloader import DataloaderCfg
from msr.constants import *


def build(
    df: pd.DataFrame, 
    unobs: dict[int, list],
    cfg: DataloaderCfg,
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
) -> DataLoader:
    sampler = NegativeSampler(
        unobs=unobs,
        ratio=cfg.ratio,
        seed=cfg.seed,
    )

    dataset = PointwiseDataset(
        df=df,
        sampler=sampler,
        user_col=user_col,
        item_col=item_col,
    )

    kwargs = dict(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
    )
    return DataLoader(**kwargs)