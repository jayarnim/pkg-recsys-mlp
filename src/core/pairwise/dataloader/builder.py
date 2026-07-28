from . import dataset
from . import collate
from .dataset.registry import DATASET_REGISTRY
from .collate.registry import COLLATE_REGISTRY
from typing import Literal
import pandas as pd
from torch.utils.data import DataLoader
from ..config.config.dataloader import DataloaderCfg
from .sampler import NegativeSampler
from msr.constants import *


def build(
    df: pd.DataFrame, 
    unobs: dict[int, list],
    task: Literal["opt", "msr"],
    cfg: DataloaderCfg,
    user_col: str=DEFAULT_USER_COL,
    item_col: str=DEFAULT_ITEM_COL,
) -> DataLoader:

    sampler = NegativeSampler(
        unobs=unobs,
        ratio=cfg.ratio,
        seed=cfg.seed,
    )
    
    dataset = DATASET_REGISTRY[task](
        df=df,
        sampler=sampler,
        user_col=user_col,
        item_col=item_col,
    )

    collate_fn = COLLATE_REGISTRY[task]

    kwargs = dict(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
    )
    return DataLoader(**kwargs)