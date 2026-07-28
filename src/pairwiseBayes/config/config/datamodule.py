from typing import Literal
from dataclasses import dataclass
from core.pairwise.config.config.dataloader import *


@dataclass
class SplitCfg:
    ratio: dict[str, int]
    min_rating: int
    filter_by: Literal["user", "item"]
    seed: int


@dataclass
class DataModuleCfg:
    split: SplitCfg
    dataloader: DataloaderCfg