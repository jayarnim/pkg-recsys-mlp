import pandas as pd
from torch.utils.data import Dataset
from ..sampler.opt import OPTNegativeSampler
from msr.constants import *
from .registry import register


@register("opt")
class OPTDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        sampler: OPTNegativeSampler,
        user_col: str=DEFAULT_USER_COL,
        item_col: str=DEFAULT_ITEM_COL,
    ):
        super().__init__()

        self.pairs = list(
            zip(
                df[user_col],
                df[item_col],
            )
        )

        self.sampler = sampler
        self.ratio = sampler.ratio

    def __len__(self) -> int:
        return len(self.pairs) * self.ratio

    def __getitem__(
        self, 
        idx: int,
    ) -> tuple[int, int, int]:
        user, pos = self.pairs[idx // self.ratio]
        neg = self.sampler(user)
        return user, pos, neg