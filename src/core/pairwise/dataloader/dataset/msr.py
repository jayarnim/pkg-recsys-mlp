import pandas as pd
from torch.utils.data import Dataset
from ..sampler import NegativeSampler
from msr.constants import *
from .registry import register


@register("msr")
class MSRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame, 
        sampler: NegativeSampler,
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
        return (
            len(self.pairs) 
            * (1 + self.ratio)
        )

    def __getitem__(
        self, 
        idx: int,
    ) -> tuple[int, int, int]:
        PAIR_IDX = idx // (1 + self.ratio)
        DECISION = (idx % (1 + self.ratio) == 0)

        user, pos = self.pairs[PAIR_IDX]

        if DECISION:
            return user, pos, 1
        else:
            neg = self.sampler(user)
            return user, neg, 0