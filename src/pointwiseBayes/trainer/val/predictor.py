from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from core.pointwise.dataloader import DataLoader
from msr.constants import *
from ..state import State


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(object):
    def __init__(
        self, 
        model: nn.Module,
        user_col: str=DEFAULT_USER_COL,
        item_col: str=DEFAULT_ITEM_COL,
        rating_col: str=DEFAULT_RATING_COL,
        prediction_col: str=DEFAULT_PREDICTION_COL,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.prediction_col = prediction_col

    @torch.no_grad()
    def __call__(
        self,
        dataloader: DataLoader,
        state: State,
    ) -> pd.DataFrame:
        # evaluation
        self.model.eval()

        # to save result
        user_idx_list = []
        item_idx_list = []
        rating_list = []
        prediction_list = []

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {state.current_epoch}/{state.num_epochs} VAL"
        )

        # start batch loop
        for user_idx, item_idx, label in tqdm(**kwargs):
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                item_idx=item_idx.to(DEVICE),
            )

            # predict
            output = self.model.predict(**kwargs)

            # to cpu & save
            user_idx_list.extend(user_idx.cpu().tolist())
            item_idx_list.extend(item_idx.cpu().tolist())
            rating_list.extend(label.cpu().tolist())
            prediction_list.extend(output.logit.cpu().tolist())

        # list -> df
        return pd.DataFrame(
            {
                self.user_col: user_idx_list,
                self.item_col: item_idx_list,
                self.rating_col: rating_list,
                self.prediction_col: prediction_list,
            }
        )