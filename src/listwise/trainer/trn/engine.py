from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from ..state import State
from .criterion import Criterion
from .optimizer import Optimizer
from core.listwise.dataloader import DataLoader


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: Optimizer, 
        criterion: Criterion,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler(device=DEVICE)

    def __call__(
        self, 
        dataloader: DataLoader, 
        state: State,
    ) -> None:
        # train
        self.model.train()

        # reset epoch loss
        epoch_score = 0.0

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {state.current_epoch}/{state.num_epochs} TRN"
        )

        # start batch loop
        for user_idx, pos_idx, neg_idx in tqdm(**kwargs):
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                pos_idx=pos_idx.to(DEVICE),
                neg_idx=neg_idx.to(DEVICE),
            )

            # forward pass
            with autocast(DEVICE.type):
                batch_score = self.batch_step(**kwargs)

            # backward pass
            self.backprop(batch_score)

            # accumulate loss
            epoch_score += batch_score.item()

        state.trn_score = epoch_score / len(dataloader)

    def batch_step(self, user_idx, pos_idx, neg_idx):
        pos_logit = self.model.predict(
            user_idx=user_idx, 
            item_idx=pos_idx,
        )
        neg_logit = self.model.predict(
            user_idx=user_idx.unsqueeze(1).expand_as(neg_idx).reshape(-1),
            item_idx=neg_idx.reshape(-1),
        )
        score = self.criterion(
            pos=pos_logit, 
            neg=neg_logit.view_as(neg_idx),
        )
        return score

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
