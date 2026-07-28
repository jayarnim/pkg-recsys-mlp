from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from ..state import State
from .criterion import Criterion
from .optimizer import Optimizer
from core.pointwise.dataloader import DataLoader


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
        for user_idx, item_idx, label in tqdm(**kwargs):
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                item_idx=item_idx.to(DEVICE),
                label=label.to(DEVICE),
            )

            # forward pass
            with autocast(DEVICE.type):
                batch_score = self.batch_step(**kwargs)

            # backward pass
            self.backprop(batch_score)

            # accumulate loss
            epoch_score += batch_score.item()

        state.trn_score = epoch_score / len(dataloader)

    def batch_step(self, user_idx, item_idx, label):
        logit = self.model.predict(
            user_idx=user_idx, 
            item_idx=item_idx,
        )
        score = self.criterion(
            pred=logit, 
            true=label,
        )
        return score

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
