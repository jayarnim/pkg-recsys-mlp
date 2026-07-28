from dataclasses import dataclass, field
from typing import Any
from .state import State


@dataclass
class Records:
    trn_nlls: list[float] = field(default_factory=list)
    trn_klds: list[float] = field(default_factory=list)
    val_scores: list[float] = field(default_factory=list)
    best_epoch: int = 0

    def update(
        self, 
        state: State,
    ) -> None:
        self.trn_nlls.append(state.trn_nll)
        self.trn_klds.append(state.trn_kld)
        self.val_scores.append(state.val_score)
        self.best_epoch = state.best_epoch

    def get(self) -> dict[str, Any]:
        trn = dict(
            nll=self.trn_nlls,
            kld=self.trn_klds,
        )
        return dict(
            trn=trn,
            val=self.val_scores,
            best_epoch=self.best_epoch,
        )