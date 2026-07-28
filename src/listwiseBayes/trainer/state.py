from dataclasses import dataclass


@dataclass
class State:
    num_epochs: int

    current_epoch: int = 0
    trn_elbo: float | None = None
    trn_nll: float | None = None
    trn_kld: float | None = None
    val_score: float | None = None

    is_best: bool = False
    best_epoch: int = 0
    best_score: float = float("-inf")

    counter: int = 0
    should_stop: bool = False
