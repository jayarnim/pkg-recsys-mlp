from .annealing import Annealing
from .criterion import Criterion
from components.base import BayesModelOutput
import torch


class ELBO(object):
    def __init__(
        self, 
        criterion: Criterion, 
        annealing: Annealing,
    ):
        super().__init__()
        self.criterion = criterion
        self.annealing = annealing

    def __call__(
        self, 
        pred: BayesModelOutput, 
        true: torch.Tensor, 
        step: int,
    ) -> dict[str, torch.Tensor]:
        nll = self.criterion(pred.logit, true)
        kld = pred.kld.mean()
        beta = self.annealing(step)
        elbo = nll + beta * kld

        return dict(
            elbo=elbo,
            nll=nll,
            kld=kld,
        )