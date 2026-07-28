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
        pos: BayesModelOutput, 
        neg: BayesModelOutput, 
        step: int,
    ) -> dict[str, torch.Tensor]:
        BATCH_SIZE, NEG_SAMPLES = neg.kld.shape
        nll = self.criterion(pos.logit, neg.logit)
        kld = (
            ( pos.kld + neg.kld.sum(dim=1) )
            / ( 1 + NEG_SAMPLES )
        ).mean()
        beta = self.annealing(step)
        elbo = nll + beta * kld

        return dict(
            elbo=elbo,
            nll=nll,
            kld=kld,
        )