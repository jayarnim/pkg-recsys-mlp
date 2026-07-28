from dataclasses import dataclass
import torch


@dataclass
class BayesModelOutput:
    logit: torch.Tensor
    kld: torch.Tensor

    def view(self, *shape):
        return BayesModelOutput(
            logit=self.logit.view(*shape),
            kld=self.kld.view(*shape),
        )

    def view_as(self, other):
        return self.view(*other.shape)