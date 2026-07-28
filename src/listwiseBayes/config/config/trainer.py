from dataclasses import dataclass
from core.listwise.config.config.criterion import *
from core.listwise.config.config.optimizer import *
from core.listwise.config.config.annealing import *
from core.listwise.config.config.elbo import *


@dataclass
class TrnCfg:
    optimizer: OptimizerCfg
    elbo: ELBOCfg


@dataclass
class MetricCfg:
    name: str
    cutoff: int


@dataclass
class ValCfg:
    metric: MetricCfg


@dataclass
class EarlyStoppingCfg:
    patience: int
    delta: float
    warmup: int


@dataclass
class TrainerCfg:
    num_epochs: int
    early_stopping: EarlyStoppingCfg
    trn: TrnCfg
    val: ValCfg