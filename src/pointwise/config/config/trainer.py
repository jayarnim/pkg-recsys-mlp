from dataclasses import dataclass
from core.pointwise.config.config.criterion import *
from core.pointwise.config.config.optimizer import *


@dataclass
class TrnCfg:
    optimizer: OptimizerCfg
    criterion: CriterionCfg


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