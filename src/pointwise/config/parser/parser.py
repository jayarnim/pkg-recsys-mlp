from ..config.config import *
from .model import model
from .datamodule import datamodule
from .trainer import trainer


def parser(cfg):
    return Config(
        model=model(cfg),
        datamodule=datamodule(cfg),
        trainer=trainer(cfg),
        seed=cfg["seed"],
    )