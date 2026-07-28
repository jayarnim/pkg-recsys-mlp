from ..config.datamodule import *
from core.listwise.config.parser.dataloader import *


def split(cfg):
    return SplitCfg(
        ratio=cfg["split"]["ratio"],
        min_rating=cfg["split"]["min_rating"],
        filter_by=cfg["split"]["filter_by"],
        seed=cfg["seed"],
    )


def datamodule(cfg):
    return DataModuleCfg(
        split=split(cfg),
        dataloader=dataloader(cfg),
    )