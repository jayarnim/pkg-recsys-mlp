from ..config.dataloader import *


def dataloader(cfg):
    return {
        task: DataloaderCfg(
            ratio=cfg["dataloader"]["negatives"][task],
            seed=cfg["seed"],
            batch_size=cfg["dataloader"]["batch_size"][task],
            shuffle=cfg["dataloader"]["shuffle"],
        )
        for task in ["opt", "msr"]
    }