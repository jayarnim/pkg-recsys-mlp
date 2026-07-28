from dataclasses import dataclass


@dataclass
class DataloaderCfg:
    ratio: int
    batch_size: int
    shuffle: bool
    seed: int