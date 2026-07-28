from ..config.model import *


def model(cfg):
    if cfg.get("ensemble"):
        return {
            k: ModelCfg(
                name=v["name"],
                params=v.get("params") or dict(),
            )
            for k, v in cfg["ensemble"].items()
        }
    else:
        return ModelCfg(
            name=cfg["model"]["name"],
            params=cfg["model"].get("params") or dict(),
        )