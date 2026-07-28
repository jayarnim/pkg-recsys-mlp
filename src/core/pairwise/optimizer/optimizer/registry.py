import torch.optim as optim

OPTIMIZER_REGISTRY = {}

def register(name):
    def wrapper(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return wrapper

register("adam")(optim.Adam)
register("adamw")(optim.AdamW)
register("adagrad")(optim.Adagrad)
register("sgd")(optim.SGD)