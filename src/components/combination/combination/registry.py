COMB_REGISTRY = {}

def register(name):
    def wrapper(cls):
        COMB_REGISTRY[name] = cls
        return cls
    return wrapper