COLLATE_REGISTRY = {}

def register(name):
    def wrapper(cls):
        COLLATE_REGISTRY[name] = cls
        return cls
    return wrapper