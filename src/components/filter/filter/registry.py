FILTER_REGISTRY = {}

def register(name):
    def wrapper(cls):
        FILTER_REGISTRY[name] = cls
        return cls
    return wrapper