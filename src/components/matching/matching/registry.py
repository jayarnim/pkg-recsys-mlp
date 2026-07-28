MATCHING_REGISTRY = {}

def register(name):
    def wrapper(cls):
        MATCHING_REGISTRY[name] = cls
        return cls
    return wrapper