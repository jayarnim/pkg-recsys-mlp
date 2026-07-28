EMBEDDING_REGISTRY = {}

def register(name):
    def wrapper(cls):
        EMBEDDING_REGISTRY[name] = cls
        return cls
    return wrapper