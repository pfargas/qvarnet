MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a model class in the MODEL_REGISTRY.
    Args:
        name: Name of the model to register.
    Returns:
        A decorator that registers the model class."""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator
