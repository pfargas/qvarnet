HAMILTONIAN_REGISTER = {}


def register_hamiltonian(name):
    """Decorator to register a model class in the MODEL_REGISTRY.
    Args:
        name: Name of the model to register.
    Returns:
        A decorator that registers the model class."""

    def decorator(cls):
        HAMILTONIAN_REGISTER[name] = cls
        return cls

    return decorator
