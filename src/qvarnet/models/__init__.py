from .mlp import MLP
from .exponential import ExponentialMLPwithPenalty
from .base import BaseModel
from .mlp_fermions import FermionicMLP
from .registry import MODEL_REGISTRY, register_model


def get_model(model_name, **kwargs):
    """Retrieve a model class from the MODEL_REGISTRY by name.
    Args:
        model_name: Name of the model to retrieve.
        **kwargs: Additional keyword arguments to pass to the model constructor.
    Returns:
        An instance of the requested model class."""
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY.")
    return model_class(**kwargs)
