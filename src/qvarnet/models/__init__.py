from .base import BaseModel as BaseModel
from .compose import LogWavefunction as LogWavefunction
from .deep_set import DeepSet as DeepSet
from .deep_set import DeepSetNoEnvelope as DeepSetNoEnvelope
from .envelopes import GaussianEnvelope as GaussianEnvelope
from .envelopes import PolynomialEnvelope as PolynomialEnvelope
from .exponential import ExponentialMLPwithPenalty as ExponentialMLPwithPenalty
from .exponential import LogAnalyticWavefunction as LogAnalyticWavefunction
from .exponential import (
    LogExponentialMLPwithGaussianPenalty as LogExponentialMLPwithGaussianPenalty,
)
from .exponential import LogExponentialMLPwithPenalty as LogExponentialMLPwithPenalty
from .jastrow import LogJastrow as LogJastrow
from .mlp import MLP as MLP
from .mlp_fermions import FermionicMLP as FermionicMLP
from .registry import MODEL_REGISTRY as MODEL_REGISTRY
from .registry import register_model as register_model


def get_model(model_name, **kwargs):
    """Retrieve a model class from the MODEL_REGISTRY by name."""
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY.")
    return model_class(**kwargs)
