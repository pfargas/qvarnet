from jax import numpy as jnp

def nan_callback(x):
    if jnp.isnan(x).any():
        return True
    return False
