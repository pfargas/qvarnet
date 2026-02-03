import jax
from jax import numpy as jnp


# @jax.jit
def harmonic_potential(x):
    """Harmonic oscillator potential."""
    return 0.5 * jnp.sum(x**2, axis=1)  # sum over dimensions
