from functools import partial
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree


@partial(jax.jit, static_argnames=["model_apply", "epsilon"])
def numerical_parameter_gradients(
    hamiltonian, energy, params, batch, model_apply, epsilon=1e-6
):
    flat_params, unravel_fn = ravel_pytree(params)

    # Create an identity matrix of perturbations
    eye = jnp.eye(flat_params.size) * epsilon

    def get_energy(p_flat):
        E, _, _ = energy(hamiltonian, unravel_fn(p_flat), batch, model_apply)
        return E

    # Vmap over the rows of the identity matrix to get all E_plus and E_minus at once
    E_plus = jax.vmap(lambda p: get_energy(flat_params + p))(eye)
    E_minus = jax.vmap(lambda p: get_energy(flat_params - p))(eye)

    grad_flat = (E_plus - E_minus) / (2 * epsilon)
    return unravel_fn(grad_flat)
