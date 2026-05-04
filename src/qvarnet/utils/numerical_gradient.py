from functools import partial
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree


@partial(jax.jit, static_argnames=["model_apply", "epsilon"])
def numerical_parameter_gradients(
    hamiltonian, energy, params, batch, model_apply, epsilon=1e-6
):
    """Compute parameter gradients via central finite differences.

    Useful for validating automatic differentiation.  All perturbations are
    evaluated in parallel with :func:`jax.vmap`.

    .. math::

        \\frac{\\partial E}{\\partial \\theta_i} \\approx
        \\frac{E(\\theta + \\epsilon e_i) - E(\\theta - \\epsilon e_i)}{2\\epsilon}

    Args:
        hamiltonian: Hamiltonian operator passed through to ``energy``.
        energy: Callable ``(hamiltonian, params, batch, model_apply) -> (E, E_loc, sigma)``.
        params: Current model parameters (PyTree).
        batch: Training batch of configurations, shape ``(batch_size, DoF)``.
        model_apply: Model forward pass function.
        epsilon: Finite-difference step size.

    Returns:
        Numerical gradient PyTree with the same structure as ``params``.
    """
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
