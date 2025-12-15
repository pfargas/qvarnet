import jax
import jax.numpy as jnp


@jax.jit
def nan_callback(x):
    return jnp.any(jnp.isnan(x))


def update_best_params(energy, best_energy, params, best_params):
    """Update the best energy and parameters if the current energy is lower."""

    def true_fn():
        return energy, params

    def false_fn():
        return best_energy, best_params

    return jax.lax.cond(energy < best_energy, true_fn, false_fn)
