from functools import partial
import jax
import jax.numpy as jnp
from .laplacian import laplacian_autodiff_new as laplacian


def kinetic_term(params, xs, model_apply, laplacian=laplacian):
    def psi_fn(x):
        # ensure input has shape (1,) as model expects last-dim features
        x = jnp.atleast_1d(x).reshape(1, -1)  # (1, DoF)
        return model_apply(params, x).squeeze()

    d2psi = laplacian(params, xs, model_apply)

    psi_vals = jax.vmap(lambda x: psi_fn(x))(xs)  # shape (batch,)

    psi_safe = psi_vals + 1e-12

    return -0.5 * (d2psi / psi_safe)  # shape (batch,)
