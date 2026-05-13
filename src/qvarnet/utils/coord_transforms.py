import jax.numpy as jnp

from ..config.coord_mode import CoordMode, LabCoords, JacobiCoords
from .jacobi import from_jacobi_to_lab


def build_effective_apply(model_apply, coord_mode: CoordMode):
    """Wrap model_apply with a coordinate pre-processing step.

    LabCoords:    identity — no wrapping.
    JacobiCoords: sampler produces N Jacobi relative coords; the wrapper pads
                  with a zero CM coord, applies the inverse Jacobi transform,
                  and passes the reconstructed N+1 lab coords to the model.
    """
    if isinstance(coord_mode, LabCoords):
        return model_apply

    if isinstance(coord_mode, JacobiCoords):
        n_phys = coord_mode.n_particles_physical
        n_d = coord_mode.n_dim

        def apply(params, x):
            # x: (..., N)  Jacobi relative coords (N = n_phys - 1)
            zeros = jnp.zeros((*x.shape[:-1], 1))
            u_tilde = jnp.concatenate([x, zeros], axis=-1)          # (..., N+1)
            x_lab = from_jacobi_to_lab(u_tilde, n_phys, n_d)        # (..., N+1)
            return model_apply(params, x_lab)

        return apply

    raise TypeError(f"Unknown CoordMode: {type(coord_mode)}")


def init_shape_for_model(sample_shape, coord_mode: CoordMode):
    """Return the shape used to initialise model parameters.

    For JacobiCoords the sampler shape has N cols but the model expects N+1.
    """
    if isinstance(coord_mode, JacobiCoords):
        return (*sample_shape[:-1], coord_mode.n_particles_physical * coord_mode.n_dim)
    return sample_shape
