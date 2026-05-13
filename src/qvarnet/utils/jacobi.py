import jax
import jax.numpy as jnp
from functools import partial

def jacobi_transformation(n_particles: int):
    """
    Perform a Jacobi transformation on the input coordinates.
    Args:
        x: A 1D array of shape (*, n_particles * dim) containing the coordinates of the particles,
        n_particles: The number of particles.
        dim: The dimensionality of the space.
    """
    # Reshape x to (n_particles, dim)
    # x = x.reshape(*x.shape[:-1], n_particles, dim)

    C = jnp.zeros((n_particles, n_particles))
    
    for row in range(n_particles-1):
        norm = jnp.sqrt((row+2)/(row+1))
        for col in range(n_particles):
            if col > row+1:
                C = C.at[row, col].set(0)
            elif col == row+1:
                C = C.at[row, col].set(-1/norm)
            else:
                C = C.at[row, col].set(1/((row+1)*norm))
    C = C.at[n_particles-1, :].set(1/jnp.sqrt(n_particles))
    # assert jnp.allclose(C @ C.T, jnp.eye(n_particles), atol=1e-6), "Jacobi transformation matrix is not orthogonal"
    return C

@partial(jax.jit, static_argnames=["n_particles", "dim", "C"])
def apply_transformation(x, C, n_particles, dim):
    # x has shape (..., n_particles * dim)
    # Reshape x to (..., n_particles, dim)
    x = x.reshape(*x.shape[:-1], n_particles, dim)
    # Apply the transformation
    transformed = jnp.einsum('ij,...jk->...ik', C, x)
    # Reshape back to (..., n_particles * dim)
    return transformed.reshape(*x.shape[:-2], -1)

@partial(jax.jit, static_argnames=["n_particles", "dim"])
def from_lab_to_jacobi(x, n_particles, dim):
    C = jacobi_transformation(n_particles)
    return apply_transformation(x, C, n_particles, dim)

@partial(jax.jit, static_argnames=["n_particles", "dim"])
def from_jacobi_to_lab(x, n_particles, dim):
    C = jacobi_transformation(n_particles)
    C_inv = C.T  # C is orthogonal, so its inverse is its transpose
    return apply_transformation(x, C_inv, n_particles, dim)

