from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class CoordMode:
    """Base class for coordinate systems used in VMC.

    Subclasses must implement three methods that together define how the
    sampler space relates to the model input space:

      model_input_shape  — shape used to initialise model parameters
      wrap_model_apply   — wraps model_apply to accept sampler-space inputs
      samples_to_lab     — transforms sampler-space samples to lab coords
                           (used by the Hamiltonian potential)

    Adding a new coordinate system = add one subclass with these three methods.
    """

    def model_input_shape(self, sample_shape: tuple) -> tuple:
        """Return the shape used to initialise model parameters.

        May differ from the sampler shape (e.g. Jacobi coords have N columns
        but the model expects N+1 lab coords).
        """
        raise NotImplementedError

    def wrap_model_apply(self, model_apply):
        """Wrap model_apply so it accepts sampler-space inputs.

        LabCoords: identity — no wrapping needed.
        JacobiCoords: pads with zero CM coord and applies inverse Jacobi transform
                      before passing to the model.
        """
        raise NotImplementedError

    def samples_to_lab(self, samples):
        """Transform sampler-space samples to lab coordinates.

        Used by ContinuousHamiltonian.local_energy to pass lab coords to
        potential_energy regardless of which sampler space is active.

        samples: (..., sampler_dof)
        returns: (..., lab_dof)
        """
        raise NotImplementedError


@dataclass(frozen=True)
class LabCoords(CoordMode):
    """Default: sampler and model both operate in Cartesian lab coordinates."""

    def model_input_shape(self, sample_shape: tuple) -> tuple:
        return sample_shape

    def wrap_model_apply(self, model_apply):
        return model_apply  # identity

    def samples_to_lab(self, samples):
        return samples  # identity


@dataclass(frozen=True)
class JacobiCoords(CoordMode):
    """Sampler works in N Jacobi relative coords; model receives N+1 lab coords.

    The centre-of-mass coordinate is removed from the sampler (set to zero,
    i.e. CM is pinned at the origin). The model sees all N+1 lab coordinates
    reconstructed via the inverse Jacobi transform.

    n_particles_physical: total number of physical particles (N + 1)
    n_dim: spatial dimension (currently 1 is supported)
    """

    n_particles_physical: int
    n_dim: int = 1

    def model_input_shape(self, sample_shape: tuple) -> tuple:
        # Sampler has N = n_particles_physical - 1 Jacobi coords,
        # but the model expects N+1 lab coords.
        return (*sample_shape[:-1], self.n_particles_physical * self.n_dim)

    def wrap_model_apply(self, model_apply):
        from .jacobi import from_jacobi_to_lab
        n_phys = self.n_particles_physical
        n_d = self.n_dim

        def apply(params, x):
            # x: (..., N)  Jacobi relative coords
            zeros = jnp.zeros((*x.shape[:-1], 1))
            u_tilde = jnp.concatenate([x, zeros], axis=-1)   # (..., N+1)
            x_lab = from_jacobi_to_lab(u_tilde, n_phys, n_d) # (..., N+1)
            return model_apply(params, x_lab)

        return apply

    def samples_to_lab(self, samples):
        from .jacobi import from_jacobi_to_lab
        zeros = jnp.zeros((*samples.shape[:-1], 1))
        u_tilde = jnp.concatenate([samples, zeros], axis=-1)
        return from_jacobi_to_lab(u_tilde, self.n_particles_physical, self.n_dim)
