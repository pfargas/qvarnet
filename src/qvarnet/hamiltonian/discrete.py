"""Discrete (lattice/spin) Hamiltonians for the mini-VMC testbed (roadmap step 3).

Deliberately modest: no operator algebra, no symmetry machinery — just enough to run the
*entire* engine against a known-E₀ system (TFIM via exact diagonalisation, ``utils/exact_diag``).

Local energy uses the connected-elements form (no Laplacian, no AD second derivatives):

    E_loc(s) = Σ_{s'} H_{ss'} ψ(s')/ψ(s)
             = Σ_{s' connected} H_{ss'} · exp(log|ψ(s')| - log|ψ(s)|).

For the transverse-field Ising chain  H = -J Σ σ^z_i σ^z_{i+1} - h Σ σ^x_i  the diagonal is
classical and σ^x_i connects s to the single-flip state F_i s, so

    E_loc(s) = -J Σ_i s_i s_{i+1}  -  h Σ_i exp(log|ψ(F_i s)| - log|ψ(s)|).
"""

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class TFIMHamiltonian:
    """Transverse-field Ising chain. Spins s ∈ {±1}^N feed the model as a flat vector."""

    n_spins: int = struct.field(pytree_node=False, default=2)
    pbc: bool = struct.field(pytree_node=False, default=True)
    J: float = 1.0
    h: float = 1.0

    def diagonal_energy(self, configs):
        """-J Σ_i s_i s_{i+1} for a batch of configs, shape (B, N) → (B,)."""
        if self.pbc:
            bonds = configs * jnp.roll(configs, -1, axis=-1)
        else:
            bonds = configs[..., :-1] * configs[..., 1:]
        return -self.J * jnp.sum(bonds, axis=-1)

    def local_energy(self, params, samples, model_apply, key=None):
        """Connected-elements local energy, shape (B, N) → (B,).

        Same signature as ``ContinuousHamiltonian.local_energy`` so it drops into the
        existing ``compute_step`` unchanged — the discrete path reuses the whole VMC
        gradient / SR / minSR / MetricsHistory machinery (the §7 factorization).
        ``key`` is accepted for interface compatibility and ignored (no stochastic
        Laplacian here).
        """

        def logpsi_fn(configs):
            return model_apply(params, configs).squeeze(-1)

        return self.local_energy_logpsi(logpsi_fn, samples)

    def local_energy_logpsi(self, logpsi_fn, configs):
        """Connected-elements local energy from a log-amplitude callable.

        ``logpsi_fn(configs) -> (B,)`` returns log|ψ|. Used directly by the full-sum path
        (``vmc/full_sum``) and via ``local_energy`` by the sampled / training path. The N
        single-site flips are evaluated in one batched call.
        """
        B, n = configs.shape
        logpsi = logpsi_fn(configs)  # (B,)
        # flipped[b, i] = configs[b] with site i negated → multiply column i by -1.
        flip_op = 1.0 - 2.0 * jnp.eye(n)  # (N, N)
        flipped = configs[:, None, :] * flip_op[None, :, :]  # (B, N, N)
        logpsi_flipped = logpsi_fn(flipped.reshape(B * n, n)).reshape(B, n)  # (B, N)
        ratios = jnp.exp(logpsi_flipped - logpsi[:, None])  # ψ(F_i s)/ψ(s), (B, N)
        return self.diagonal_energy(configs) - self.h * jnp.sum(ratios, axis=-1)
