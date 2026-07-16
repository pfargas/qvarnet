"""TrainedWavefunction — a frozen ψ with everything baked, for measuring properties.

Wraps (model, params, coord_mode, box) into a single object so that
post-training analysis is done in terms of *the wavefunction*, not the model:

    wf = TrainedWavefunction.from_result(model, result, n_particles=N)
    wf.sample(n_chains=1024, n_steps=400, burn_in=150)
    x, n   = wf.density(bins=60)
    r, g   = wf.pair_correlation()
    k, S   = wf.structure_factor()
    grid, rho = wf.obdm(np.linspace(-4, 4, 41))
    n0     = wf.condensate_fraction()

Invariants:
  - ``wf.samples`` are always **lab coordinates** — coord_mode conversion
    (e.g. Jacobi → lab) happens once, right after sampling.
  - ``wf.log_psi(x)`` takes lab coordinates; the model always consumes lab
    coordinates regardless of coord_mode, so this is universal.
  - The MCMC walker runs in sampler space (``coord_mode.wrap_model_apply``);
    repeated ``sample()`` calls warm-start from the last walker positions.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..config.coord_mode import CoordMode, JacobiCoords, LabCoords
from ..samplers import mh_chain, resolve_proposal
from ..samplers.diagnostics import chain_stats
from ..vmc.probability import build_prob_fn
from . import kernels


class TrainedWavefunction:
    """A trained wavefunction ψ(x) with params, coordinates and box baked in.

    Args:
        model: Flax module mapping lab coords ``(..., N*d) -> log|ψ|``.
        params: full Flax variables dict (as returned by ``model.init`` /
            ``TrainResult.best_params()`` / ``load_run().params``).
        n_particles: number of physical particles N.
        n_dim: spatial dimension (currently 1).
        coord_mode: ``LabCoords()`` (default) or ``JacobiCoords(...)``. Must
            match the mode the model was trained with.
        box_L: periodic box side length; 0 means no box. Used both by the
            sampler (PBC folding of proposals) and by the estimators
            (histogram folding, minimum image, commensurate k).
        seed: PRNG seed used when ``sample()`` is called without a key.
    """

    def __init__(
        self,
        model,
        params,
        n_particles: int,
        n_dim: int = 1,
        coord_mode: CoordMode | None = None,
        box_L: float = 0.0,
        seed: int = 0,
    ):
        self.model = model
        self.params = params
        self.n_particles = int(n_particles)
        self.n_dim = int(n_dim)
        self.coord_mode = coord_mode if coord_mode is not None else LabCoords()
        self.box_L = float(box_L)

        if isinstance(self.coord_mode, JacobiCoords):
            if self.coord_mode.n_particles_physical != self.n_particles:
                raise ValueError(
                    f"n_particles={self.n_particles} disagrees with "
                    f"coord_mode.n_particles_physical={self.coord_mode.n_particles_physical}"
                )
            self._sampler_dof = (self.n_particles - 1) * self.n_dim
        else:
            self._sampler_dof = self.n_particles * self.n_dim
        self.lab_dof = self.n_particles * self.n_dim

        # walker-space log|ψ|² for MCMC; lab-space log|ψ| for everything else
        self._prob_fn = build_prob_fn(self.coord_mode.wrap_model_apply(model.apply))
        self._key = jax.random.PRNGKey(seed)

        self._samples = None  # lab coords, (M, lab_dof)
        self._last_positions = None  # sampler space, (n_chains, sampler_dof)
        self._step_size = 0.5
        self._obdm_cache = None  # (grid, rho) of the last obdm() call
        self.acceptance_rate = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_result(cls, model, result, n_particles, params=None, **kwargs):
        """Build from a ``TrainResult``: best snapshot params, warm-started sampler.

        Uses ``result.best_params()`` (override with ``params=``) and seeds the
        walker at ``result.final_positions`` with ``result.final_step_size``,
        so the first ``sample()`` call resumes the trained distribution
        instead of re-thermalising. Extra kwargs go to the constructor
        (``coord_mode``, ``box_L``, ``n_dim``, ``seed``).
        """
        wf = cls(
            model, params if params is not None else result.best_params(), n_particles, **kwargs
        )
        if result.final_positions is not None:
            wf._last_positions = jnp.asarray(result.final_positions)
        if result.final_step_size is not None:
            wf._step_size = float(result.final_step_size)
        return wf

    @classmethod
    def from_checkpoint(
        cls, path, n_particles=None, checkpoint_filename="checkpoint.msgpack", **kwargs
    ):
        """Build from a run directory saved with ``save_run_config`` (uses ``load_run``).

        ``n_particles`` is inferred from a JacobiCoords run; for LabCoords it
        must be given (the flat dof alone doesn't determine N vs d). Extra
        kwargs go to the constructor (``box_L``, ``n_dim``, ``seed``).
        """
        from ..utils.checkpoint import load_run

        run = load_run(path, checkpoint_filename=checkpoint_filename)
        if n_particles is None:
            if isinstance(run.coord_mode, JacobiCoords):
                n_particles = run.coord_mode.n_particles_physical
            else:
                raise ValueError(
                    "n_particles cannot be inferred from a LabCoords checkpoint — pass it explicitly"
                )
        return cls(run.model, run.params, n_particles, coord_mode=run.coord_mode, **kwargs)

    # -- the wavefunction itself --------------------------------------------

    def log_psi(self, x):
        """log|ψ(x)| with everything baked. ``x``: lab coords ``(..., N*d)`` → ``(...,)``."""
        return jnp.squeeze(self.model.apply(self.params, jnp.asarray(x)), axis=-1)

    # -- sampling ------------------------------------------------------------

    def sample(
        self,
        n_chains: int = 1024,
        n_steps: int = 400,
        burn_in: int = 100,
        thinning: int = 1,
        step_size: float | None = None,
        proposal="gaussian",
        key=None,
        reset: bool = False,
    ):
        """Draw samples from |ψ|² via Metropolis-Hastings; caches them in lab coords.

        Warm-starts from the last walker positions when available (from a
        previous ``sample()`` or ``from_result``) unless ``reset=True`` or
        ``n_chains`` changed. Returns the lab-coordinate samples
        ``(n_chains * n_effective, N*d)``, also available as ``wf.samples``.
        """
        if key is None:
            self._key, key = jax.random.split(self._key)
        if step_size is not None:
            self._step_size = float(step_size)

        init = self._last_positions
        if reset or init is None or init.shape[0] != n_chains:
            key, init_key = jax.random.split(key)
            if self.box_L > 0:
                init = jax.random.uniform(
                    init_key, (n_chains, self._sampler_dof), maxval=self.box_L
                )
            else:
                init = jax.random.normal(init_key, (n_chains, self._sampler_dof))

        # Run the raw per-chain trajectories ourselves (rather than going through
        # sample_and_process, which flattens (n_chains, n_steps, dof) -> (M, dof)
        # before we ever see it) so chain_stats has real chain/step structure to
        # diagnose *before* we burn-in/thin/flatten into the final sample batch.
        chain_keys = jax.random.split(key, n_chains)
        raw_batch, acc = jax.vmap(
            lambda k, x0: mh_chain(
                k,
                self._prob_fn,
                self.params,
                x0,
                self._step_size,
                n_steps,
                resolve_proposal(proposal),
                self.box_L,
            )
        )(chain_keys, init)  # raw_batch: (n_chains, n_steps, sampler_dof)

        self._last_positions = raw_batch[:, -1, :]
        self.acceptance_rate = float(jnp.mean(acc))

        cropped = raw_batch[:, n_steps - burn_in :, :]  # keep the last `burn_in` steps
        taus, ess = chain_stats(cropped)  # diagnose on raw sampler-space chains
        self._iat = float(jnp.mean(taus))
        print(f"IAT: {self._iat:.2f}  mean ESS: {float(jnp.mean(ess)):.1f}")

        thin = int(np.ceil(self._iat))
        processed = cropped[:, ::thin, :]
        batch_flat = processed.reshape(-1, self._sampler_dof)

        self._samples = np.asarray(self.coord_mode.samples_to_lab(batch_flat))
        self._obdm_cache = None  # samples changed — cached ρ₁ is stale
        return self._samples

    @property
    def samples(self):
        """Cached samples from the last ``sample()`` call, lab coords ``(M, N*d)``."""
        if self._samples is None:
            raise RuntimeError("No samples yet — call wf.sample(...) first.")
        return self._samples
    
    @property
    @staticmethod
    def samples_diagnostics(samples):
        """Diagnostics of the last ``sample()`` call: dict with keys ``n_chains``, ``n_steps``, ``burn_in``, ``thinning``, ``step_size``, ``acceptance_rate``."""
        iat, ess = chain_stats(samples)
        print(f"IAT: {iat}, ESS: {ess}")
        return {"iat": iat, "ess": ess}

    # -- estimators -----------------------------------------------------------

    def _L(self):
        return self.box_L if self.box_L > 0 else None

    def _resolve_samples(self, samples):
        return self.samples if samples is None else np.asarray(samples)

    def density(self, bins=60, value_range=None, samples=None):
        """Single-particle density n(x), ∫ n dx = N. Returns ``(centers, n)``."""
        return kernels.density_histogram(
            self._resolve_samples(samples),
            self.n_particles,
            self.n_dim,
            bins=bins,
            value_range=value_range,
            L=self._L(),
        )

    def pair_correlation(self, bins=60, value_range=None, samples=None):
        """Pair-distance distribution over all i<j pairs. Returns ``(centers, counts)``."""
        return kernels.pair_correlation(
            self._resolve_samples(samples),
            self.n_particles,
            self.n_dim,
            bins=bins,
            L=self._L(),
            value_range=value_range,
        )

    def pair_correlation_grid(self, grid, samples=None):
        """Full pair correlation g(x, x′) = ρ₂(x, x′)/(ρ(x)ρ(x′)) on ``grid``.

        Unlike ``pair_correlation`` this keeps both coordinates explicit (valid for
        trapped/inhomogeneous systems too). Returns ``(grid, g)``, g symmetric (G, G).
        """
        return kernels.pair_correlation_grid(
            self._resolve_samples(samples),
            grid,
            self.n_particles,
            self.n_dim,
            L=self._L(),
        )

    def structure_factor(self, k_values=None, n_max=20, samples=None):
        """Static structure factor S(k). Returns ``(k, S)``.

        ``k_values=None`` uses the commensurate k = 2πn/L, n = 1..n_max
        (requires ``box_L > 0``).
        """
        if k_values is None:
            if self.box_L <= 0:
                raise ValueError(
                    "k_values=None needs a periodic box (box_L > 0) to pick commensurate k"
                )
            k_values = kernels.commensurate_k(self.box_L, n_max)
        return kernels.structure_factor(
            self._resolve_samples(samples), self.n_particles, k_values, self.n_dim
        )

    def obdm(self, grid, samples=None):
        """One-body density matrix ρ₁(x, x′) on ``grid``. Returns ``(grid, rho)``.

        The result is cached so ``natural_orbitals()`` / ``condensate_fraction()``
        can be called without recomputing.
        """
        grid, rho = kernels.obdm_grid(
            self.log_psi,
            self._resolve_samples(samples),
            grid,
            self.n_particles,
            self.n_dim,
        )
        self._obdm_cache = (grid, rho)
        return grid, rho

    def natural_orbitals(self, grid=None):
        """Natural occupations and orbitals of ρ₁ (computes ``obdm(grid)`` if needed)."""
        grid, rho = self._obdm(grid)
        return kernels.natural_orbitals(rho, grid)

    def condensate_fraction(self, grid=None):
        """n₀ = λ_max / Σλ of ρ₁ (computes ``obdm(grid)`` if needed)."""
        grid, rho = self._obdm(grid)
        return kernels.condensate_fraction(rho, grid)

    def _obdm(self, grid):
        if grid is not None:
            return self.obdm(grid)
        if self._obdm_cache is None:
            raise RuntimeError("No OBDM yet — call wf.obdm(grid) first or pass grid=.")
        return self._obdm_cache

    def obdm_displacement(self, displacements, particle=0, samples=None):
        """Translationally-averaged ρ₁(Δ) for homogeneous systems. Returns ``(Δ, ρ₁)``."""
        return kernels.obdm_displacement(
            self.log_psi,
            self._resolve_samples(samples),
            displacements,
            particle=particle,
            n_dim=self.n_dim,
        )

    # -- misc ------------------------------------------------------------------

    def __repr__(self):
        sampled = self._samples.shape[0] if self._samples is not None else 0
        return (
            f"TrainedWavefunction(N={self.n_particles}, d={self.n_dim}, "
            f"coord_mode={type(self.coord_mode).__name__}, box_L={self.box_L}, "
            f"cached_samples={sampled})"
        )
