"""The atomic experiment unit: one VMC training run of the dilute soft-sphere gas.

``run_point(potential, x, N, seed, hp)`` trains a neural wavefunction for ``N`` bosons at
gas parameter ``x`` for a given soft-sphere ``potential`` and returns a :class:`PointResult`
in **paper units** (see ``CONVENTIONS.md``). Everything else in this study — the x-sweep,
the finite-N extrapolation, the seed averaging, the shape sweep — just orchestrates calls to
this function.

Split of responsibilities (see ``CONVENTIONS.md`` §7):

* ``potential``, ``x``, ``N``  — the **physics** (define the true answer);
* ``seed``                     — RNG;
* ``hp``                       — the **solver** (optimizer / sampler / model / epochs), the
                                 only thing the hyperparameter tuner varies.

The single factor-of-two engine bridge (paper ``ℏ²/2m=1`` ↔ engine ``ℏ=m=1``) lives in
``dilute_gas.engine_V0`` / ``dilute_gas.to_paper_energy`` and is applied here, nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np
import optax
from dilute_gas import (
    box_side_for_gas_parameter,
    engine_V0,
    first_order_energy_upper_bound,
    soft_sphere_V0_for_scattering_length,
    to_paper_energy,
)

from qvarnet import PenetrableSphereHamiltonian, PeriodicBoundary
from qvarnet.callbacks import EarlyStopCallback
from qvarnet.config.coord_mode import LabCoords
from qvarnet.config.training_setup import SamplingConfig, TrainingConfig
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.deep_set import DeepSet
from qvarnet.train import train

N_DIM = 3  # this study is 3D only (the dilute_gas parametrization assumes it)

assert (
    N_DIM == 3
), "soft_sphere_gas/point.py is 3D only (the dilute_gas parametrization assumes it)"

# ── what we solve: the potential (one free number, R; V0 inferred at a = 1) ────────


@dataclass(frozen=True)
class Potential:
    """A soft (penetrable) sphere at scattering length ``a = 1`` (paper units).

    Only ``R`` is free: ``V0_paper`` is fixed by requiring ``a = 1`` (Eq. 10). Build with
    :meth:`from_R`. ``R = 1`` is the hard-sphere limit (V0 → ∞) and is *not* representable
    here; use ``R > 1``.
    """

    R: float
    V0_paper: float
    label: str

    @classmethod
    def from_R(cls, R: float, label: str | None = None) -> Potential:
        if R <= 1.0:
            raise ValueError(
                f"need R > 1 (a = 1; R = 1 is the hard sphere, V0 → ∞); got R={R}"
            )
        V0_paper = soft_sphere_V0_for_scattering_length(1.0, R)
        return cls(R=R, V0_paper=V0_paper, label=label or f"SS{R:g}")


SS10 = Potential.from_R(10.0, "SS10")
SS5 = Potential.from_R(5.0, "SS5")


# ── how we solve it: hyperparameters (the only thing the tuner varies) ─────────────


@dataclass(frozen=True)
class HP:
    """Solver knobs. Frozen + serializable, so ``(potential, x, N, seed, hp)`` is a DB key."""

    # HP means HyperParameters
    # model (DeepSet inside a LogWavefunction)
    phi_hidden: tuple[int, ...] = (64,)
    F_hidden: tuple[int, ...] = (64,)
    # optimizer
    lr: float = 3e-3
    n_epochs: int = 400
    # sampler
    n_chains: int = 512
    sampler: str = "mh"
    step_size: float = 0.0  # 0 = box-aware auto (~½ interparticle spacing); >0 overrides
    # With warm_walkers the chain persists across epochs, so we only need to *decorrelate* one
    # fresh sample per epoch: a 21-step chain, discard the first 20, keep the last 1 (n_eff=1).
    # This is the cheap path — the per-epoch Laplacian batch is n_chains·n_eff, and n_eff drove
    # the cost (was 30 with the old 80/20/2). Raise chain_length/thinning only if autocorrelation
    # demands it. (n_eff = (chain_length − thermalization_steps) // thinning_factor.)
    chain_length: int = 21
    thermalization_steps: int = 20
    thinning_factor: int = 1
    target_acceptance: float = 0.5
    # homogeneous-gas sampling regime: start walkers uniformly across the box and carry them
    # across epochs, so the chain equilibrates once instead of restarting in a speck every epoch
    # (the default normal/cold-start is disastrous when L ~ (N/x)^{1/3} is hundreds of a).
    init_positions: str = "uniform"
    warm_walkers: bool = True
    # early stopping: end a run once the three-referee verdict passes `es_patience` checks in a
    # row (after `es_min_epochs`), so `n_epochs` is only a ceiling. es_target_rel_err>0 also
    # requires err/|E| below it before stopping. Set early_stop=False for a fixed-length run.
    early_stop: bool = True
    es_check_every: int = 50
    es_min_epochs: int = 200
    es_patience: int = 2
    es_target_rel_err: float = 0.0  # 0 = verdict-only (no relative-error gate)
    es_plateau_rel: float = 0.0     # >0: also stop when tail-E improves < this/check (NN drift)
    # parameter retrieval: keep the best ``snapshot_frac`` of epochs (by ``select``, lower=better)
    select: str = "std"
    snapshot_frac: float = 0.10
    model_with_pbc: bool = (
        True  # whether the model sees the periodic boundary (default: yes) if True, the model has 2x inputs (sin/cos)
    )

    def k_best(self) -> int:
        """Number of param snapshots to retain = ``snapshot_frac`` of the epochs (>=1)."""
        import math

        return max(1, math.ceil(self.snapshot_frac * self.n_epochs))

    def to_dict(self) -> dict:
        return {
            "phi_hidden": list(self.phi_hidden),
            "F_hidden": list(self.F_hidden),
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "n_chains": self.n_chains,
            "sampler": self.sampler,
            "step_size": self.step_size,
            "chain_length": self.chain_length,
            "thermalization_steps": self.thermalization_steps,
            "thinning_factor": self.thinning_factor,
            "target_acceptance": self.target_acceptance,
            "init_positions": self.init_positions,
            "warm_walkers": self.warm_walkers,
            "model_with_pbc": self.model_with_pbc,
            "early_stop": self.early_stop,
            "es_check_every": self.es_check_every,
            "es_min_epochs": self.es_min_epochs,
            "es_patience": self.es_patience,
            "es_target_rel_err": self.es_target_rel_err,
            "es_plateau_rel": self.es_plateau_rel,
            "select": self.select,
            "snapshot_frac": self.snapshot_frac,
        }


# ── the result ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PointResult:
    """Outcome of one run, energies **per particle in paper units**."""

    potential: Potential
    x: float
    N: int
    seed: int
    hp: HP

    e_per_n: float  # E/N, paper units (ℏ²/2ma²)
    err_per_n: float  # error of the mean, paper units
    sigma_e_per_n: float  # sqrt(Var(E_loc))/N, paper units (per-sample spread)
    acceptance: float
    passed: bool  # three-referee verdict
    verdict: dict  # full result.diagnose() dict (engine-unit tail_energy etc.)

    L: float  # box side used (= (N/x)^(1/3))
    upper_bound: float  # Eq. 31 first-order UB at this (x, potential), paper units

    # ── artifacts (in-process only; persisted to the run dir by io.write_run_artifacts,
    #    never part of the DB key/scalar row) ──
    history_rows: tuple = field(
        default=(), compare=False, repr=False
    )  # per-epoch dicts (engine units)
    snapshots: tuple = field(
        default=(), compare=False, repr=False
    )  # best-k: {step, metric, params}


# ── the run ────────────────────────────────────────────────────────────────────────


def _build_model(N: int, L: float, hp: HP) -> LogWavefunction:
    return LogWavefunction(
        n_particles=N,
        n_dim=N_DIM,
        transform=PeriodicBoundary(L=L) if hp.model_with_pbc else None,
        network=DeepSet(phi_hidden=list(hp.phi_hidden), F_hidden=list(hp.F_hidden)),
    )


def run_point(
    potential: Potential,
    x: float,
    N: int,
    seed: int,
    hp: HP,
    *,
    checkpoint_dir: str | None = None,
) -> PointResult:
    """Train one (potential, x, N) point and return E/N etc. in paper units.

    ``checkpoint_dir`` (set by the sweep to this run's own dir) is where the engine drops its
    end-of-run ``checkpoints/final_state.msgpack``; leaving it ``None`` writes under the cwd
    (fine for one-off calls, but the sweep always passes a per-run dir to avoid collisions).
    """
    L = box_side_for_gas_parameter(x, a=1.0, N=N)
    boundary = PeriodicBoundary(
        L=L
    )  # Here the boundary is always periodic, since the soft-sphere gas is defined in a box with PBCs.

    # Box-aware MH step. The relevant length is the interparticle spacing d = L / N^{1/3}
    # (= x^{-1/3} a). Start at ~½ d and let the adaptive controller refine toward
    # target_acceptance; raise max_step well above the default 5.0 so it isn't capped in the
    # big dilute boxes (where d can be tens of a). step_size>0 in HP overrides the auto value.
    spacing = L / N ** (1.0 / 3.0)
    step0 = hp.step_size if hp.step_size > 0 else 0.5 * spacing
    max_step = max(5.0, 0.5 * L)

    hamiltonian = PenetrableSphereHamiltonian(
        n_dim=N_DIM,
        R=potential.R,
        V0=engine_V0(potential.V0_paper),  # ← /2 : paper -> engine (ℏ=m=1)
        boundary=boundary,
    )

    callbacks = []
    if hp.early_stop:
        callbacks.append(EarlyStopCallback(
            check_every=hp.es_check_every,
            min_epochs=hp.es_min_epochs,
            patience=hp.es_patience,
            target_rel_err=hp.es_target_rel_err or None,
            plateau_rel=hp.es_plateau_rel or None,
        ))

    result = train(
        shape=(hp.n_chains, N * N_DIM),
        model=_build_model(N, L, hp),
        optimizer=optax.adam(hp.lr),
        hamiltonian=hamiltonian,
        training_config=TrainingConfig(
            n_epochs=hp.n_epochs,
            rng_seed=seed,
            is_update_step_size=True,
            target_acceptance=hp.target_acceptance,
            save_checkpoints=False,
            checkpoint_path=checkpoint_dir or "./",
            init_positions=hp.init_positions,
            warm_walkers=hp.warm_walkers,
            max_step=max_step,
        ),
        sampler_params=SamplingConfig(
            step_size=step0,
            chain_length=hp.chain_length,
            thermalization_steps=hp.thermalization_steps,
            thinning_factor=hp.thinning_factor,
            box_L=L,
            sampler=hp.sampler,
        ),
        coord_mode=LabCoords(),
        callbacks=callbacks,
        select=hp.select,
        k_best=hp.k_best(),  # retain the best snapshot_frac of epochs (by `select`)
    )

    verdict = result.diagnose(print_report=False)
    # Record how the run terminated: epochs actually run (< n_epochs ⇒ early-stopped) + where.
    verdict["epochs_ran"] = len(result.history)
    verdict["early_stopped_at"] = callbacks[0].stopped_at if callbacks else None
    verdict["early_stop_reason"] = callbacks[0].stop_reason if callbacks else None

    # Tail estimates live in the verdict (engine units); convert to per-particle paper units.
    e_engine = float(verdict["tail_energy"])
    err_engine = float(verdict["tail_error_of_mean"])
    tail = list(result.history)[len(result.history) // 2 :]
    sigma_engine = float(np.mean([float(s.std) for s in tail]))
    # acceptance_rate is per-chain (an array); average over chains then over the tail
    acceptance = float(
        np.mean([float(np.mean(np.asarray(s.acceptance_rate))) for s in tail])
    )

    # Per-epoch history (engine units) and the best-k param snapshots, for the run dir.
    history_rows = tuple(_history_row(s) for s in result.history)
    snapshots = tuple(
        result.best_k()
    )  # each: {"step", "metric", "params"} (host pytrees)

    return PointResult(
        potential=potential,
        x=x,
        N=N,
        seed=seed,
        hp=hp,
        e_per_n=to_paper_energy(e_engine) / N,
        err_per_n=to_paper_energy(err_engine) / N,
        sigma_e_per_n=to_paper_energy(sigma_engine) / N,
        acceptance=acceptance,
        passed=bool(verdict["passed"]),
        verdict=verdict,
        L=L,
        upper_bound=first_order_energy_upper_bound(x, potential.V0_paper, potential.R),
        history_rows=history_rows,
        snapshots=snapshots,
    )


def _history_row(record) -> dict:
    """One per-epoch metrics row (engine units, scalars only — per-chain arrays reduced to means).

    ``acceptance_rate`` is per-chain; we store its mean. Energy/std stay in engine units here
    (``CONVENTIONS.md``): the run-dir writer adds paper-unit columns via ``to_paper_energy``.
    """

    def val(name, as_int=False):
        try:
            v = record[name]
        except (KeyError, AttributeError):
            return None
        arr = np.asarray(v)
        scalar = (
            float(arr.mean()) if arr.size > 1 else float(arr)
        )  # reduce per-chain arrays
        return int(round(scalar)) if as_int else scalar

    return {
        "epoch": val("step", as_int=True),
        "energy_engine": val("energy"),
        "std_engine": val("std"),
        "error_of_mean_engine": val("error_of_mean"),
        "acceptance": val("acceptance_rate"),
        "step_size": val("step_size"),
        "cm_mean": val("cm_mean"),
        "cm_std": val("cm_std"),
        "wall_time": val("wall_time"),
    }


# ── free, training-free correctness gate: uniform gas ⟨V⟩/N must equal Eq. 31 ──────


def box_fits_interaction(potential: Potential, x: float, N: int) -> bool:
    """Whether the interaction range fits the minimum-image box: ``R < L/2``.

    For ``R >= L/2`` the soft-sphere overlaps its own periodic images and the dilute picture
    (and Eq. 31) breaks down. With ``L = (N/x)^(1/3)`` this caps ``x``: SS10 (R=10) needs
    ``L > 20`` ⇒ ``x < N/8000``. The sweep must respect this per potential.
    """
    return potential.R < 0.5 * box_side_for_gas_parameter(x, a=1.0, N=N)


def sanity_check_uniform_potential(
    potential: Potential, x: float, N: int, n_samples: int = 40_000, seed: int = 0
) -> dict:
    """Draw particles **uniformly** and check ⟨V⟩/N against the finite-N mean field (paper units).

    For an uncorrelated (uniform-density) gas the mean potential per particle is, for N points in
    a min-image box with ``R < L/2``,

        ⟨V⟩/N = ½ V0 (N-1)/L³ (4π/3) R³  =  Eq.31 · (N-1)/N,

    i.e. the first-order upper bound ``½ρṼ(0)`` (Eq. 31) reduced by the finite-N pair-counting
    factor ``(N-1)/N``. This exercises the potential, the engine bridge, and ``L = (N/x)^(1/3)``
    with **no training** — run it before trusting any curve. Requires ``R < L/2`` to be meaningful
    (``fits_box``); otherwise Eq. 31 itself is invalid.
    """
    L = box_side_for_gas_parameter(x, a=1.0, N=N)
    hamiltonian = PenetrableSphereHamiltonian(
        n_dim=N_DIM,
        R=potential.R,
        V0=engine_V0(potential.V0_paper),
        boundary=PeriodicBoundary(L),
    )
    key = jax.random.PRNGKey(seed)
    samples = (
        jax.random.uniform(key, (n_samples, N * N_DIM)) * L
    )  # uniform in [0, L)^(N*d)
    v_engine = np.asarray(hamiltonian.potential_energy(samples))
    measured = to_paper_energy(float(v_engine.mean())) / N
    eq31 = first_order_energy_upper_bound(x, potential.V0_paper, potential.R)
    expected = eq31 * (N - 1) / N  # finite-N mean field
    sem = to_paper_energy(float(v_engine.std() / np.sqrt(n_samples))) / N
    return {
        "measured_V_per_N": measured,
        "expected_finite_N": expected,
        "eq31_thermodynamic": eq31,
        "sem": sem,
        "rel_error": abs(measured - expected) / expected,
        "n_sigma": abs(measured - expected) / (sem + 1e-300),
        "fits_box": potential.R < 0.5 * L,
        "L": L,
    }
