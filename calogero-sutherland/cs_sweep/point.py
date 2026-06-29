"""The atomic experiment unit: one VMC training run of the Calogero-Sutherland model.

``run_point(physics, seed, hp)`` trains a wavefunction for ``physics.N`` particles at coupling
``physics.L`` and returns a :class:`CSResult`. Everything else in the study — the grid sweep, the
seed averaging — just orchestrates calls to this function.

Split of responsibilities (mirrors ``soft_sphere_gas/point.py``):

* ``physics`` (``L``, ``N``, ``n_dim``, ``epsilon``) — defines the **true answer**
  ``E0 = N(1 + L(N-1))`` (code convention ℏ²/m=1, ω=1);
* ``seed``  — RNG;
* ``hp``    — the **solver** (model / optimizer / sampler / epochs): the only thing a tuner varies.

The harness never hardcodes which of these vary — see ``sweep.build_grid``: you can grid over any
subset of physics *or* solver fields. ``(physics, seed, hp)`` is the DB key.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace  # noqa: F401  (replace re-exported for sweeps)

import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from qvarnet.train import train
from qvarnet.callbacks import EarlyStopCallback
from qvarnet.config.coord_mode import LabCoords
from qvarnet.config.training_setup import SamplingConfig, TrainingConfig
from qvarnet.boundaries import NoBoundary
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.mlp import MLP
from qvarnet.models.envelopes import GaussianEnvelope
from qvarnet.models.jastrow import LogJastrow
from qvarnet.models.analytic import CalogeroSutherlandAnalyticModel
from qvarnet.hamiltonian.continuous import CalogeroSutherlandHamiltonian


# ── what we solve: the physics (defines the exact ground-state energy) ───────────────


@dataclass(frozen=True)
class Physics:
    """The CS problem. Frozen + serializable, so it is part of the DB key.

    ``L`` is the coupling (exact Jastrow exponent λ = L; L>1 repulsive, L<1 attractive/harder).
    ``epsilon`` softens the 1/(x_i-x_j)² singularity in the Hamiltonian — it is a property of the
    Hamiltonian (it changes the operator), so it lives here, not in ``HP``.
    """

    L: float = 0.8
    N: int = 5
    n_dim: int = 1
    epsilon: float = 1e-4

    @property
    def dof(self) -> int:
        return self.N * self.n_dim

    def exact_energy(self) -> float:
        """E0 = N(1 + L(N-1))  (code convention ℏ²/m=1, ω=1)."""
        return self.N * (1.0 + self.L * (self.N - 1))

    def to_dict(self) -> dict:
        return {"L": self.L, "N": self.N, "n_dim": self.n_dim, "epsilon": self.epsilon}

    @classmethod
    def from_dict(cls, d: dict) -> "Physics":
        return cls(L=float(d["L"]), N=int(d["N"]),
                   n_dim=int(d.get("n_dim", 1)), epsilon=float(d.get("epsilon", 1e-4)))

    @property
    def label(self) -> str:
        return f"L{self.L:g}_N{self.N}"


# ── how we solve it: hyperparameters (the only thing a tuner varies) ─────────────────


@dataclass(frozen=True)
class HP:
    """Solver knobs. Frozen + serializable, so ``(physics, seed, hp)`` is a DB key."""

    # model
    kind: str = "mlp_jastrow"  # "mlp_jastrow" | "jastrow" | "analytic" | "mlp"
    mlp_width: int = 128       # width of each MLP hidden layer (grid axis)
    mlp_layers: int = 1        # number of MLP hidden layers (grid axis)
    mlp_hidden: tuple[int, ...] | None = None  # explicit per-layer widths; overrides width/layers
    lambda_init: float = 1.2  # Jastrow exponent start; should drift toward L
    # optimizer
    lr: float = 1e-3
    lr_schedule: str = "constant"  # "constant" | "cosine" | "exponential"
    lr_final_frac: float = 0.1
    n_epochs: int = 2000
    # sampler (open boundary — the harmonic trap confines the walkers, no box_L)
    n_chains: int = 4096
    step_size: float = 0.5
    chain_length: int = 21
    thermalization_steps: int = 20
    thinning_factor: int = 1
    target_acceptance: float = 0.5
    init_positions: str = "normal"
    warm_walkers: bool = True
    min_step: float = 1e-5
    max_step: float = 5.0
    # early stopping (n_epochs is then a ceiling). early_stop=False ⇒ fixed-length run.
    early_stop: bool = False
    es_check_every: int = 50
    es_min_epochs: int = 300
    es_patience: int = 2
    es_target_rel_err: float = 0.0  # 0 = verdict-only
    es_plateau_rel: float = 0.0     # >0: also stop on tail-E plateau
    # parameter retrieval: keep the best snapshots by ``select`` (lower=better). ``n_snapshots``
    # is an absolute count (e.g. 100); if None, fall back to ``snapshot_frac`` of the epochs.
    # Prefer the absolute count for long runs — ``snapshot_frac`` of 50k epochs is 5000 pytrees.
    select: str = "std"
    n_snapshots: int | None = 100
    snapshot_frac: float = 0.10

    def hidden(self) -> list[int]:
        """Per-layer MLP widths: explicit ``mlp_hidden`` if set, else ``[width] * layers``."""
        if self.mlp_hidden is not None:
            return list(self.mlp_hidden)
        return [self.mlp_width] * self.mlp_layers

    def k_best(self) -> int:
        """Number of param snapshots to retain: absolute ``n_snapshots`` if set, else a fraction."""
        import math
        if self.n_snapshots is not None:
            return max(1, self.n_snapshots)
        return max(1, math.ceil(self.snapshot_frac * self.n_epochs))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "mlp_width": self.mlp_width,
            "mlp_layers": self.mlp_layers,
            "mlp_hidden": list(self.mlp_hidden) if self.mlp_hidden is not None else None,
            "lambda_init": self.lambda_init,
            "lr": self.lr,
            "lr_schedule": self.lr_schedule,
            "lr_final_frac": self.lr_final_frac,
            "n_epochs": self.n_epochs,
            "n_chains": self.n_chains,
            "step_size": self.step_size,
            "chain_length": self.chain_length,
            "thermalization_steps": self.thermalization_steps,
            "thinning_factor": self.thinning_factor,
            "target_acceptance": self.target_acceptance,
            "init_positions": self.init_positions,
            "warm_walkers": self.warm_walkers,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "early_stop": self.early_stop,
            "es_check_every": self.es_check_every,
            "es_min_epochs": self.es_min_epochs,
            "es_patience": self.es_patience,
            "es_target_rel_err": self.es_target_rel_err,
            "es_plateau_rel": self.es_plateau_rel,
            "select": self.select,
            "n_snapshots": self.n_snapshots,
            "snapshot_frac": self.snapshot_frac,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HP":
        d = dict(d)
        mh = d.get("mlp_hidden")
        d["mlp_hidden"] = tuple(mh) if mh is not None else None
        return cls(**d)


# ── the result ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CSResult:
    """Outcome of one run (code-convention energies, ℏ²/m=1, ω=1)."""

    physics: Physics
    seed: int
    hp: HP

    e_total: float       # tail-averaged E
    e_per_n: float       # E / N
    err_total: float     # error of the mean on E
    err_per_n: float
    sigma_e: float       # sqrt(Var(E_loc)) (per-sample spread)
    acceptance: float
    passed: bool         # three-referee verdict
    verdict: dict        # full result.diagnose() dict

    e_exact: float       # N(1 + L(N-1))
    gap: float           # e_total - e_exact

    # artifacts (in-process only; persisted to the run dir by artifacts.write_run_artifacts)
    history_rows: tuple = field(default=(), compare=False, repr=False)
    snapshots: tuple = field(default=(), compare=False, repr=False)


# ── the run ────────────────────────────────────────────────────────────────────────


class _NoNetwork(nn.Module):
    """Network contributing nothing to log|ψ|: returns zeros of shape (..., 1)."""

    @nn.compact
    def __call__(self, x):
        return jnp.zeros((*x.shape[:-1], 1))


def _build_optimizer(hp: HP) -> optax.GradientTransformation:
    steps = max(1, hp.n_epochs)
    if hp.lr_schedule == "constant":
        lr = hp.lr
    elif hp.lr_schedule == "cosine":
        lr = optax.cosine_decay_schedule(hp.lr, decay_steps=steps, alpha=hp.lr_final_frac)
    elif hp.lr_schedule == "exponential":
        lr = optax.exponential_decay(hp.lr, transition_steps=steps, decay_rate=hp.lr_final_frac)
    else:
        raise ValueError(f"unknown lr_schedule {hp.lr_schedule!r}")
    return optax.adam(lr)


def _build_model(physics: Physics, hp: HP) -> object:
    """Mirror of the calogero_sutherland notebook's ``make_model`` (HP-parametrised)."""
    N, kind = physics.N, hp.kind
    if kind == "analytic":
        return CalogeroSutherlandAnalyticModel(lambda_init=hp.lambda_init)
    if kind == "mlp_jastrow":
        return LogWavefunction(
            transform=NoBoundary(),
            network=MLP(hidden=hp.hidden()),
            envelope=GaussianEnvelope(),
            jastrow=LogJastrow(n_particles=N, lambda_init=hp.lambda_init),
        )
    if kind == "mlp":
        return LogWavefunction(
            transform=NoBoundary(),
            network=MLP(hidden=hp.hidden()),
            envelope=GaussianEnvelope(),
        )
    if kind == "jastrow":
        # exact functional form: Gaussian envelope × exp(λ Σ log|x_i-x_j|), no network
        return LogWavefunction(
            transform=NoBoundary(),
            envelope=GaussianEnvelope(),
            jastrow=LogJastrow(n_particles=N, lambda_init=hp.lambda_init),
            network=_NoNetwork(),
        )
    raise ValueError(f"unknown model kind {kind!r}")


def run_point(
    physics: Physics,
    seed: int,
    hp: HP,
    *,
    checkpoint_dir: str | None = None,
    init_params=None,
) -> CSResult:
    """Train one (physics, seed, hp) point and return E etc. in the code convention."""
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
        shape=(hp.n_chains, physics.dof),
        model=_build_model(physics, hp),
        optimizer=_build_optimizer(hp),
        hamiltonian=CalogeroSutherlandHamiltonian(L=physics.L, epsilon=physics.epsilon),
        training_config=TrainingConfig(
            n_epochs=hp.n_epochs,
            rng_seed=seed,
            is_update_step_size=True,
            target_acceptance=hp.target_acceptance,
            save_checkpoints=False,
            checkpoint_path=checkpoint_dir or "./",
            init_positions=hp.init_positions,
            warm_walkers=hp.warm_walkers,
            min_step=hp.min_step,
            max_step=hp.max_step,
        ),
        sampler_params=SamplingConfig(
            step_size=hp.step_size,
            chain_length=hp.chain_length,
            thermalization_steps=hp.thermalization_steps,
            thinning_factor=hp.thinning_factor,
            sampler="mh",
        ),
        coord_mode=LabCoords(),
        callbacks=callbacks,
        select=hp.select,
        k_best=hp.k_best(),
        init_params=init_params,
    )

    verdict = result.diagnose(print_report=False)
    verdict["epochs_ran"] = len(result.history)
    verdict["early_stopped_at"] = callbacks[0].stopped_at if callbacks else None
    verdict["early_stop_reason"] = callbacks[0].stop_reason if callbacks else None

    e_total = float(verdict["tail_energy"])
    err_total = float(verdict["tail_error_of_mean"])
    tail = list(result.history)[len(result.history) // 2:]
    sigma_e = float(np.mean([float(s.std) for s in tail]))
    acceptance = float(np.mean([float(np.mean(np.asarray(s.acceptance_rate))) for s in tail]))

    e_exact = physics.exact_energy()
    return CSResult(
        physics=physics,
        seed=seed,
        hp=hp,
        e_total=e_total,
        e_per_n=e_total / physics.N,
        err_total=err_total,
        err_per_n=err_total / physics.N,
        sigma_e=sigma_e,
        acceptance=acceptance,
        passed=bool(verdict["passed"]),
        verdict=verdict,
        e_exact=e_exact,
        gap=e_total - e_exact,
        history_rows=tuple(_history_row(s) for s in result.history),
        snapshots=tuple(result.best_k()),
    )


def _history_row(record) -> dict:
    """One per-epoch metrics row (scalars only — per-chain arrays reduced to their mean)."""

    def val(name, as_int=False):
        try:
            v = record[name]
        except (KeyError, AttributeError, TypeError):
            try:
                v = getattr(record, name)
            except AttributeError:
                return None
        arr = np.asarray(v)
        scalar = float(arr.mean()) if arr.size > 1 else float(arr)
        return int(round(scalar)) if as_int else scalar

    return {
        "epoch": val("step", as_int=True),
        "energy": val("energy"),
        "std": val("std"),
        "error_of_mean": val("error_of_mean"),
        "acceptance": val("acceptance_rate"),
        "step_size": val("step_size"),
        "cm_mean": val("cm_mean"),
        "cm_std": val("cm_std"),
        "wall_time": val("wall_time"),
    }
