"""The atomic experiment unit: one VMC training run of the dilute soft-sphere gas.

The **runq target** is the flat :func:`run_point` at the bottom — one plain function with
keyword defaults, one flat namespace:

    runq run point.py --axis x=1e-5,1e-4,1e-3 --axis N=32,64 --seeds 0 1 2 --gpus 0,1

(use ``enqueue_curve.py`` for the box-feasible log-spaced x grid). Internally the code keeps
the conceptual grouping: :class:`Potential` + ``x`` + ``N`` are the **physics** (define the
true answer), :class:`HyperParams` is the **solver**, and
:func:`train_point(potential, x, N, seed, hyper_params)` does one training run, returning a
:class:`PointResult` in **paper units** (see ``CONVENTIONS.md``). A box-infeasible point
(``R >= L/2``) raises :class:`runq.Skip` and is recorded as skipped, not failed.

The single factor-of-two engine bridge (paper ``ℏ²/2m=1`` ↔ engine ``ℏ=m=1``) lives in
``dilute_gas.engine_V0`` / ``dilute_gas.to_paper_energy`` and is applied here, nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
from dilute_gas import (
    box_side_for_gas_parameter,
    engine_V0,
    first_order_energy_upper_bound,
    soft_sphere_V0_for_scattering_length,
    softcore_jastrow_params,
    to_paper_energy,
)
from flax import linen as nn
from jastrow import SoftCoreJastrow
from runq import Skip

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
class HyperParams:
    """Solver knobs — everything about *how* we solve, nothing about *what* we solve."""

    # model (DeepSet inside a LogWavefunction)
    phi_hidden: tuple[int, ...] = (64,)
    F_hidden: tuple[int, ...] = (64,)
    # Analytic two-body soft-core Jastrow (jastrow.py) as the short-range correlation factor:
    # log Ψ = Σ_{i<j} j(r_ij) + log Ψ_NN. Off by default (the DeepSet alone is mean-field and sits
    # at the uncorrelated UB Eq.31); turn on to capture the r_ij correlation hole and head toward
    # Lee-Yang. α/Rc are fixed by the potential (frozen, no params), so it adds zero optimiser cost.
    use_jastrow: bool = False
    # Build the analytic Jastrow from a *different* potential than the Hamiltonian (ablation): the
    # short-range correlation hole is then tuned for the WRONG potential, and the DeepSet has to
    # repair it. None ⇒ use the run's own (physical) potential — the normal, matched case. Set to a
    # core range R' (e.g. 5.0 on an R=10 / SS10 run) to plug SS5's Jastrow into an SS10 calculation.
    # α/Rc are taken from Potential.from_R(jastrow_R) (V0 fixed by a=1), so only R' is needed. Must
    # satisfy Rc'=R' < L/2 of every run point (guaranteed when jastrow_R ≤ the physical R).
    jastrow_R: float | None = None
    # Include the neural network (DeepSet) at all. False ⇒ bare analytic Jastrow only (no trainable
    # params): a fixed two-body-correlated trial whose energy is the lowest-order (Bijl–Jastrow)
    # value — the analog of the paper's IPC/SR. Pair with use_jastrow=True (else log ψ = 0 = ideal
    # gas). See _ZeroNetwork.
    use_network: bool = True
    # Pooled-latent size of the DeepSet (the φ output / aggregation width). This is the real
    # capacity bottleneck: phi_hidden/F_hidden only widen the per-particle and readout MLPs, while
    # *all* particle information is squeezed through this one vector before F sees it. The Deep Sets
    # universality result needs it ≥ N for the mean-pool to be injective; the default 20 is far
    # below that. Bumping it (e.g. 256) tests how much a one-body ansatz can fake correlations — it
    # still can't build an r_ij-dependent correlation hole (that needs a Jastrow), so expect
    # marginal gains at rising cost. (Was hardcoded to 20 in DeepSet.)
    hidden_internal_dim: int = 20
    # Laplacian (kinetic energy) estimator. "forward_ad" is exact, O(DoF) JVPs/sample — fine at
    # small N but the per-epoch cost grows with N·d (768 JVPs at N=256, d=3). "hutchinson" is the
    # stochastic trace estimator: hutchinson_n_terms probe vectors instead of DoF, vmapped (often
    # faster on GPU at large N), at the price of extra kinetic-energy variance — raise n_terms if
    # the error bars or gradient noise blow up. The engine threads a fresh per-epoch key, so it is
    # unbiased. Keep "forward_ad" when you need the cleanest E/N for the N→∞ extrapolation.
    # "folx" = forward-Laplacian (LapNet): one forward pass yields gradient + Laplacian
    # together — exact like forward_ad but without the O(DoF) sequential JVP loop, so it
    # is the fast exact path at large N (where hutchinson was previously the only escape).
    laplacian_method: str = "forward_ad"  # "forward_ad" | "folx" | "hutchinson" | "central_difference"
    hutchinson_n_terms: int = 16
    hutchinson_distribution: str = "rademacher"  # "rademacher" (variance-optimal) | "gaussian"
    folx_sparsity_threshold: int = 0  # folx only; 0 = dense (see folx docs; helps at large N)
    # optimizer
    lr: float = 3e-3
    # LR schedule (Adam): "constant" (default), "cosine" (lr → lr·lr_final_frac over n_epochs,
    # smooth — good default to settle into a minimum), or "exponential" (same endpoint, geometric).
    # One optimizer step per epoch, so decay_steps = n_epochs; with early stopping the schedule
    # simply isn't fully traversed. For the low-LR exact-AD fine-tune, set a small lr + "constant"
    # (or a short "cosine"). Stops you from hand-hunting a single fixed LR.
    lr_schedule: str = "constant"  # "constant" | "cosine" | "exponential"
    lr_final_frac: float = 0.1     # final LR as a fraction of lr (cosine alpha / exp decay_rate)
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
            "use_jastrow": self.use_jastrow,
            "jastrow_R": self.jastrow_R,
            "use_network": self.use_network,
            "lr_schedule": self.lr_schedule,
            "lr_final_frac": self.lr_final_frac,
            "hidden_internal_dim": self.hidden_internal_dim,
            "laplacian_method": self.laplacian_method,
            "hutchinson_n_terms": self.hutchinson_n_terms,
            "hutchinson_distribution": self.hutchinson_distribution,
            "folx_sparsity_threshold": self.folx_sparsity_threshold,
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
    hp: HyperParams

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


def _build_optimizer(hp: HyperParams) -> optax.GradientTransformation:
    """Adam with the chosen LR schedule. ``optax.adam`` accepts a float or a step→lr schedule."""
    steps = max(1, hp.n_epochs)  # one optimizer update per epoch
    if hp.lr_schedule == "constant":
        lr = hp.lr
    elif hp.lr_schedule == "cosine":
        lr = optax.cosine_decay_schedule(hp.lr, decay_steps=steps, alpha=hp.lr_final_frac)
    elif hp.lr_schedule == "exponential":
        lr = optax.exponential_decay(hp.lr, transition_steps=steps, decay_rate=hp.lr_final_frac)
    else:
        raise ValueError(f"unknown lr_schedule {hp.lr_schedule!r}")
    return optax.adam(lr)


class _ZeroNetwork(nn.Module):
    """A parameter-free network that outputs log-contribution 0 — i.e. *no* neural part.

    With ``use_network=False`` the ansatz is the bare analytic Jastrow, ``log ψ = Σ_{i<j} j(r_ij)``:
    a *fixed* trial wavefunction (no trainable parameters), so "training" just MC-samples |ψ_J|²
    and averages the local energy — the lowest-order (Bijl–Jastrow) energy, the analog of the
    paper's IPC/SR two-body-correlation-only result.
    """

    @nn.compact
    def __call__(self, x):  # x: (..., N, ppd) after LogWavefunction's reshape
        return jnp.zeros((*x.shape[:-2], 1), dtype=x.dtype)


def _build_model(N: int, L: float, hp: HyperParams, potential: Potential) -> LogWavefunction:
    jastrow = None
    if hp.use_jastrow:
        # Analytic two-body soft-core correlation factor (the r_ij-dependent hole the DeepSet
        # can't build). α/Rc are fixed by the potential (a_s=1); see jastrow.py / spec §9.
        # hp.jastrow_R lets the Jastrow be tuned for a DIFFERENT potential than the Hamiltonian
        # (mismatch ablation): None ⇒ matched (the run's own potential).
        jastrow_pot = Potential.from_R(hp.jastrow_R) if hp.jastrow_R is not None else potential
        alpha, Rc = softcore_jastrow_params(jastrow_pot.V0_paper, jastrow_pot.R)
        jastrow = SoftCoreJastrow(n_particles=N, n_dim=N_DIM, L=L, alpha=alpha, Rc=Rc)
    network = (
        DeepSet(
            phi_hidden=list(hp.phi_hidden),
            F_hidden=list(hp.F_hidden),
            hidden_internal_dim=hp.hidden_internal_dim,
        )
        if hp.use_network
        else _ZeroNetwork()  # bare-Jastrow run: no neural part, no trainable params
    )
    return LogWavefunction(
        n_particles=N,
        n_dim=N_DIM,
        transform=PeriodicBoundary(L=L) if hp.model_with_pbc else None,
        network=network,
        jastrow=jastrow,
    )


def train_point(
    potential: Potential,
    x: float,
    N: int,
    seed: int,
    hp: HyperParams,
    *,
    checkpoint_dir: str | None = None,
    init_params=None,
) -> PointResult:
    """Train one (potential, x, N) point and return E/N etc. in paper units.

    ``checkpoint_dir`` (set by the sweep to this run's own dir) is where the engine drops its
    end-of-run ``checkpoints/final_state.msgpack``; leaving it ``None`` writes under the cwd
    (fine for one-off calls, but the sweep always passes a per-run dir to avoid collisions).

    ``init_params`` warm-starts from an earlier run's parameters (a ``{"params": ...}`` pytree, e.g.
    ``artifacts.load_params(dir)["params"][0]``) with a *fresh* optimizer — a separate training that
    merely begins at a good point. The fine-tune pattern: fast ``laplacian_method="hutchinson"`` run
    → reload its best params → re-run here with a small ``lr`` and ``laplacian_method="forward_ad"``.
    See :func:`fine_tune`.
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
        laplacian_method=hp.laplacian_method,
        hutchinson_n_terms=hp.hutchinson_n_terms,
        hutchinson_distribution=hp.hutchinson_distribution,
        folx_sparsity_threshold=hp.folx_sparsity_threshold,
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
        model=_build_model(N, L, hp, potential),
        optimizer=_build_optimizer(hp),
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
        init_params=init_params,
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


def fine_tune(
    potential: Potential,
    x: float,
    N: int,
    seed: int,
    ft_hp: HyperParams,
    *,
    from_run_dir: str,
    snapshot_index: int = 0,
    checkpoint_dir: str | None = None,
) -> PointResult:
    """Warm-started fine-tune: a *separate* training that starts from a previous run's best params.

    Reloads ``best_params.msgpack`` from ``from_run_dir`` (an ``outputs/runs/<run_id>/`` dir, e.g.
    the fast Hutchinson run) and trains afresh with ``ft_hp`` — typically a small ``lr`` and
    ``laplacian_method="forward_ad"`` (exact, low-variance) to polish the Hutchinson minimum. The
    optimizer state is fresh (step 0); only the parameters carry over. ``snapshot_index`` picks
    which retained snapshot to seed from (0 = best by the source run's ``select``).

    Tip: the source run's snapshots are ranked by ``hp.select`` (default "std"); under Hutchinson
    noise σ_E is jittery, so seeding from a run trained with ``select="energy"`` or
    ``select="e_plus_sigma"`` often gives a cleaner starting point.
    """
    import os

    import artifacts

    lp = artifacts.load_params(os.path.join(from_run_dir, "best_params.msgpack"))
    init = lp["params"][snapshot_index]
    return train_point(
        potential, x, N, seed, ft_hp, checkpoint_dir=checkpoint_dir, init_params=init
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
        "grad_norm": val("grad_norm"),
        "theta_ratio": val("theta_ratio"),
        "cm_mean": val("cm_mean"),
        "cm_std": val("cm_std"),
        "wall_time": val("wall_time"),
    }


# ── the runq target: one flat function, one flat namespace ─────────────────────────
#
# Defaults here MUST mirror the HyperParams dataclass — test_point_contract.py enforces it.
# Representation notes for the flat interface:
#   * phi_hidden / F_hidden : dash-separated widths as a string ("64", "128-128"), because
#     the CLI already uses commas to separate axis values;
#   * jastrow_R             : 0.0 means "matched" (build the Jastrow from the run's own
#     potential, i.e. the dataclass's None); pass e.g. 5.0 for the mismatch ablation.


def _parse_hidden(spec: str) -> tuple[int, ...]:
    """``"64" -> (64,)``, ``"128-128" -> (128, 128)`` (dash-separated layer widths)."""
    try:
        widths = tuple(int(w) for w in str(spec).split("-") if w != "")
    except ValueError:
        widths = ()
    if not widths:
        raise ValueError(f"bad hidden spec {spec!r}; use dash-separated widths, e.g. 128-128")
    return widths


def run_point(
    # physics (potential R at a=1; gas parameter x; particle number N)
    R=10.0,
    x=1e-4,
    N=64,
    # solver: model
    phi_hidden="64",
    F_hidden="64",
    use_jastrow=False,
    jastrow_R=0.0,  # 0 = matched (the run's own potential); >1 = mismatch ablation R'
    use_network=True,
    hidden_internal_dim=20,
    laplacian_method="forward_ad",
    hutchinson_n_terms=16,
    hutchinson_distribution="rademacher",
    folx_sparsity_threshold=0,
    # solver: optimizer
    lr=3e-3,
    lr_schedule="constant",
    lr_final_frac=0.1,
    n_epochs=400,
    # solver: sampler
    n_chains=512,
    sampler="mh",
    step_size=0.0,
    chain_length=21,
    thermalization_steps=20,
    thinning_factor=1,
    target_acceptance=0.5,
    init_positions="uniform",
    warm_walkers=True,
    # solver: early stopping
    early_stop=True,
    es_check_every=50,
    es_min_epochs=200,
    es_patience=2,
    es_target_rel_err=0.0,
    es_plateau_rel=0.0,
    # solver: parameter retention + boundary features
    select="std",
    snapshot_frac=0.10,
    model_with_pbc=True,
    # runq
    seed=0,
    run_dir=None,
) -> dict:
    """Train one soft-sphere point (the runq target). Returns scalar metrics in paper units.

    Raises :class:`runq.Skip` when the box is infeasible (``R >= L/2``), so the point is
    recorded as skipped rather than failed. Artifacts (``meta.json``, ``history.csv``,
    ``verdict.json``, ``best_params.msgpack``) go into ``run_dir`` (injected by runq).
    """
    import artifacts

    potential = Potential.from_R(R)
    if not box_fits_interaction(potential, x, N):
        L = box_side_for_gas_parameter(x, a=1.0, N=N)
        raise Skip(f"R={R:g} >= L/2={L / 2:g} (box too small; needs x < N/(8R^3))")

    hyper_params = HyperParams(
        phi_hidden=_parse_hidden(phi_hidden),
        F_hidden=_parse_hidden(F_hidden),
        use_jastrow=use_jastrow,
        jastrow_R=jastrow_R if jastrow_R else None,
        use_network=use_network,
        hidden_internal_dim=hidden_internal_dim,
        laplacian_method=laplacian_method,
        hutchinson_n_terms=hutchinson_n_terms,
        hutchinson_distribution=hutchinson_distribution,
        folx_sparsity_threshold=folx_sparsity_threshold,
        lr=lr, lr_schedule=lr_schedule, lr_final_frac=lr_final_frac, n_epochs=n_epochs,
        n_chains=n_chains, sampler=sampler, step_size=step_size,
        chain_length=chain_length, thermalization_steps=thermalization_steps,
        thinning_factor=thinning_factor, target_acceptance=target_acceptance,
        init_positions=init_positions, warm_walkers=warm_walkers,
        early_stop=early_stop, es_check_every=es_check_every,
        es_min_epochs=es_min_epochs, es_patience=es_patience,
        es_target_rel_err=es_target_rel_err, es_plateau_rel=es_plateau_rel,
        select=select, snapshot_frac=snapshot_frac, model_with_pbc=model_with_pbc,
    )
    result = train_point(potential, x, N, seed, hyper_params, checkpoint_dir=run_dir)
    if run_dir is not None:
        artifacts.write_artifacts(run_dir, result)
    return {
        "e_per_n": result.e_per_n,
        "err_per_n": result.err_per_n,
        "sigma_e_per_n": result.sigma_e_per_n,
        "acceptance": result.acceptance,
        "passed": result.passed,
        "L": result.L,
        "upper_bound": result.upper_bound,
        "epochs_ran": result.verdict.get("epochs_ran"),
    }


def feasible_x_grid(potential: Potential, N: int, x_lo=1e-5, x_hi=1e-2, n=8) -> np.ndarray:
    """Log-spaced x grid clipped to the box-feasible range ``R < L/2 <=> x < N/(8R^3)``.

    For an N-ladder, build the grid with the *smallest* N (the most restrictive), so every x
    is feasible at every N and the fixed-x ``1/N -> 0`` extrapolation has data at each rung.
    """
    x_max_box = N / (8.0 * potential.R**3)
    return np.geomspace(x_lo, min(x_hi, 0.999 * x_max_box), n)


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
