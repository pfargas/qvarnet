# A ground-up walkthrough of the soft-sphere Bose-gas project

**Who this is for.** Someone comfortable with Variational Monte Carlo (VMC) — you know what a
trial wavefunction, a local energy, a Metropolis walker and the variational principle are — but
who has *never worked on cold-atom / Bose-gas physics*. This document explains the physics from
scratch, every modelling assumption, every engineering decision, and the limitations, with
pointers into the actual code (`file:line`).

It is deliberately long. If you want the *operational* manual (commands, flags, output formats),
read [`RUNNING.md`](RUNNING.md). If you want the terse physics/units cheat-sheet, read
[`CONVENTIONS.md`](CONVENTIONS.md). This file is the "explain it to me like I've never seen a
Bose gas" version that ties the two together.

The goal of the whole project: **reproduce Mazzanti, Polls & Fabrocini, *"Energy and structure of
dilute hard- and soft-sphere Bose gases"* (PRA 68, 023611, 2003; arXiv:cond-mat/0305502)** with a
neural-network VMC ansatz instead of their correlated/DMC method. The paper PDF is at
`../../comments/hard-soft-spheres-bosons.pdf`.

---

## Part 1 — The physics, from scratch

### 1.1 What system are we simulating?

`N` identical **bosons** (spinless, so the spatial wavefunction is fully *symmetric* under
exchanging any two particles) living in a **cubic box of side `L` with periodic boundary
conditions (PBC)**. There is **no external trap** — the particles are free except that they
*interact with each other*. This is the textbook model of a uniform (homogeneous) quantum gas:
density `ρ = N/L³` everywhere, no edges.

The Hamiltonian (paper Eq. 7 + 9):

```
H = −(ℏ²/2m) Σᵢ ∇²ᵢ  +  Σ_{i<j} V(r_ij),     V(r) = V₀  for r < R, else 0.
```

Two pieces:

- **Kinetic**: ordinary Laplacian per particle.
- **Interaction**: a pairwise **soft (penetrable) sphere** — a flat repulsive step of height `V₀`
  whenever two particles are closer than `R`. "Penetrable" / "soft" because `V₀` is *finite*:
  particles *can* overlap, they just pay energy `V₀` for it. Contrast the **hard sphere**
  (`V₀ → ∞`), where overlap is strictly forbidden and the wavefunction must vanish at `r = R`.

Why soft and not hard? A hard core forces a *node* in ψ at contact (ψ = 0 for `r < R`), which is
brutal for a smooth neural-network ansatz — the network would have to learn a discontinuous
derivative. The soft core keeps ψ smooth and differentiable everywhere, which is exactly what
makes it "NN-friendly". This is stated in the Hamiltonian's docstring,
`src/qvarnet/hamiltonian/periodic.py:63-66`.

### 1.2 The single most important concept: the scattering length `a`

Here is the one idea you must absorb to understand everything else.

A dilute gas almost never "feels" the detailed shape of the interaction potential. When two
particles are far apart most of the time (dilute!), the only thing that matters about their
collisions is a *single number* that summarises low-energy two-body scattering: the **s-wave
scattering length `a`**. Two completely different potentials (a soft step of height `V₀` and range
`R`; a hard sphere of radius `a`; a real atomic potential) that happen to have the *same* `a`
produce the *same* dilute-gas physics to leading order. `a` is the "effective size" the particles
present to each other.

For the soft-sphere step, `a` is known in closed form (paper Eq. 10):

```
a = R [ 1 − tanh(K₀R)/(K₀R) ],     K₀ = √(V₀ · m/ℏ²).
```

Code: `dilute_gas.soft_sphere_scattering_length` (`dilute_gas.py:99`). Read the two limits in its
docstring (`dilute_gas.py:106-108`):

- `V₀ → ∞` (impenetrable): `a → R`. The soft sphere becomes a hard sphere of radius `R`.
- `V₀ → 0` (no barrier): `a → 0`. Non-interacting.

So `a` slides between 0 and `R` as you raise the wall.

> **Numerical aside the code handles for you.** For small `K₀R`, `1 − tanh(z)/z` suffers
> catastrophic cancellation (you subtract two nearly-equal numbers). `dilute_gas.py:115-119`
> switches to the Taylor series `z²/3 − 2z⁴/15 + …` below `K₀R < 1e-3`. This is the kind of detail
> that silently poisons a scattering-length calculation if you forget it.

### 1.3 Why we fix `a = 1` and infer `V₀`

Because only `a` matters physically, we are free to choose our **unit of length to be `a`
itself**: set `a = 1`. Then a "soft sphere" is described by **one free number, not two**: pick the
range `R`, and the wall height `V₀` is *forced* by the requirement `a(V₀, R) = 1`.

That inversion is `dilute_gas.soft_sphere_V0_for_scattering_length` (`dilute_gas.py:123`) — a
robust bisection on `K₀R` (the map is monotone, so bisection can't go wrong). The two systems the
paper studies are built this way in `point.py:74-75`:

```python
SS10 = Potential.from_R(10.0, "SS10")   # R = 10 a,  V₀ inferred  → 0.00681670 (paper units)
SS5  = Potential.from_R(5.0,  "SS5")    # R =  5 a,  V₀ inferred  → 0.06308561
```

`Potential.from_R` (`point.py:64`) refuses `R ≤ 1`, because `R = 1` *is* the hard sphere
(`a → R = 1` needs `V₀ → ∞`, not representable). This is why the class docstring at
`point.py:53-57` says "use `R > 1`".

**Why does `V₀` look absurdly small (0.0068)?** It's not weak — it's *wide*. A barrier `R = 10`
scattering lengths across, tuned to scatter like a tiny `a = 1`, must be shallow. The honest
dimensionless interaction strength is `β = (K₀R)² ≈ 0.34` for SS10 — an O(1) number. Full
explanation with the Born-approximation estimate in `CONVENTIONS.md` §2.1.

### 1.4 The gas parameter `x = ρa³` — the diluteness knob

Now combine the two-body scale `a` with the many-body density `ρ = N/L³` into the single
dimensionless **gas parameter**:

```
x = ρ a³.
```

This is *the* control parameter of a dilute Bose gas. With `a = 1` it's simply `x = ρ` (the
density in units of `a⁻³`). Small `x` = very dilute = particles rarely within `a` of each other.
The paper sweeps `x` from `10⁻⁵` (extremely dilute) to `10⁻²` (moderately dense).

Crucially, in our unit system the box size is **not** a free parameter — it is *determined* by
`x` and `N`:

```
ρ = N/L³ = x/a³ = x   ⇒   L = (N/x)^{1/3}.
```

Code: `dilute_gas.box_side_for_gas_parameter` (`dilute_gas.py:161`). This single line is why
dilute runs are expensive: at `x = 10⁻⁵`, `N = 64`, the box is `L ≈ 186` — a *huge* volume for the
walkers to explore (see Part 7 on the step-size consequence).

So a "physics point" in this study is the triple **(which potential = `R`, gas parameter `x`,
particle number `N`)**. Everything else — `V₀`, `L` — is derived. This is exactly the split
encoded in `run_point(potential, x, N, seed, hp)` (`point.py:176`), discussed in Part 4.

### 1.5 The two dimensionless axes (and what each physically means)

| knob | meaning | values in this study |
|---|---|---|
| `x = ρa³` | **diluteness** | `10⁻⁵ … 10⁻²` (the main sweep) |
| `R/a` | **shape / softness** of the core | SS10 → 10, SS5 → 5, hard sphere → 1 |

Everything else is an overall scale set to 1. The first axis (`x`) is the energy-curve sweep; the
second (`R`) is the "shape dependence" story — how much the *detailed* shape of the potential
(beyond just `a`) shifts the energy. The whole point of the paper is that at low `x` only `a`
matters (universal), but as `x` grows the shape (`R/a`) starts to matter (non-universal).

### 1.6 What the answer should look like (Lee–Yang)

For a dilute Bose gas the energy per particle has a famous low-density expansion (Lee–Yang, paper
Eq. 1), here in our units (`ℏ²/2m = 1`, `a = 1`):

```
E/N = 4πx [ 1 + (128/15)√(x/π) + … ].
```

Code: `dilute_gas.lee_yang_energy_per_particle` (`dilute_gas.py:173`). Two things to take from it:

- The leading term **`4πx`** is a *rigorous lower bound* on the exact energy (Lieb–Yngvason).
  Any correct calculation must lie **above** it.
- The full Lee–Yang expression is *universal*: it depends only on `x` and `a`, not on the shape.
  So as `x → 0`, **every** short-range potential with `a = 1` must converge to this same curve.
  That makes Lee–Yang our low-`x` validation target.

This is what makes the project falsifiable: we know, analytically, what the dilute end must give.

---

## Part 2 — Why this is a VMC problem, and what's unusual about it

You know VMC. Here is what's *different* about this particular problem versus, say, a harmonic
trap or a small molecule:

1. **Periodic, homogeneous, no trap.** There's no confining potential to localise ψ. The
   wavefunction must be *translationally invariant* and *L-periodic*. A Gaussian envelope (common
   in trapped problems) is actively *wrong* here — it would break periodicity. The engine warns
   about exactly this mistake in `src/qvarnet/models/compose.py:53-63`.

2. **The energy is tiny and the signal is in a small correction.** `E/N ≈ 4πx`. At `x = 10⁻⁴`
   that's `≈ 1.3 × 10⁻³`. We're trying to resolve a few-percent *correction* on top of an already
   small number. This demands low-variance estimators and careful convergence checks — hence the
   three-referee verdict (Part 5) and the per-sample-spread tracking.

3. **Bosons ⇒ symmetric ⇒ permutation invariance is mandatory.** The ansatz must be invariant
   under particle exchange. We get this *architecturally* with a **DeepSet** (Part 4.4), not by
   symmetrising by hand.

4. **The interaction is the whole physics.** With no trap, the *only* reason the energy isn't just
   the free-gas value is the pairwise soft core. So the wavefunction must learn the two-body
   correlation hole — the suppression of amplitude when two particles are within `R`. A plain
   DeepSet can represent *some* of this, but the cleanest representation is an explicit pairwise
   Jastrow factor (Part 7 — not yet implemented; this is the project's main known gap).

---

## Part 3 — Units and the one factor of two (the decision most likely to bite you)

This is the single most error-prone part of the whole project, so it gets its own section. There
are **two different unit conventions in play**, and exactly one bridge between them.

### 3.1 The paper's convention (what we report in)

`ℏ²/2m = 1`, lengths in units of `a` (`a = 1`), energies in units of `ℏ²/2ma²`. In this
convention the kinetic operator is **`−∇²`** (no ½), and Eq. 10 reduces to `K₀² = V₀/2`. We adopt
this everywhere in the *problem layer* so our numbers drop straight onto the paper's tables and
figures. The constant `HBAR2_OVER_M = 2.0` at `dilute_gas.py:64` encodes the `m/ℏ² = 1/2`.

### 3.2 The engine's convention (what qvarnet computes in)

The qvarnet engine works in `ℏ = m = 1`, so its kinetic operator is **`−½∇²`**. You can see this
directly in the local kinetic energy estimator `src/qvarnet/hamiltonian/kinetic.py:27`:

```python
return -0.5 * (lap + jnp.sum(grad_log_psi**2, axis=-1))   # = −½(Δlog|ψ| + |∇log|ψ||²)
```

(That formula is the standard log-domain local kinetic energy for a real ψ; every model in qvarnet
outputs `log|ψ|`, so this is what's used.)

### 3.3 The bridge: a factor of two, in exactly two places

The two conventions differ by a factor of two in the kinetic term. To make the engine compute the
*paper's* Hamiltonian, you must compensate. The rule (and the entire reason it's safe) is that the
correction lives in **two helpers and nowhere else**:

| direction | helper | factor | code |
|---|---|---|---|
| build the Hamiltonian | `engine_V0(V0_paper)` | `× ½` | `dilute_gas.py:70` |
| read the energy back | `to_paper_energy(E_engine)` | `× 2` | `dilute_gas.py:80` |

You can see both applied in `run_point`: the Hamiltonian is built with
`V0=engine_V0(potential.V0_paper)` (`point.py:199`), and every energy coming out is wrapped in
`to_paper_energy(...)` before being divided by `N` (`point.py:253-255`). **Lengths (`a`, `R`, `L`) are identical in both conventions — only `V₀` and
the energy carry the 2.**

Why does halving `V₀` and doubling `E` give the paper's answer? Because rescaling `H → H/2`
(engine vs paper kinetic) and also halving `V₀` makes the *engine's* `H_engine = ½ H_paper`; its
eigen-energies are then `½ E_paper`, so multiplying the result by 2 recovers `E_paper`. Worked
through in `CONVENTIONS.md` §4.

> **Get this wrong and you are silently off by 2× in `V₀` (→ 5.7× in `x` via `a ∝ V₀`) or 2× in
> energy.** That's why it's regression-tested: `test_paper_published_potentials` and
> `test_engine_bridge_is_a_factor_of_two` in `test_dilute_gas.py`. Run them (`uv run python -m
> pytest soft_sphere_gas/test_dilute_gas.py`) before trusting any number.

---

## Part 4 — The code, layer by layer

The architecture is a thin **problem layer** (`soft_sphere_gas/`) sitting on top of the general
**qvarnet engine** (`src/qvarnet/`). The problem layer knows about bosons, scattering lengths and
the paper; the engine knows about VMC, sampling and autodiff. The contract between them is one
function, `run_point`.

```
              soft_sphere_gas/  (this project)              src/qvarnet/  (the engine)
   ┌──────────────────────────────────────────┐   ┌────────────────────────────────────┐
   │ dilute_gas.py   units, a(V₀,R), x↔L, L-Y  │   │ train()            VMC driver        │
   │ point.py        run_point: 1 training run ├──▶│ PenetrableSphereHamiltonian  V        │
   │ sweep.py        nest x/N/seed/HP, resume  │   │ LogWavefunction + DeepSet    ψ        │
   │ db.py           SQLite index              │   │ PeriodicBoundary   sin/cos + min-img  │
   │ artifacts.py    per-run output files      │   │ kinetic_log        local kinetic E    │
   │ analysis.py     plot E/N(x)               │   │ SnapshotCallback   best-k params      │
   └──────────────────────────────────────────┘   └────────────────────────────────────┘
```

### 4.1 `dilute_gas.py` — the physics/units layer (no VMC here)

Pure functions, no state, no engine calls. This is where all the boson-specific formulae live:

- `engine_V0` / `to_paper_energy` (`:70`, `:80`) — the factor-of-two bridge (Part 3).
- `soft_sphere_scattering_length` (`:99`) and its inverse `soft_sphere_V0_for_scattering_length`
  (`:123`) — Eq. 10 forwards and backwards.
- `box_side_for_gas_parameter` (`:161`) — `L = (N/x)^{1/3}`.
- `lee_yang_energy_per_particle` (`:173`) — the low-`x` benchmark (Eq. 1).
- `first_order_energy_upper_bound` (`:188`) — Eq. 31, explained in Part 5.

This module is the most heavily unit-tested (24 tests in `test_dilute_gas.py`) because it's pure
and the formulae are exact — there's no excuse for a wrong scattering length.

### 4.2 `point.py` — one training run = `run_point`

This is the heart. Three dataclasses and one function.

**`Potential`** (`point.py:51`) — a soft sphere. Frozen, carries `(R, V0_paper, label)`; built via
`from_R` so `V₀` is always consistent with `a = 1`.

**`HP`** (`point.py:81`) — "**H**yper**P**arameters": *every solver knob*, and nothing physical.
Model widths (`phi_hidden`, `F_hidden`), optimiser (`lr`, `n_epochs`), sampler (`n_chains`,
`step_size`, `chain_length`, `thermalization_steps`, `thinning_factor`, `target_acceptance`), and
the snapshot policy (`select`, `snapshot_frac`). It is **frozen and serialisable** (`to_dict`,
`point.py:113`) on purpose: that's what lets `(potential, x, N, seed, hp)` be a database key and a
reproducible run identity (Part 6). `k_best()` (`point.py:107`) turns `snapshot_frac` into "keep
the best ⌈frac·n_epochs⌉ parameter snapshots" (Part 4.7).

**`PointResult`** (`point.py:134`) — the outcome, energies **per particle in paper units**. The
scalar fields (`e_per_n`, `err_per_n`, `passed`, …) become the DB row; the two trailing fields
`history_rows` and `snapshots` (`point.py:156-161`, marked `compare=False, repr=False`) carry the
heavy per-epoch data *in process only* — they are written to files, never into the DB row.

**`run_point`** (`point.py:176`) — the actual work, and worth reading top to bottom:

1. `L = box_side_for_gas_parameter(x, a=1, N)` — derive the box from the physics (`:191`).
2. Build `PenetrableSphereHamiltonian(n_dim=3, R, V0=engine_V0(V0_paper), boundary=PeriodicBoundary(L))`
   (`:196-201`) — note the `engine_V0` bridge applied right here.
3. Build the model via `_build_model` (`:167`) — a `LogWavefunction` wrapping a `DeepSet`, with the
   `PeriodicBoundary` transform (Parts 4.3–4.4).
4. Call the engine `train(...)` (`:203`) with an Adam optimiser, the `TrainingConfig` and
   `SamplingConfig` assembled from `hp`, `coord_mode=LabCoords()`, and
   `select=hp.select, k_best=hp.k_best()` so the engine retains the best snapshots (Part 4.7).
5. `verdict = result.diagnose(print_report=False)` — the three-referee convergence check (Part 5).
6. Convert energies: `to_paper_energy(e_engine)/N` etc. (`point.py:253-255`) — the *only* place the
   energy crosses back to paper units.
7. Reduce the per-epoch history to plain rows via `_history_row` (`point.py:266`) and grab the
   best-k snapshots, attach both to the `PointResult`.

There are also two **training-free** helpers at the bottom:

- `box_fits_interaction` (`point.py:300`) — the constraint `R < L/2` (Part 4.6).
- `sanity_check_uniform_potential` (`point.py:310`) — draw particles *uniformly* (no training) and
  check that `⟨V⟩/N` equals the analytic mean-field `Eq.31 · (N−1)/N`. This exercises the
  potential, the bridge and `L` with zero training — the cheapest possible correctness gate. It's
  how we know the engine bridge is right to <0.1%.

### 4.3 The Hamiltonian: `PenetrableSphereHamiltonian`

`src/qvarnet/hamiltonian/periodic.py:56`. The interesting method is `potential_energy`
(`:78`): reshape the flat `(batch, N·d)` walker into `(batch, N, d)`, form all pairwise
displacement vectors, apply the **minimum-image convention** `self._min_image(dx)` (`:83`), compute
pair distances, count how many pairs are closer than `R`, multiply by `V₀`:

```python
inside = (r_pairs < self.R)
return self.V0 * jnp.sum(inside, axis=-1)     # V₀ × (number of overlapping pairs)
```

That's the whole soft-sphere potential: a count of overlapping pairs times `V₀`. The **kinetic**
energy is handled generically by the engine (`kinetic_log`, Part 3.2); the Hamiltonian subclass
only has to supply `potential_energy`. Note `V0` here is the *engine* `V₀` (already halved).

### 4.4 The wavefunction: `LogWavefunction(DeepSet, PeriodicBoundary)`

Two engine pieces compose into ψ.

**`PeriodicBoundary`** (`src/qvarnet/boundaries.py:64`) does two jobs:

- `encode(x)` (`:74`) maps each raw coordinate to **`[sin(2πx/L), cos(2πx/L)]`**. This is how the
  network is made *exactly L-periodic*: feeding sin/cos of the position means the model literally
  cannot tell `x` from `x + L`. Note the careful *interleaving* per coordinate (`:75-82`) so that a
  downstream per-particle reshape keeps each particle's features together — a naive
  `concat([sin(all), cos(all)])` would scramble the DeepSet input.
- `min_image(dx)` (`:87`) = `dx − L·round(dx/L)`, the shortest periodic image of a displacement —
  used by the Hamiltonian for distances.

**`DeepSet`** (`src/qvarnet/models/deep_set.py:12`) gives **permutation invariance for free**.
Its `__call__` (`:51-55`):

```python
h = self.phi(x)            # per-particle embedding φ(rᵢ),  (..., N, hidden)
h = jnp.mean(h, axis=-2)   # SUM/MEAN over particles  →  symmetric pooling
return self.F(h)           # F of the pooled vector    →  log|ψ|, shape (..., 1)
```

Because the pooling is a mean over particles, swapping any two particles leaves the output
unchanged — exactly the boson symmetry we need, baked into the architecture rather than enforced by
hand. `phi_hidden` / `F_hidden` (from `HP`) size the two MLPs.

**`LogWavefunction`** (`src/qvarnet/models/compose.py:9`) is the glue: its `__call__`
(`:50-78`) applies the boundary transform, reshapes to per-particle features, runs the network, and
optionally adds an envelope and/or a Jastrow. For us **envelope = None** (no trap; the engine warns
if you mix an envelope with PBC, `:53-63`) and **jastrow = None** for now (Part 7). So today
`log|ψ| = DeepSet(sin/cos-encoded positions)`.

### 4.5 The sampler and optimiser

Standard VMC, configured from `HP`: Metropolis–Hastings (`sampler="mh"`) with `n_chains` parallel
walkers, `chain_length` proposals per epoch, `thermalization_steps` burn-in, `thinning_factor`
decorrelation, and an **adaptive step size** (`is_update_step_size=True` in
`point.py:211`) that nudges `step_size` toward `target_acceptance = 0.5`. The optimiser is
`optax.adam(hp.lr)` (`point.py:206`). See Part 7 for the step-size pitfall in big dilute boxes.

### 4.6 `sweep.py` — orchestrating many runs, resumably

`run_point` is one point. `sweep.py` nests the axes around it (the hierarchy is spelled out in
`CONVENTIONS.md` §6: shape → x → N → seed → HP).

- **`feasible_x_grid`** (`sweep.py:142`) — the log-spaced `x` grid, *clipped to the box-feasible
  range*. The constraint is `R < L/2` ⇔ `x < N/(8R³)` (minimum-image validity: a particle must not
  interact with its own periodic image). For SS10 (`R=10`), `x < N/8000`. Points above the ceiling
  are recorded `skipped_box`, never run. (We discussed this grid in detail earlier in the session.)
- **HP strategies** (`sweep.py:25-72`): `Frozen` (one fixed HP everywhere) or `ASHA` (successive
  halving over candidate HPs — re-trains at growing epoch budgets, keeps the best by
  `e_per_n + err_per_n` among *verdict-passing* trials). `tune_then_freeze` (`:69`) runs ASHA once
  at an anchor `x` and freezes the winner for the whole sweep — cheap and good enough.
- **`run_one`** (`sweep.py:78`) — the resumable unit: skip if already `done`; skip-box if
  infeasible; else enqueue → mark running → `run_point` → write artifacts → save result. It creates
  the per-run directory up front and passes it as `checkpoint_dir` so the engine's checkpoint can't
  collide in the cwd (Part 6).
- **`sweep_x`** (`sweep.py:95`) — loop `x`, then seeds, calling `run_one`.
- **`load_curve`** (`sweep.py:114`) — aggregate `done`, *verdict-passing* runs into one `E/N` per
  `x`, combining seeds; the error bar is the **larger** of the across-seed SEM and the mean
  within-seed error, so uncertainty is never understated.

### 4.7 Parameter retention: keeping the best 10% of epochs

VMC energies fluctuate epoch-to-epoch. For downstream observables (Part 7) you don't want a single
arbitrary final snapshot — you want the *good* parameter sets. The engine already supports this via
`SnapshotCallback` with `policy="best_k"` (`src/qvarnet/callbacks/snapshot.py:35`): it keeps the
`k` epochs with the lowest selection metric (`select="std"` = lowest per-batch σ_E), copying their
params to host RAM.

We expose this through `HP.snapshot_frac` (default **0.10**) and `HP.k_best()`: keep the best
**10% of epochs** by σ_E. `run_point` passes `select`/`k_best` into `train` and pulls
`result.best_k()` back out — each entry is `{step, metric, params}` (best first). These get written
to `best_params.msgpack` (Part 6). The full round-trip (reload → `model.apply` → finite `log|ψ|`)
is validated.

### 4.8 `analysis.py` — the figure

`plot_curve` (`analysis.py:12`) overlays the VMC `E/N(x)` on the three benchmarks (lower bound
`4πx`, Lee–Yang, Eq.31 upper bound). With `scaled=True` (default) it plots **`E/N / 4πx`** — the
paper's Fig. 1 / Fig. 12 axis — on which the rigorous lower bound is the horizontal line `1` and
all the interesting physics (correlations, shape) is the deviation above it.

---

## Part 5 — How we know it's right: the validation ladder

Four independent checks, increasing in strength (`CONVENTIONS.md` §5):

1. **Rigorous lower bound `4πx`** (Lieb–Yngvason): the leading Lee–Yang term. The exact energy —
   and therefore any honest VMC upper bound that's converged — must lie *above* it.

2. **Lee–Yang `E/N`** (`dilute_gas.py:173`): the universal `x → 0` limit. Our dilute points must
   converge onto it. This is the primary low-`x` target.

3. **First-order upper bound, Eq. 31** (`first_order_energy_upper_bound`, `dilute_gas.py:188`):
   `E₁/N = ½ρV₀(4π/3)R³ = ½ρṼ(0)` — the mean potential energy of the *uncorrelated* (uniform)
   wavefunction. Two uses: (a) by the variational principle the *converged* VMC energy must lie
   **below** it; (b) an *untrained* run evaluated on uniform samples must *equal* it — the
   training-free `sanity_check_uniform_potential` gate (`point.py:310`). This is a free, exact,
   no-training correctness check, and it's how the engine bridge was verified to <0.1%.

4. **DMC points** from the paper's tables: the "exact" benchmark. The hierarchy
   `DMC < VMC < Eq.31` must hold.

A subtlety we learned (recorded in `NEXT_STEPS.md`): **Lee–Yang and Eq.31 cross at `x ≈ 8×10⁻⁴`.**
Below that, Lee–Yang sits below Eq.31 and is a good anchor; above it, Lee–Yang is *not* a valid
bound and you must validate against DMC, not Lee–Yang. So Lee–Yang is an `x → 0` anchor, while the
*rigorous* per-point gate is `VMC ≤ Eq.31`.

**Convergence (per run): the three-referee verdict.** `result.diagnose()` runs three independent
MCMC convergence tests on the energy history: Geweke/Heidelberger–Welch *stationarity*, split-`R̂`
*chain mixing* (≤ 1.1), and an *MC-error-floor* check. `passed` is their AND. `load_curve` only
aggregates `passed` runs, so non-stationary / unmixed runs are discarded rather than polluting the
curve. (See `CONVENTIONS.md` §6, the seed axis.)

**Gate result on record (2026-06-15):** SS10, `x=10⁻⁴`, `N=32` reached `E/N ≈ 0.001406` (paper
units) inside `[4πx=0.0012566, Eq.31=0.0014277]`, descending toward Lee–Yang `0.0013172` — physics
confirmed sane end-to-end. Re-confirmed at `N=64` this session: `E/N ≈ 0.001437` by 200 epochs.

---

## Part 6 — The output architecture (and the decisions behind it)

Every run produces a self-contained, portable, correctly-labelled artifact set. The layout:

```
outputs/
├── soft_sphere.db                         # SQLite index: one row per run (scalars + pointer)
└── runs/SS10_x1.000e-04_N64_s0_3f9a1c2b/   # one run dir; name = run_id
    ├── meta.json            # identity, units, full HP, box L, paper benchmarks at this x
    ├── history.csv          # per-epoch metrics (engine + paper-unit columns)
    ├── verdict.json         # the full three-referee diagnose() dict
    ├── best_params.msgpack  # best-10% param snapshots by σ_E (best first)
    └── checkpoints/final_state.msgpack    # engine's final-epoch state (auxiliary)
```

**Why a SQLite index + per-run dirs (not one big file)?** The DB (`db.py:18` schema) holds only
*scalars* — one row keyed by `(potential, x, N, seed, hp_json)` (`db.py:40`, the UNIQUE
constraint). That makes it tiny, queryable (`pandas.read_sql`, `sqlite3 …`), and the natural place
for sweep bookkeeping: status `todo → running → done | failed | skipped_box`, resumability
(`requeue_interrupted`, `db.py:132`, resets crashed `running` rows), and aggregation
(`fetch_done`, `db.py:139`). The heavy data (per-epoch history, 200 parameter snapshots) lives in
per-run files, linked from the row by the `run_dir` column (added this session, `db.py`).

**The `run_id` label.** `run_id` (`artifacts.py:42`) =
`<potential>_x<x>_N<N>_s<seed>_<hp8>`, e.g. `SS10_x1.000e-04_N64_s0_3f9a1c2b`. The physics axes are
human-readable; `hp8` is the first 8 hex of a SHA-1 of the *full* HP dict — short enough for a
folder name, unique enough that two runs differing in any solver setting don't collide. The full HP
is recorded in both `meta.json` and the DB's `hp_json`, so the hash never has to be inverted.

**Why the module is `artifacts.py`, not `io.py`.** A local `io.py` would *shadow the standard
library `io`* for the whole process, because the run directory is `sys.path[0]` when you launch the
scripts — that would break flax/csv/anything doing `import io`. Renamed deliberately
(`artifacts.py:30-...` header note).

**Why `checkpoint_dir` is threaded through.** The engine auto-adds a `RunOutputCallback` that
writes `checkpoints/final_state.msgpack` to its `checkpoint_path`, defaulting to the cwd. Without
intervention *every* run would clobber the same `./checkpoints/final_state.msgpack`. So `run_one`
creates the per-run dir up front and passes it as `checkpoint_dir` to `run_point`
(`point.py:183`, `:214`), landing the engine's checkpoint *inside* the run dir. We also keep our
*own* best-10% snapshots (Part 4.7), so the engine's single final state is auxiliary.

**Why best-10%-by-σ_E for the saved params.** Downstream observables — `g(r)`, `S(k)`,
`n(k)/n₀` — are evaluated *from the wavefunction*, so they need the trained parameters. Saving only
the final epoch throws away a 60-GPU-hour sweep's worth of reusable ψ if you later want observables.
Saving an *ensemble* of the lowest-variance epochs lets you average observables over good snapshots
(noise reduction) rather than trusting one arbitrary final step. σ_E (`select="std"`) was the
user's chosen ranking; `e_plus_sigma` is also available in the engine if you want to penalise
energy too.

**Portability (cluster ↔ PC).** Nothing stores an absolute path (`run_dir` is relative; params are
plain msgpack/numpy), so the whole `outputs/` tree is copyable. The one gotcha is SQLite's
write-ahead log: `PRAGMA wal_checkpoint(TRUNCATE)` before copying, or copy the `-wal`/`-shm` files
too. Recipe in `RUNNING.md` §5.

`artifacts.load_params` (`artifacts.py:126`) reloads `best_params.msgpack` into
`{select, steps, metrics, params:[pytree, …]}`; each `params[i]` is already a full Flax variables
dict (`{"params": {...}}`) and feeds straight into `model.apply(params[i], x)`.

---

## Part 7 — Limitations and known gaps (read before trusting a result)

1. **No Jastrow factor yet — this gates the hard-sphere end.** Today `log|ψ| = DeepSet(...)` with no
   explicit pairwise correlation factor. A DeepSet can represent *some* two-body correlation through
   the pooled embedding, but it cannot cleanly place the *node/cusp at contact* that a near-hard core
   demands. As `R → 1` (toward the hard sphere) the variance blows up and the ansatz fails. The fix
   is a **3D pairwise Jastrow** added to `LogWavefunction.jastrow`; the shipped `LogJastrow` is **1D
   only** (`CONVENTIONS.md` §6 aside, and `point.py` / README roadmap step 5). **SS10/SS5 at low-to-
   moderate `x` are fine without it; the hard sphere and large-`x` are not.**

2. **We are at finite `N`; the paper is the thermodynamic limit.** Every run is a finite box of `N`
   particles, which carries a finite-size bias. Comparing to the paper requires an **N-ladder + 1/N
   extrapolation** (run several `N`, fit `E/N` vs `1/N`, take the intercept). That's roadmap step 3,
   not yet built — so a single-`N` curve is *qualitatively* right but not yet the TL number.

3. **The box constraint caps `x` at fixed `N`.** `x < N/(8R³)` (minimum-image validity). For SS10,
   `x = 10⁻²` needs `N ≳ 80`. The dense, shape-sensitive regime the paper highlights (`x ≳ 10⁻³`) is
   exactly where you most need *large* `N`, which is also the most expensive. The top point of
   `feasible_x_grid` sits *at* `R = L/2` (the `0.999` factor keeps it just inside) and is the least
   trustworthy point on any curve — consider dropping it or raising `N`.

4. **Step-size adaptation is slow in the huge dilute boxes.** At `x = 10⁻⁴`, `N = 64`, `L ≈ 86`, but
   the default `step_size = 0.3` (`HP`) is tiny relative to the box, so early acceptance is ≈ 0.99
   (walkers barely move) and the adaptive controller takes many epochs to grow the step. It *does*
   converge (E/N ≈ 0.00144 by 200 epochs), but a **box-aware initial `step_size`** would save a lot
   of the epoch budget on a 60-GPU-hour grid. Not yet implemented; flagged for before the next big
   launch.

5. **GPU memory ceiling on the dev machine.** On an 8 GB card (RTX 3070), `N = 64` **OOMs at 2048
   chains**; the driver default was lowered to **1024 chains** (`run_ss10_curve.py`). On a larger
   cluster GPU raise `--chains` for lower variance. Budget ~3 hr per `(x, seed)` point at
   1024 chains / 2000 epochs.

6. **Error bars are MC-naive unless/until autocorrelation-corrected.** The per-epoch
   `error_of_mean` is `σ_E/√M` (naive), upgraded to an autocorrelation-time-aware estimate only in a
   later roadmap step. `load_curve`'s across-seed SEM partially covers this, but treat single-seed
   error bars as optimistic.

7. **Structure observables (`g(r)`, `S(k)`, `n(k)`, condensate fraction `n₀`) are not implemented.**
   These are the headline "NQS-over-DMC" deliverable (`n₀`/ODLRO especially), and the reason we now
   save the best-10% parameters — but the estimators themselves are roadmap step 6, still to come.

8. **`HP.model_with_pbc`** (recently added, `point.py:103`) lets you turn off the sin/cos encoding.
   With PBC *off* the model sees raw coordinates and is no longer L-periodic — only meaningful for
   open-boundary experiments. For the soft-sphere gas it should stay `True`; the Hamiltonian is
   always built with a `PeriodicBoundary` regardless (`point.py:192`).

---

## Part 8 — Where to go next

- **Run it:** [`RUNNING.md`](RUNNING.md) — commands, flags, output formats, cluster↔PC moves.
- **Physics/units cheat-sheet:** [`CONVENTIONS.md`](CONVENTIONS.md).
- **Session handoff / task list:** [`NEXT_STEPS.md`](NEXT_STEPS.md).
- **The roadmap of what's left** (Jastrow, finite-N, shape sweep, observables): steps 3–6 in
  `NEXT_STEPS.md`, and the qvarnet roadmap in `../comments/`.

The one-sentence summary: *we built a thin, tested, resumable problem layer that turns the abstract
control parameters of a dilute Bose gas (`x`, `R`) into concrete qvarnet VMC runs in the paper's
units, validated against analytic bounds, and persists every run (energies + convergence + the best
parameters) in a portable, labelled store — with the Jastrow factor, finite-N extrapolation, and
structure observables as the named next steps.*
