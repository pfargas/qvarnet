# Running the soft-sphere gas sweeps & reading the outputs

Reproducing **Mazzanti, Polls & Fabrocini (2003)**, *Energy and structure of dilute hard- and
soft-sphere Bose gases* (`../../comments/hard-soft-spheres-bosons.pdf`) with qvarnet NQS-VMC.

Sweeps run on **runq** (the sibling `~/personal-os/phd/runq` package; `uv sync` in qvarnet
installs it, on a cluster `pip install -e ~/runq`). The project-side code is:

* `point.py` — the physics. The runq target is the flat `run_point(...)` (one plain function,
  keyword defaults); internally it builds `Potential` + `HyperParams` and calls `train_point`.
  A box-infeasible point (`R >= L/2`) raises `runq.Skip` → recorded as `skipped`, never run.
* `enqueue_curve.py` — the physics-aware grid planner: box-feasible log-spaced x grids.
* `artifacts.py` — what a run leaves in its run dir; `check_db.py` — corridor sanity check;
  `analysis.py` — curves, seed aggregation, `1/N → 0` extrapolation.

Read `CONVENTIONS.md` first for the physics (units, the factor-of-two engine bridge, the sweep
design). This file is only about *operating* the pipeline.

---

## 1. How to run

### The curve sweep (feasible x grid → drain on GPUs)

The x grid must be clipped to the box ceiling `x < N/(8R³)` of the *smallest* N — that is
physics, so it lives in `enqueue_curve.py`, not the generic CLI:

```bash
cd soft_sphere_gas

# fill the queue: SS10, N-ladder, seeds, 7-point feasible x grid + fixed solver settings
python enqueue_curve.py --R 10 --N 32 64 128 --seeds 0 1 2 --n-x 7 \
    --set use_jastrow=true --set lr=3e-3 --set lr_schedule=cosine \
    --set n_chains=1024 --set n_epochs=2000 --set es_plateau_rel=0.005

# drain it: one worker per GPU, all sharing the queue
runq run point.py --db outputs/runq.db --gpus 0,1

runq status / runq failed / runq requeue [--failed]
python check_db.py --db outputs/runq.db          # physics corridor check
```

Ad-hoc grids (no feasibility clipping needed) go straight through the CLI — one flat `--axis`
vocabulary for physics and solver alike; defaults and types come from `run_point`'s signature:

```bash
runq run point.py --axis x=1e-5,1e-4 --axis N=64 --axis phi_hidden=64,128-128 --seeds 0 1
```

(`phi_hidden`/`F_hidden` take dash-separated widths — `128-128` = two layers of 128;
`jastrow_R=0` means "matched", `--axis jastrow_R=5` is the mismatch ablation.)

**Fully resumable.** Re-running skips `done` points and requeues anything left `running`
(a killed process). The DB key is the full parameter dict, so changing any value produces a
*new* run rather than overwriting. Infeasible points show up as `skipped` with the reason.

**GPU memory.** On an 8 GB card, N=64 OOMs at 2048 chains — use `--set n_chains=1024`
(~5.6 s/epoch). One worker per *distinct* GPU (JAX preallocates VRAM). Rough budget at 1024
chains / 2000 epochs: ~3 hr per (x, seed) point.

### On a cluster (conda, no uv)

```bash
conda activate <env>
pip install -e ~/qvarnet --no-deps && pip install -e ~/runq
export MPLBACKEND=Agg
cd ~/dilute-bose
python ~/qvarnet/soft_sphere_gas/enqueue_curve.py --R 10 ... --db outputs/runq.db
python -m runq.cli run ~/qvarnet/soft_sphere_gas/point.py --db outputs/runq.db --gpus 0,1
```

Outputs land relative to your CWD (`outputs/` next to the `--db`). On SLURM, wrap the same
two commands in an sbatch script with the DB on `$SLURM_TMPDIR` and rsync back (runq README).

### Multi-PC (independent machines, merge afterwards)

SQLite coordinates within **one** machine; across machines, partition the sweep and merge:

1. **Identical settings on every PC** (same `--set`/`--axis` values ⇒ same parameter dicts ⇒
   rows merge as the same physics points). *Only the partition axis and `--db`/`--gpus` differ.*
2. **Partition by seed** (one seed per PC, each running the full N-ladder) — after merging,
   every `(x, N, seed)` is present so the `1/N` extrapolation works unchanged.

```bash
# per PC (inside tmux):  --seeds 0 / 1 / 2, own --db
python enqueue_curve.py --R 10 --N 32 64 128 256 --seeds 0 --n-x 10 --set ... --db outputs/pc1.db
runq run point.py --db outputs/pc1.db

# gather + merge on one machine
runq merge merged.db pc1.db pc2.db pc3.db
rsync -a pc1/outputs/runs/ merged/runs/        # union run dirs (labels unique per point)
python check_db.py --db merged.db
```

WAL note: flush before copying a live DB — `sqlite3 outputs/runq.db "PRAGMA wal_checkpoint(TRUNCATE);"`.

### A single point (no DB, quick check)

```bash
uv run --project .. python -c "from point import run_point; \
print(run_point(x=1e-4, N=64, n_epochs=2000, n_chains=1024, seed=0))"
```

### Early stopping (don't waste epochs)

`n_epochs` is a **ceiling**: by default each run stops once converged (`early_stop=true`).
Two triggers (stop when either holds `es_patience` checks in a row after `es_min_epochs`):
**verdict** (three-referee convergence: stationary + at MC floor + chains mixed) and
**plateau** (opt-in, `es_plateau_rel>0`: tail-mean energy improved less than that per check —
more aggressive, saves the most GPU time). The run's `verdict.json` records `epochs_ran`,
`early_stopped_at`, `early_stop_reason`. `--set early_stop=false` for fixed-length runs.

---

## 2. What is computed

For each parameter point the pipeline trains a neural-network wavefunction by VMC and records
the converged energy plus diagnostics. The unit of work is the flat `point.run_point`.

- **R** — the soft (penetrable) sphere core range at scattering length `a = 1`; `V0` is fixed
  by Eq.10 (`Potential.from_R`). SS10 = `R=10`, SS5 = `R=5`; `R=1` is the hard-sphere limit
  (not representable — use R slightly above 1 to approach it).
- **x** — the gas parameter `ρa³`. The box side is `L = (N/x)^(1/3)`. Box-feasibility requires
  `R < L/2` ⇔ `x < N/(8R³)`; infeasible points are recorded as `skipped`, not run.
- **N** — particle number. **seed** — RNG. **solver** — everything else in the signature
  (model widths, Jastrow, Laplacian estimator, lr, chains, sampler, early stop, snapshots).
- **model** — a permutation-invariant `DeepSet` inside a `LogWavefunction` with periodic
  (sin/cos) boundary features, optionally × the analytic soft-core two-body Jastrow
  (`use_jastrow`; `use_network=false` = bare-Jastrow baseline).

All energies are reported in **paper units** (`ℏ²/2m=1`, `a=1`, energies in `ℏ²/2ma²`). The
engine runs `ℏ=m=1`; the single factor-of-two bridge lives only in `dilute_gas.engine_V0`
(V0/2 in) and `dilute_gas.to_paper_energy` (×2 out). See `CONVENTIONS.md`.

**Parameter retention.** Training keeps the best `snapshot_frac` of epochs (default 10%)
ranked by `select` (default `"std"` = lowest σ_E) — these snapshots are what later analysis
(g(r), S(k), n(k)/n₀) loads. The final-epoch state is also kept.

**Verdict.** Each run gets the engine's three-referee convergence verdict
(`result.diagnose()`): stationary? at the MC floor? chains mixed? `passed` is the AND of the
three. Curve aggregation only uses `passed` runs by default.

---

## 3. Output architecture & labeling

```
outputs/
├── runq.db                                    # SQLite index: one row per run
└── runs/
    └── R10_x1e-04_N64_seed0_3f9a1c2b/         # label = swept axes + hash8 of full params
        ├── run.json             # runq's record: params + returned metrics
        ├── meta.json            # identity, units, full hyper-params, box L, paper benchmarks
        ├── history.csv          # per-epoch metrics (engine + paper-unit columns)
        ├── verdict.json         # full three-referee diagnose() dict
        ├── best_params.msgpack  # best-10% param snapshots by σ_E (best first)
        └── checkpoints/final_state.msgpack
```

### DB schema (`runs`)

Generic runq schema: `params_json (UNIQUE, the full parameter dict), label, status,
result_json, run_dir, error, started_at, finished_at`. Status:
`todo → running → done | failed | skipped`. The result metrics are
`e_per_n, err_per_n, sigma_e_per_n, acceptance, passed, L, upper_bound, epochs_ran`
(paper units) inside `result_json`.

### `history.csv` columns (per epoch)

`epoch, energy_engine, std_engine, error_of_mean_engine, e_per_n_paper, sigma_per_n_paper,
acceptance, step_size, cm_mean, cm_std, wall_time`. Use the `*_paper` columns for physics.

---

## 4. Reading / extracting results

### Sanity-check the physics (`check_db.py`)

```bash
python check_db.py --db outputs/runq.db
```

Checks every done point against the rigorous corridor (`4πx ≤ E/N ≤ Eq.31`) and Lee-Yang,
prints the queue status, failures, and the seed-aggregated curve. "all done points inside
[4πx, Eq31]" = the physics is correct.

### The curve (seed-averaged, verdict-gated)

```python
from runq import connect
from analysis import load_curve, plot_curve
from point import SS10

conn = connect("outputs/runq.db")
curve = load_curve(conn, R=10.0, N=64)         # dict: x, e_per_n, err, upper_bound, n_per_x
plot_curve(curve, SS10, save="outputs/ss10_curve_N64.png")
```

### Everything, as a DataFrame

```python
from runq import connect, load_table
df = load_table(connect("outputs/runq.db"))    # params + result metrics as columns
df[["R", "x", "N", "seed", "e_per_n", "err_per_n", "passed"]]
```

### Finite-size extrapolation (the N-ladder)

```python
from analysis import extrapolate_thermodynamic, plot_extrapolations
fits = extrapolate_thermodynamic(conn, R=10.0, N_list=[32, 64, 128])
plot_extrapolations(fits, SS10)
```

### The best-10% parameters (for observables / re-evaluation)

```python
import artifacts
blob = artifacts.load_params("outputs/runs/<label>/best_params.msgpack")
best = blob["params"][0]          # full Flax variables dict: {"params": {...}}
# rebuild the same model (point._build_model with the run's hyper-params + L) and apply.
```

### Warm-started fine-tune

`point.fine_tune(...)` reloads a previous run's best params (e.g. a fast Hutchinson run) and
retrains with new `HyperParams` (small lr + `laplacian_method="forward_ad"`) — Python API,
see its docstring.

---

## 5. Moving outputs between a cluster and a PC

The whole `outputs/` tree is self-contained and host-independent. To move it:

```bash
sqlite3 outputs/runq.db "PRAGMA wal_checkpoint(TRUNCATE);"   # flush the WAL first
rsync -av cluster:/path/to/outputs/  ./outputs/
```

Merging machines: rows are keyed by the full parameter dict, so `runq merge` only fills in
missing points (done-precedence), and re-pointing a sweep at a copied DB extends it.

> **Old DBs** (pre-runq `soft_sphere.db` with the fixed-column schema) are readable with the
> old code from git history (`git log -- soft_sphere_gas/db.py`); ask for a migration script
> if a campaign needs to be folded into a runq DB.
