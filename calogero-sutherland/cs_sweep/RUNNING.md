# Running the Calogero-Sutherland grid sweeps & reading the outputs

A resumable, **grid-agnostic** sweep harness for the CS model — the analogue of
`../../soft_sphere_gas/`. The unit of work is `point.run_point(physics, seed, hp)`: one VMC
training run for `N` particles at coupling `L`. The harness saves every run (SQLite index + a
self-contained per-run dir) and lets you retrieve results as a DataFrame or a seed-averaged curve.

**Code convention** (ℏ²/m = 1, ω = 1, as in `calogero_sutherland.ipynb`):
`H = -Σ ∂² + Σ x² + 2L(L-1) Σ 1/(xᵢ-xⱼ)²`, exact ground state `E₀ = N(1 + L(N-1))`.

---

## The split: physics vs solver (what you grid over)

| | dataclass | fields | role |
|---|---|---|---|
| **physics** | `Physics` | `L, N, n_dim, epsilon` | defines the *true answer* `E₀`; part of the DB key |
| **seed** | `int` | — | RNG |
| **solver** | `HP` | `kind, mlp_hidden, lambda_init, lr, n_epochs, n_chains, …, early_stop, select` | how we solve it |

`kind ∈ {jastrow, mlp_jastrow, analytic, mlp}` (same as the notebook's `make_model`). The DB key is
`(physics, seed, hp)`, so changing **any** field — physics or solver — produces a *new* run rather
than overwriting. **Nothing hardcodes which axes vary**: you grid over whatever you pass to
`build_grid`.

---

## 1. How to run

These are plain scripts — no `uv` needed. Run with the `jax` conda env active and `qvarnet`
importable. Outputs go to `./outputs/` **relative to your CWD**. Run the scripts *by path* (or from
this dir) so the sibling imports (`db`, `point`, …) resolve.

```bash
conda activate jax
export MPLBACKEND=Agg          # headless plotting (and avoids the dashboard backend issue)
cd ~/Desktop/PhD/qvarnet/calogero-sutherland/cs_sweep
```

### Multi-GPU (worker-per-GPU) — the fast path

One worker process per GPU, all draining one shared SQLite queue; `db.claim_next` hands each worker
a distinct `(physics, seed, hp)` atomically (`BEGIN IMMEDIATE`), so two GPUs never run the same
point. Axes are **repeatable flags**: `--axis` for physics, `--hp-axis` for solver.

```bash
# grid over L and N, two model kinds, seeds 0..2, on GPUs 0 and 1
python run_workers.py --gpus 0,1 --seeds 0 1 2 \
    --axis L=0.5,0.8,1.0,1.5,2.0 --axis N=2,5,10 \
    --hp-axis kind=jastrow,mlp_jastrow --hp-axis n_epochs=2000 \
    --db outputs/cs.db --out-root outputs
```

`run_workers.py` requeues crashed `running` rows, enqueues the whole grid (`INSERT OR IGNORE`),
then launches one worker per GPU with `CUDA_VISIBLE_DEVICES` pinned. **Fully resumable** — re-run the
same command to extend (add seeds / axes); finished points are skipped. Auto-detects GPUs if
`--gpus` is omitted; with no GPU it runs a single CPU worker. By hand, single device:

```bash
CUDA_VISIBLE_DEVICES=0 python worker.py --db outputs/cs.db --out-root outputs
```

> **One worker per *distinct* GPU.** JAX preallocates most of a GPU's VRAM per process — don't point
> two workers at the same device (you'll OOM) unless you set `XLA_PYTHON_CLIENT_MEM_FRACTION` low.

> **Same node only (SQLite).** The shared queue is a local-filesystem SQLite DB. Across nodes, give
> each its own `--db` and `merge_db.py` afterwards (every row is keyed, so the merge is exact).

### Serial driver (from Python / a notebook)

```python
import sys; sys.path.insert(0, "cs_sweep")   # or run from inside cs_sweep/
import db, sweep
from point import Physics, HP

conn = db.connect("outputs/cs.db")
grid = sweep.build_grid(
    physics_axes={"L": [0.5, 0.8, 1.0, 1.5, 2.0], "N": [2, 5, 10]},
    hp_axes={"kind": ["jastrow", "mlp_jastrow"]},
    base_hp=HP(n_epochs=2000, n_chains=2048),
)
sweep.sweep_grid(conn, grid, seeds=[0, 1, 2], out_root="outputs")   # resumable
```

You can grid over solver knobs the same way (`hp_axes={"lr": [1e-3, 3e-3], "lambda_init": [1.0, 1.2]}`)
or hold N fixed and scan only L (`physics_axes={"L": [...]}`). The harness doesn't care which.

### A single point (no DB, for a quick check)

```python
from point import Physics, HP, run_point
r = run_point(Physics(L=0.8, N=5), seed=0, hp=HP(n_epochs=2000, n_chains=2048))
print(r.e_total, "+/-", r.err_total, " exact", r.e_exact, " gap", r.gap)
```

### Early stopping

`hp.early_stop=True` makes `n_epochs` a ceiling: the run stops once the three-referee verdict (and
optionally a plateau / relative-error gate) passes `es_patience` checks in a row after
`es_min_epochs`. Knobs: `es_check_every`, `es_min_epochs`, `es_patience`, `es_target_rel_err`,
`es_plateau_rel`. Default `early_stop=False` = fixed-length run.

---

## 2. Output architecture

```
outputs/
├── cs.db                              # SQLite index: one row per run (scalars + pointer)
└── runs/
    └── L0.8_N5_s0_3f9a1c2b/           # run_id = <physics-label>_s<seed>_<hp8>
        ├── meta.json                  # physics, hp, exact energy, result
        ├── history.csv                # per-epoch: energy, std, error_of_mean, acceptance, …
        ├── verdict.json               # full three-referee diagnose() dict
        ├── best_params.msgpack        # best snapshot_frac of epochs by `select` (best first)
        └── checkpoints/final_state.msgpack
```

`hp8` = first 8 hex of a SHA-1 of the full HP dict (distinct solver settings → distinct dirs). The
whole `outputs/` tree is self-contained and copyable between machines (no absolute paths; `run_dir`
stored relative to `outputs/`).

### DB schema (`runs`)

Physics and solver settings are stored as **JSON** (`physics_json`, `hp_json`) — *not* fixed columns
— which is what makes the harness grid-agnostic. Result scalars are columns:
`status, e_total, e_per_n, err_total, err_per_n, sigma_e, acceptance, passed, e_exact, gap,
verdict_json, run_dir, error, started_at, finished_at`. Status:
`todo → running → done | failed`.

Query an axis with SQLite JSON1: `json_extract(physics_json,'$.L')` — or use `load_table` (below).

---

## 3. Reading / extracting results

### Sanity-check (`check_db.py`)

```bash
python check_db.py --db outputs/cs.db
```
Prints the queue status and every done point's `E/N`, exact `E/N`, the variational **gap** (should
be ≥ 0 within a few σ — the variational principle), verdict, and epochs. A gap well below 0 flags a
bias (e.g. the `epsilon`-softened potential), not a better-than-exact result.

### Everything, as a DataFrame (grid-agnostic)

```python
import db
from sweep import load_table
df = load_table(db.connect("outputs/cs.db"))      # physics fields + hp_* columns expanded
df[["L", "N", "seed", "hp_kind", "e_per_n", "err_per_n", "gap", "passed"]]
```
Whatever axes you swept become columns you can `groupby` — no schema knowledge needed.

### A seed-averaged curve along any axis

```python
from sweep import load_curve
c = load_curve(db.connect("outputs/cs.db"), x_axis="L", fixed={"N": 5})
# dict of sorted arrays: x, e_per_n, err, e_exact (per particle), n_per_x
```
`fixed` selects a slice; seeds are combined per x (error = max of across-seed SEM and mean
within-seed error, never understated). Plot `e_per_n` vs `x` against `e_exact`.

### Per-epoch convergence / best params (one run)

```python
import pandas as pd, artifacts
h = pd.read_csv("outputs/runs/<run_id>/history.csv")        # h.plot(x="epoch", y="energy")
blob = artifacts.load_params("outputs/runs/<run_id>/best_params.msgpack")
best = blob["params"][0]    # full Flax variables dict; rebuild model via point._build_model + apply
```

---

## 4. Multi-machine merge

Each row is keyed by `(physics, seed, hp)`, so spread a sweep over independent machines (partition
by seed), then gather and merge:

```bash
python merge_db.py merged.db pc1.db pc2.db pc3.db      # INSERT OR IGNORE + done-precedence
rsync -a pc1/outputs/runs/ merged/runs/                # union the run dirs (names are unique)
```

WAL note: before copying a live DB, flush it — `sqlite3 outputs/cs.db "PRAGMA wal_checkpoint(TRUNCATE);"`.
