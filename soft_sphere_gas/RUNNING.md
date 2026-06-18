# Running the soft-sphere gas sweeps & reading the outputs

Reproducing **Mazzanti, Polls & Fabrocini (2003)**, *Energy and structure of dilute hard- and
soft-sphere Bose gases* (`../../comments/hard-soft-spheres-bosons.pdf`) with qvarnet NQS-VMC.

This document covers: **(1)** how to launch a run, **(2)** what is actually computed, **(3)** the
output architecture and how runs are labeled, **(4)** how to read/extract results for analysis,
and **(5)** how to move outputs between a cluster and a PC.

Read `CONVENTIONS.md` first for the physics (units, the factor-of-two engine bridge, the sweep
design). This file is only about *operating* the pipeline.

---

## 1. How to run

### On a cluster with conda (no `uv`)

The scripts have **no dependency on `uv`** — that's just the local dev launcher. With an activated
conda env you only need `qvarnet` importable and the deps present (`jax`+CUDA, `flax`, `optax`,
`numpy`, `matplotlib`). Outputs go to `./outputs/` **relative to your current directory**, so run
from wherever you want them:

```bash
conda activate <env>
pip install -e ~/qvarnet --no-deps          # one-time; makes `import qvarnet` work
#   (or, without installing:  export PYTHONPATH=~/qvarnet/src)
export MPLBACKEND=Agg                        # headless plotting

cd ~/dilute-bose
python ~/qvarnet/soft_sphere_gas/run_workers.py --gpus 0,1 --seeds 0 1 2
#   -> writes ~/dilute-bose/outputs/{soft_sphere.db, runs/}
```

Running the script *by path* puts its own directory on `sys.path`, so the sibling imports
(`db`, `point`, …) resolve while outputs stay relative to your CWD. Everything below shows the
local `uv` form; on the cluster drop `uv run --project ..` and just use `python`.

### Local (uv)

Everything runs through `uv` from the qvarnet project (the venv lives one level up):

```bash
cd qvarnet/soft_sphere_gas

# Full SS10 E/N(x) curve: N=64, seeds 0/1/2, 7-point x-grid, 2000 epochs, 1024 chains
uv run --project .. python run_ss10_curve.py

# Common overrides
uv run --project .. python run_ss10_curve.py --N 64 --seeds 0 --n-x 7 --epochs 2000 --chains 1024
uv run --project .. python run_ss10_curve.py --tune          # ASHA-tune HPs at an anchor x, then freeze
uv run --project .. python run_ss10_curve.py --db outputs/soft_sphere.db
```

**Resumable.** Re-running the same command skips `done` points and re-queues anything left
`running` (a killed process). Safe to Ctrl-C and restart. The `(potential, x, N, seed, hp)` tuple
is the key, so changing any of those produces a *new* run rather than overwriting.

**GPU memory (important).** On an 8 GB card (e.g. RTX 3070), N=64 OOMs at 2048 chains — the
default is **1024 chains** (~5.6 s/epoch). 512 chains (~2.8 s/epoch) also works and is noisier.
On a bigger cluster GPU you can raise `--chains` (and lower wall-clock). Rough budget at 1024
chains / 2000 epochs: **~3 hr per (x, seed) point**; a full 7×3 grid is ~60 GPU-hours.

### Multi-GPU (worker-per-GPU) — the fast path

The sweep is a set of *independent* points, so the efficient way to use several GPUs is one
**worker process per GPU**, all draining a single shared SQLite queue. `db.claim_next` hands each
worker a distinct `(potential, x, N, seed, hp)` atomically (`BEGIN IMMEDIATE`), so two GPUs never
run the same point, and it load-balances (a free GPU claims the next point immediately).

```bash
cd ~/qvarnet/soft_sphere_gas        # or wherever the project lives

# fill the queue + spawn one worker per detected GPU (Titan V + Titan Xp -> 2 workers)
python run_workers.py --N 64 --seeds 0 1 2 --epochs 2000 --chains 1024 \
       --db ~/dilute-bose/outputs/soft_sphere.db

# explicit GPUs and a finite-N ladder
python run_workers.py --gpus 0,1 --N 32 64 128 --seeds 0 1 --db ~/dilute-bose/outputs/soft_sphere.db
```

`run_workers.py` requeues any crashed `running` rows, enqueues the whole sweep (feasible x-grid
*per N*), then launches the workers with `CUDA_VISIBLE_DEVICES` pinned per GPU. It's fully
resumable: re-run the same command to extend (add seeds / N / x-points) — finished points are
skipped. To run a single GPU by hand: `CUDA_VISIBLE_DEVICES=0 python worker.py --db <path>`.

> **One worker per *distinct* GPU.** JAX preallocates most of a GPU's VRAM per process, so do not
> point two workers at the same device (you'll OOM) unless you also set
> `XLA_PYTHON_CLIENT_MEM_FRACTION` low. The Titans have 12 GB, so `--chains 1024` (and likely
> 2048) fits one worker comfortably.

> **Same node only (SQLite).** The shared-queue coordination relies on a local-filesystem SQLite
> DB. For multiple GPUs on one node this is ideal. Across nodes, give each node its own `--db` and
> merge afterwards (every row is keyed, so `INSERT OR IGNORE` merges cleanly). See *Multi-PC* below.

### Multi-PC (independent machines, merge afterwards)

SQLite coordinates within **one** machine only. To spread a sweep over several independent PCs
(no shared filesystem), run each one independently and merge at the end. Two rules:

1. **Identical HP on every PC** — same flags ⇒ same `hp_json` hash ⇒ the rows are the same physics
   points and merge (and the per-run dir names match the physics axes). *Only the partition flags
   and `--db`/`--gpus` may differ.*
2. **Partition the work so no two PCs run the same point** — easiest by **seed** (one seed per PC,
   each running the full `--N` ladder); after merging, every `(x, N, seed)` is present so the
   `1/N` extrapolation works unchanged. (Alternatively split by `--N` to put the heavy N=256 on the
   strongest GPU.)

First push the code from your dev machine, then on **each PC** pull + install (the engine changed —
`init_params` for warm-start — so an editable install is required):

```bash
cd ~/qvarnet && git pull
conda run -n jax pip install -e ~/qvarnet --no-deps
conda run -n jax pip install pandas          # used by analysis
```

Then the sweep — run inside `tmux` (an SSH drop kills the run otherwise). Identical except
`--seeds`, `--db`, `--gpus`:

```bash
# PC 1
tmux new -s sweep ; conda activate jax ; cd ~/soft-hard-bosons
python ~/qvarnet/soft_sphere_gas/run_workers.py \
  --gpus 0 --seeds 0 \
  --N 32 64 128 256 --n-x 10 \
  --jastrow --laplacian hutchinson --hutchinson-n-terms 32 \
  --lr 3e-3 --lr-schedule cosine \
  --chains 1024 --epochs 2000 --plateau-rel 0.005 --es-min-epochs 300 \
  --db outputs/pc1.db
# PC 2:  --seeds 1 ... --db outputs/pc2.db
# PC 3:  --seeds 2 ... --db outputs/pc3.db
```

`--gpus` is a device *index*, not a count (single-GPU box ⇒ `0`). If N=256 OOMs, lower `--chains`
on **all** PCs (keep HP identical). Validate on PC 1 first: let one point finish, then
`check_db.py --db outputs/pc1.db` and confirm `--jastrow` energies drop below Eq.31 toward
Lee-Yang before committing the other machines.

Gather + merge on one machine (`merge_db.py`: `INSERT OR IGNORE` for new keys + a done-precedence
pass so a finished result never loses to a leftover `todo`):

```bash
rsync -av user@pc1:~/soft-hard-bosons/outputs/ ./merge/pc1/
rsync -av user@pc2:~/soft-hard-bosons/outputs/ ./merge/pc2/
rsync -av user@pc3:~/soft-hard-bosons/outputs/ ./merge/pc3/

conda run -n jax python ~/qvarnet/soft_sphere_gas/merge_db.py \
  ./merge/merged.db  ./merge/pc1/pc1.db ./merge/pc2/pc2.db ./merge/pc3/pc3.db

# union the per-run artifact dirs (names unique since partitioned by seed)
rsync -a ./merge/pc1/runs/ ./merge/runs/
rsync -a ./merge/pc2/runs/ ./merge/runs/
rsync -a ./merge/pc3/runs/ ./merge/runs/

conda run -n jax python ~/qvarnet/soft_sphere_gas/check_db.py --db ./merge/merged.db
```

WAL note: rsync the whole tree (it grabs the `-wal` sidecar with the newest writes), or
`PRAGMA wal_checkpoint(TRUNCATE)` each source first.

### Early stopping (don't waste epochs)

`n_epochs` is a **ceiling**, not a fixed length: by default each run stops once it looks converged,
via `EarlyStopCallback`. Two triggers (a run stops when either holds for `es_patience` checks in a
row, after `es_min_epochs`):

- **verdict** (default): the three-referee convergence verdict passes (stationary + at MC floor +
  chains mixed). Principled and never stops while the energy is still meaningfully improving — but
  for a slowly-improving NN ansatz it can wait a long time.
- **plateau** (opt-in, `es_plateau_rel > 0`): the tail-mean energy improved by less than
  `es_plateau_rel` (relative) over the last `es_check_every` epochs — i.e. the optimiser has
  effectively stopped lowering the energy. More aggressive; saves the most GPU time.

`HP` knobs: `early_stop` (default `True`), `es_min_epochs` (200), `es_check_every` (50),
`es_patience` (2), `es_target_rel_err` (0 = off; require `err/|E|` below it), `es_plateau_rel`
(0 = off). The run's `verdict.json` records `epochs_ran`, `early_stopped_at`, `early_stop_reason`.
Set `early_stop=False` for a fixed-length run.

**A single point** (no DB, for a quick check):

```bash
uv run --project .. python -c "from point import SS10, HP, run_point; \
print(run_point(SS10, x=1e-4, N=64, seed=0, hp=HP(n_epochs=2000, n_chains=1024)))"
```

---

## 2. What is computed

For each `(potential, x, N, seed, hp)` the pipeline trains a neural-network wavefunction by VMC
and records the converged energy plus diagnostics. The unit of work is `point.run_point`.

- **potential** — a soft (penetrable) sphere at scattering length `a = 1`; only `R` is free
  (`V0` is fixed by Eq.10). `SS10` = `R=10`, `SS5` = `R=5`. (`R=1` is the hard-sphere limit.)
- **x** — the gas parameter `ρa³`. The box side is `L = (N/x)^(1/3)`. Box-feasibility requires
  `R < L/2` ⇔ `x < N/(8R³)`; infeasible points are recorded as `skipped_box`, not run.
- **N** — particle number. **seed** — RNG. **hp** — the solver knobs (`HP`): model widths,
  learning rate, epochs, chains, sampler settings, and the snapshot policy.
- **model** — a permutation-invariant `DeepSet` inside a `LogWavefunction` with periodic
  (sin/cos) boundary features. No Jastrow yet (that gates the hard-sphere end; roadmap step 5).

All energies are reported in **paper units** (`ℏ²/2m=1`, `a=1`, energies in `ℏ²/2ma²`). The
engine runs `ℏ=m=1`; the single factor-of-two bridge lives only in `dilute_gas.engine_V0`
(V0/2 in) and `dilute_gas.to_paper_energy` (×2 out). See `CONVENTIONS.md`.

**Parameter retention.** During training we keep the **best `snapshot_frac` of epochs** (default
**10%**) ranked by the selection metric `select` (default `"std"` = lowest σ_E) — these param
snapshots are what later analysis (g(r), S(k), n(k)/n₀) loads. The final-epoch state is also kept.

**Verdict.** Each run gets the engine's three-referee convergence verdict
(`result.diagnose()`): stationary? at the MC floor? chains mixed? `passed` is the AND of the
three. Curve aggregation only uses `passed` runs by default.

---

## 3. Output architecture & labeling

Everything lives under one self-contained, copyable tree:

```
outputs/
├── soft_sphere.db                         # SQLite index: one row per run (scalars + pointer)
└── runs/
    └── SS10_x1.000e-04_N64_s0_3f9a1c2b/   # one run dir, name = run_id (see below)
        ├── meta.json            # identity, units, full HP, box L, paper benchmarks at this x
        ├── history.csv          # per-epoch metrics (engine + paper-unit columns)
        ├── verdict.json         # full three-referee diagnose() dict
        ├── best_params.msgpack  # best-10% param snapshots by σ_E (best first)
        └── checkpoints/
            └── final_state.msgpack   # engine's final-epoch state (auxiliary)
```

**`run_id` = `<potential>_x<gas-param>_N<particles>_s<seed>_<hp8>`**
e.g. `SS10_x1.000e-04_N64_s0_3f9a1c2b`. The physics axes are human-readable; `hp8` is the first
8 hex of a SHA-1 of the *full* HP dict, so two runs differing only in solver settings get distinct
dirs. The full HP is in both `meta.json` and the DB `hp_json` column — the hash is never inverted.

### DB schema (`runs` table)

| column | meaning |
|---|---|
| `potential_label, R, V0_paper` | the potential |
| `x, N, seed` | physics point + RNG |
| `hp_json` | full hyperparameters (JSON; part of the unique key) |
| `status` | `todo` → `running` → `done` \| `failed` \| `skipped_box` |
| `e_per_n, err_per_n, sigma_e_per_n` | E/N, error of mean, per-sample spread — **paper units** |
| `acceptance, passed` | mean MH acceptance; three-referee verdict (0/1) |
| `verdict_json` | full diagnose() dict |
| `L, upper_bound` | box side; Eq.31 first-order upper bound (paper units) |
| `run_dir` | path (relative to `outputs/`) of this run's artifact dir |
| `error, started_at, finished_at` | bookkeeping |

### `history.csv` columns (per epoch)

`epoch, energy_engine, std_engine, error_of_mean_engine, e_per_n_paper, sigma_per_n_paper,
acceptance, step_size, cm_mean, cm_std, wall_time`. Energy/std are stored both in raw **engine**
units and as **paper-unit per-particle** columns (`e_per_n_paper = 2·energy_engine / N`). Use the
`*_paper` columns for physics; the engine columns are there for debugging the bridge.

---

## 4. Reading / extracting results

### Sanity-check the physics (`check_db.py`)

The fastest way to confirm a run is sound — reads the DB and checks every done point against the
rigorous corridor (`4πx ≤ E/N ≤ Eq.31`) and Lee–Yang, prints the queue status, any failures, and
the seed-aggregated curve:

```bash
python check_db.py --db outputs/soft_sphere.db
```

Each row shows `E/N` (per-particle paper units, straight from the DB — no progress-bar guessing),
the bounds, `E/LY`, a `corridor` ok/VIOLATION flag, the verdict, and `epochs_ran`. "all done
points inside [4πx, Eq31]" = the physics is correct.

### The curve (seed-averaged, verdict-gated)

```python
import db
from sweep import load_curve
from point import SS10
from analysis import plot_curve

conn = db.connect("outputs/soft_sphere.db")
curve = load_curve(conn, "SS10", N=64)        # dict: x, e_per_n, err, upper_bound, n_per_x
plot_curve(curve, SS10, save="outputs/ss10_curve_N64.png")   # Fig.1/Fig.12-style E/N / 4πx vs x
```

`load_curve` combines seeds per x; the error is the larger of the across-seed SEM and the mean
within-seed error (never understated).

### Everything, as a DataFrame

```python
import sqlite3, pandas as pd
df = pd.read_sql("SELECT * FROM runs", sqlite3.connect("outputs/soft_sphere.db"))
df[df.status == "done"][["x", "N", "seed", "e_per_n", "err_per_n", "passed"]]
```

### Quick SQL

```bash
sqlite3 outputs/soft_sphere.db \
  "SELECT x, seed, e_per_n, err_per_n, passed FROM runs WHERE status='done' ORDER BY x, seed"
sqlite3 outputs/soft_sphere.db \
  "SELECT potential_label, x, N, seed, error FROM runs WHERE status='failed'"   # diagnose failures
```

### Per-epoch convergence (one run)

```python
import pandas as pd
h = pd.read_csv("outputs/runs/SS10_x1.000e-04_N64_s0_3f9a1c2b/history.csv")
h.plot(x="epoch", y="e_per_n_paper")   # convergence in paper units
```

### The best-10% parameters (for observables / re-evaluation)

```python
import artifacts
blob = artifacts.load_params("outputs/runs/<run_id>/best_params.msgpack")
# blob = {"select", "steps", "metrics", "params": [pytree, ...]}  (best first, numpy arrays)
best = blob["params"][0]          # already a full Flax variables dict: {"params": {...}}
# rebuild the same model (see point._build_model with the run's HP + L) and apply:
#   logpsi = model.apply(best, x_batch)
# average an observable over blob["params"] for a snapshot-ensemble estimate.
```

---

## 5. Moving outputs between a cluster and a PC

The whole `outputs/` tree is self-contained and host-independent (no absolute paths; `run_dir`
is stored relative to `outputs/`; params are plain msgpack/numpy). To move it:

```bash
# On the cluster: flush SQLite's write-ahead log into the main .db file first
sqlite3 outputs/soft_sphere.db "PRAGMA wal_checkpoint(TRUNCATE);"

# Then copy the whole tree (rsync handles the run dirs + DB in one shot)
rsync -av cluster:/path/to/qvarnet/soft_sphere_gas/outputs/  ./outputs/
```

If you copy the `.db` while a run is live, also copy `soft_sphere.db-wal` and `soft_sphere.db-shm`
(or run the `wal_checkpoint` above first — simplest). Once on the PC, all of §4 works unchanged
as long as you run from `soft_sphere_gas/` (so the relative `run_dir` paths resolve), or pass an
absolute `--db`/path.

Merging two machines' results: since each row is keyed by `(potential, x, N, seed, hp)`, you can
re-point a sweep at a copied DB and it will only fill in the missing points.
```
