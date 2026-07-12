# Running the Calogero-Sutherland grid sweeps & reading the outputs

Resumable, grid-agnostic sweeps of the CS model, run on **runq** (the sibling
`~/personal-os/phd/runq` package; `uv sync` here installs it, on a cluster
`pip install -e ~/runq`). The project-side code is just:

* `point.py` — the physics. The runq target is the flat `run_point(...)` at the bottom
  (one plain function, keyword defaults, one flat namespace); internally it builds
  `Physics` + `HyperParams` and calls `train_point`.
* `artifacts.py` — what a run leaves in its run dir (meta/history/verdict/params).
* `check_db.py` — physics sanity check of a sweep DB.

**Code convention** (ℏ²/m = 1, ω = 1, as in `calogero_sutherland.ipynb`):
`H = -Σ ∂² + Σ x² + 2L(L-1) Σ 1/(xᵢ-xⱼ)²`, exact ground state `E₀ = N(1 + L(N-1))`.

---

## 1. How to run

One flat `--axis` vocabulary — physics axes (`L`, `N`, `epsilon`) and solver axes (`kind`,
`lr`, `n_epochs`, …) are passed identically; anything not swept keeps the default from
`run_point`'s signature (which also fixes the type CLI values are coerced to).

```bash
cd calogero-sutherland/cs_sweep

# grid over L and N, two model kinds, seeds 0..2, one worker per GPU
runq run point.py --axis L=0.5,0.8,1.0,1.5,2.0 --axis N=2,5,10 \
    --axis kind=jastrow,mlp_jastrow --axis n_epochs=2000 --seeds 0 1 2 --gpus 0,1

# sampler axes: subset proposals keep acceptance high at large N (~2× effective
# samples per model eval at N=30 moving one particle; docs/SAMPLERS.md).
# proposal_ratio is a FRACTION resolved per point (n_move = round(ratio·N)), so one
# axis scales across a mixed-N grid; ratio=1.0 moves all particles ≡ gaussian, so
# the baseline needs no separate proposal value:
runq run point.py --axis L=0.8 --axis N=10,30,60 \
    --axis proposal=particle-subset \
    --axis proposal_ratio=0.1,0.3,0.5,0.7,1.0 --seeds 0 1 2

runq status                 # queue counts
runq failed                 # tracebacks of failed points
runq requeue [--failed]     # reset interrupted (and optionally failed) rows to todo
```

**Fully resumable**: re-run the same command to extend (add seeds / axes) — finished points
are skipped; interrupted (`running`) rows are requeued automatically. The DB key is the full
resolved parameter dict, so changing *any* value produces a new run rather than overwriting.

With >1 worker each gets its own log (`outputs/logs/worker_gpu<N>.log`, `tail -f` to watch).
One worker per *distinct* GPU (JAX preallocates VRAM). Single device by hand:
`CUDA_VISIBLE_DEVICES=0 python -m runq.worker --db outputs/runq.db --target point.py`.

On SLURM use `sweep.sbatch` (this directory): edit its `AXES` array and `sbatch sweep.sbatch`.
It puts the DB on `$SLURM_TMPDIR` when available and rsyncs back; partition across nodes with
`--export=ALL,SEEDS="0 1"` and merge afterwards. Note runq is no longer a qvarnet dependency —
install both editable in the cluster venv: `pip install -e ~/qvarnet -e ~/runq` (locally:
`uv pip install -e ../runq`, kept out of pyproject deliberately).

### From Python / a notebook

```python
from runq import ParamSpace, build_grid, connect, key_json, run_label, store
from point import run_point

space = ParamSpace.from_function(run_point)
axes = {"L": [0.5, 0.8, 1.0], "N": [2, 5, 10], "kind": ["jastrow"], "seed": [0, 1, 2]}
conn = connect("outputs/runq.db")
for params in build_grid(space, axes):
    store.enqueue(conn, key_json(params), run_label(params, list(axes)))
# drain from the shell: runq run point.py --db outputs/runq.db
```

### A single point (no DB, quick check)

```python
from point import run_point
r = run_point(L=0.8, N=5, kind="jastrow", n_epochs=2000, n_chains=2048, seed=0)
print(r["e_total"], "+/-", r["err_total"], " exact", r["e_exact"], " gap", r["gap"])
```

### Early stopping

`--axis early_stop=true` makes `n_epochs` a ceiling: the run stops once the three-referee
verdict (and optionally a plateau / relative-error gate) passes `es_patience` checks in a row
after `es_min_epochs`. Knobs: `es_check_every`, `es_min_epochs`, `es_patience`,
`es_target_rel_err`, `es_plateau_rel`. Default `early_stop=false` = fixed-length run.

---

## 2. Output architecture

```
outputs/
├── runq.db                            # SQLite index: one row per run
└── runs/
    └── L0.8_N5_seed0_3f9a1c2b/       # label = swept axes + hash8 of the full params
        ├── run.json                   # runq's record: params + returned metrics
        ├── meta.json                  # physics, hyper-params, exact energy, result
        ├── history.csv                # per-epoch: energy, std, error_of_mean, acceptance, …
        ├── verdict.json               # full three-referee diagnose() dict
        ├── best_params.msgpack        # best n_snapshots epochs by `select` (best first)
        └── checkpoints/final_state.msgpack
```

The whole `outputs/` tree is self-contained and copyable between machines (no absolute
paths; `run_dir` stored relative to the DB's directory).

### DB schema (`runs`)

Generic runq schema — parameters and results are JSON, so any axis works with no schema
change: `params_json (UNIQUE), label, status, result_json, run_dir, error, started_at,
finished_at`. Status: `todo → running → done | failed | skipped`. Query an axis with SQLite
JSON1 (`json_extract(params_json,'$.L')`) — or just use `load_table` (below).

---

## 3. Reading / extracting results

### Sanity-check (`check_db.py`)

```bash
python check_db.py --db outputs/runq.db
```
Prints the queue status and every done point's `E/N`, exact `E/N`, the variational **gap**
(should be ≥ 0 within a few σ — the variational principle), verdict, and epochs. A gap well
below 0 flags a bias (e.g. the `epsilon`-softened potential), not a better-than-exact result.

### Everything, as a DataFrame (grid-agnostic)

```python
from runq import connect, load_table
df = load_table(connect("outputs/runq.db"))    # params + result metrics as columns
df[["L", "N", "seed", "kind", "e_per_n", "err_per_n", "gap", "passed"]]

# seed-averaged E/N vs L at fixed N — plain pandas, no harness API
sel = df[(df.N == 5) & (df.passed == 1)]
curve = sel.groupby("L")["e_per_n"].agg(["mean", "sem"])
```

### Per-epoch convergence / best params (one run)

The `history.csv` header is a `#` comment line, so gnuplot reads it directly:

```gnuplot
set datafile separator ','
plot 'outputs/runs/<label>/history.csv' using 1:2 with lines title 'E'   # epoch:energy
```

```python
import pandas as pd, artifacts
h = pd.read_csv("outputs/runs/<label>/history.csv")
h.columns = h.columns.str.lstrip("# ")                     # strip the gnuplot comment marker
h.plot(x="epoch", y="energy")
blob = artifacts.load_params("outputs/runs/<label>/best_params.msgpack")
best = blob["params"][0]    # full Flax variables dict; rebuild model via point._build_model + apply
```

---

## 4. Multi-machine merge

Each row is keyed by its full parameter dict, so spread a sweep over independent machines
(partition by seed), then gather and merge:

```bash
runq merge merged.db pc1.db pc2.db pc3.db      # done-precedence upsert
rsync -a pc1/outputs/runs/ merged/runs/        # union the run dirs (labels are unique)
```

WAL note: before copying a live DB, flush it —
`sqlite3 outputs/runq.db "PRAGMA wal_checkpoint(TRUNCATE);"`.
