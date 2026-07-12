#!/bin/bash
#SBATCH --job-name=cs-sweep
#SBATCH --gres=gpu:1            # one worker per GPU (JAX preallocates VRAM)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#
# CS grid sweep on SLURM. Edit the AXES line, then: sbatch sweep.sbatch
# Resumable: resubmitting the same job continues the same DB (interrupted rows
# are requeued; finished points are skipped). Partition across nodes by seed:
#   sbatch --export=ALL,SEEDS="0 1" sweep.sbatch
#   sbatch --export=ALL,SEEDS="2 3" sweep.sbatch     (then: runq merge)

set -euo pipefail

# ── site setup: adapt to the cluster's module system ────────────────────────
# module load cuda/12 python/3.12
# source ~/venvs/qvarnet/bin/activate      # venv with: pip install -e ~/qvarnet -e ~/runq

cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2}"
AXES=(
    --axis L=0.5,0.8,1.0,1.5,2.0
    --axis N=10,30,60
    --axis kind=mlp_jastrow
    --axis proposal=gaussian,particle-subset
    --axis n_epochs=20000
)

# DB on the node-local scratch when available (fast SQLite WAL), rsync back after.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    OUT="$SLURM_TMPDIR/outputs"
    mkdir -p "$OUT"
    [[ -f outputs/runq.db ]] && cp outputs/runq.db "$OUT/"    # resume a previous DB
else
    OUT="outputs"
fi

runq run point.py --db "$OUT/runq.db" "${AXES[@]}" --seeds $SEEDS

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    sqlite3 "$OUT/runq.db" "PRAGMA wal_checkpoint(TRUNCATE);"
    mkdir -p outputs
    rsync -a "$OUT/" outputs/
fi
