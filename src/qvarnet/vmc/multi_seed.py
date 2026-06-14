"""Multi-seed seed-safety protocol (roadmap §5.2).

Run ``train`` over m ≥ 4 independent seeds (sequential — vmapping a full training run isn't
worth it on one GPU), take each run's tail window, and compute Gelman-Rubin R̂ across the
tails. **Pass: R̂ ≤ 1.1.** This detects optimisation-landscape variance — the more dangerous
failure mode for NQS — which a single run cannot see.
"""

import dataclasses

import numpy as np

from ..diagnostics import split_rhat
from .train import train


def multi_seed_run(seeds, *, tail_frac: float = 0.5, rhat_threshold: float = 1.1, **train_kwargs):
    """Train once per seed, then R̂ over the tail energy windows.

    ``train_kwargs`` are forwarded to ``train``; its ``training_config.rng_seed`` is overridden
    per seed. Returns ``{results, tail_means, rhat, seed_safe}``.
    """
    if "training_config" not in train_kwargs:
        raise ValueError("multi_seed_run requires a training_config in train_kwargs")
    base_cfg = train_kwargs.pop("training_config")

    results = []
    tails = []
    for seed in seeds:
        cfg = dataclasses.replace(base_cfg, rng_seed=int(seed))
        result = train(training_config=cfg, **train_kwargs)
        results.append(result)
        energy = result.history.get("energy")
        k = max(int(tail_frac * len(energy)), 2)
        tails.append(np.asarray(energy[-k:]))

    # Trim to common length so the tails form a clean (m, n) array for R̂.
    min_len = min(t.shape[0] for t in tails)
    tail_matrix = np.stack([t[-min_len:] for t in tails], axis=0)  # (m, min_len)
    rhat = split_rhat(tail_matrix)
    return {
        "results": results,
        "tail_means": [float(t.mean()) for t in tails],
        "rhat": rhat,
        "seed_safe": bool(rhat <= rhat_threshold),
    }
