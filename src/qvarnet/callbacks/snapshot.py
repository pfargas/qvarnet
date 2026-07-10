"""Parameter snapshot policy (roadmap §2.1) — explicit, not accidental.

Saving params during training is a *policy*, configured here rather than a side effect of
logging energies. Snapshots are ``jax.device_get``-ed to host RAM (or could be appended to
disk), so even ``"all"`` is affordable — the slim ``VMCState`` keeps VRAM free.

Policies:
    "none"     keep nothing (default for long production runs)
    "every_n"  keep params every ``every_n`` epochs (animations, post-hoc analysis)
    "all"      keep every epoch (short diagnostic runs)
    "best_k"   keep the ``k`` best epochs by ``metric`` (model selection) — this restores the
               full-parameter ``best()`` retrieval that the param-free MetricsHistory dropped.

``metric`` is the selection key (lower = better). It may be a **string** shortcut —
``"std"`` (default; minimise σ_E), ``"energy"``, ``"e_plus_sigma"`` (Ē + σ_E), or any raw
key present in the per-epoch ``metrics`` dict — or a **callable** ``(metrics_dict) -> float``
(e.g. a V-score once an E_∞ is known, or σ/|E|).
"""

import jax

from .base import Callback


def resolve_metric_fn(metric):
    """Turn a string shortcut or callable into a key ``(metrics_dict) -> float`` (lower=better)."""
    if callable(metric):
        return metric
    if metric == "e_plus_sigma":
        return lambda m: float(m["energy"]) + float(m["std"])
    # "std", "energy", or any raw metrics key
    return lambda m: float(m[metric])


class SnapshotCallback(Callback):
    def __init__(self, policy: str = "best_k", every_n: int = 50, k: int = 3, metric="std"):
        if policy not in ("none", "every_n", "all", "best_k"):
            raise ValueError(f"unknown snapshot policy {policy!r}")
        self.policy = policy
        self.every_n = every_n
        self.k = k
        self.metric = metric
        self._key = resolve_metric_fn(metric)
        self.snapshots: list[dict] = []  # each: {step, metric, params, **cb_metrics}

    @staticmethod
    def _snap(value, state, metrics):
        """Host copy of params plus *every* key the loop exposes to callbacks.

        No key is hardcoded here: whatever ``cb_metrics`` in ``vmc/train.py`` contains
        ("step", scalars, "grads", any future on-device extra) is ``device_get``-ed into
        the snapshot. To retain something new, add it to ``cb_metrics`` — nothing else.
        Only two keys are added on top, because they exist nowhere in ``metrics``:
        "metric" (the resolved selection value used for ranking) and "params".
        """
        return {
            **jax.device_get(metrics),
            "metric": value,
            "params": jax.device_get(state.params),
        }

    def on_step_end(self, step, state, metrics) -> bool:
        if self.policy == "none":
            return False
        if self.policy == "every_n" and step % self.every_n != 0:
            return False
        if self.policy in ("all", "every_n"):
            self.snapshots.append(self._snap(self._key(metrics), state, metrics))
        elif self.policy == "best_k":
            value = self._key(metrics)
            # keep only if it could be among the k smallest (lower = better)
            if len(self.snapshots) < self.k or value < self.snapshots[-1]["metric"]:
                self.snapshots.append(self._snap(value, state, metrics))
                self.snapshots.sort(key=lambda s: s["metric"])
                del self.snapshots[self.k :]
        return False

    def best_params(self):
        """Parameters of the best retained snapshot (lowest ``metric``), or None."""
        if not self.snapshots:
            return None
        return min(self.snapshots, key=lambda s: s["metric"])["params"]

    def best_snapshot(self):
        """Full best snapshot dict (params, grads, all metrics), or None."""
        if not self.snapshots:
            return None
        return min(self.snapshots, key=lambda s: s["metric"])
