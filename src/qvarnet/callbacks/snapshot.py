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
"""

import jax

from .base import Callback


class SnapshotCallback(Callback):
    def __init__(self, policy: str = "best_k", every_n: int = 50, k: int = 3, metric: str = "energy"):
        if policy not in ("none", "every_n", "all", "best_k"):
            raise ValueError(f"unknown snapshot policy {policy!r}")
        self.policy = policy
        self.every_n = every_n
        self.k = k
        self.metric = metric
        self.snapshots: list[dict] = []  # each: {step, metric, params}

    def on_step_end(self, step, state, metrics) -> bool:
        if self.policy == "none":
            return False
        if self.policy == "every_n" and step % self.every_n != 0:
            return False
        if self.policy in ("all", "every_n"):
            self.snapshots.append(
                {"step": step, "metric": metrics.get(self.metric), "params": jax.device_get(state.params)}
            )
        elif self.policy == "best_k":
            value = float(metrics[self.metric])
            # keep only if it could be among the k smallest (lower = better)
            if len(self.snapshots) < self.k or value < self.snapshots[-1]["metric"]:
                self.snapshots.append(
                    {"step": step, "metric": value, "params": jax.device_get(state.params)}
                )
                self.snapshots.sort(key=lambda s: s["metric"])
                del self.snapshots[self.k :]
        return False

    def best_params(self):
        """Parameters of the best retained snapshot (lowest ``metric``), or None."""
        if not self.snapshots:
            return None
        return min(self.snapshots, key=lambda s: s["metric"])["params"]
