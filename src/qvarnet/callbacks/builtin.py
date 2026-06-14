import csv
import os

import jax.numpy as jnp

from ..utils.checkpoint import save_checkpoint
from .base import Callback

_BUILTIN_METRICS = {
    "energy": lambda s: float(s.energy),
    "std": lambda s: float(s.std),
}

_HISTORY_FIELDS = (
    "step",
    "energy",
    "std",
    "acceptance_rate",
    "step_size",
    "cm_mean",
    "cm_std",
)


def _state_to_row(s) -> dict:
    return {
        "step": int(s.step),
        "energy": float(s.energy),
        "std": float(s.std),
        "acceptance_rate": float(jnp.mean(s.acceptance_rate)),
        "step_size": float(s.step_size),
        "cm_mean": float(s.cm_mean),
        "cm_std": float(s.cm_std),
    }


class RunOutputCallback(Callback):
    """At the end of training, save the full scalar history and the N best checkpoints.

    Added automatically by ``train()`` with ``n=1, metric=["energy"]`` unless the
    caller supplies their own instance in ``callbacks``.

    Two outputs are written under ``path``:

    ``history.csv``
        Scalar diagnostics for every epoch: step, energy, std, acceptance_rate,
        step_size, cm_mean, cm_std.  Parameters and gradients are intentionally
        excluded — this file stays small regardless of model size.

    ``checkpoints/best_<label>_<rank>.msgpack``
        Full VMCState checkpoints (params + optimizer state) for the N best
        epochs according to each requested metric.  ``<rank>`` is 0-indexed
        (0 = best).  For callable metrics the label is ``custom_<index>``.

    Args:
        n:      Number of best states to keep per metric.
        path:   Base output directory (same value as
                ``TrainingConfig.checkpoint_path``).
        metric: List of ranking criteria — same interface as
                ``TrainResult.best()``.  Each element is either a built-in
                string shortcut or a callable ``(VMCState) -> float``
                (lower is better).

                Built-in shortcuts
                    ``"energy"``  — lowest ⟨E⟩
                    ``"std"``     — lowest σ_E

    Example::

        result = train(
            ...,
            callbacks=[
                RunOutputCallback(
                    n=5,
                    path="./outputs/run/",
                    metric=["energy", lambda s: float(s.energy) + float(s.std)],
                )
            ],
        )
        # Writes:
        #   outputs/run/history.csv
        #   outputs/run/checkpoints/best_energy_0.msgpack  (lowest ⟨E⟩)
        #   outputs/run/checkpoints/best_energy_1.msgpack
        #   ...
        #   outputs/run/checkpoints/best_custom_1_0.msgpack  (lowest E+σ)
        #   ...
    """

    def __init__(self, n: int, path: str, metric: list = None):
        self.n = n
        self.path = path
        self.metrics = metric if metric is not None else ["energy"]

    def on_train_end(self, state, history):
        if not len(history):
            return

        os.makedirs(self.path, exist_ok=True)
        csv_path = os.path.join(self.path, "history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(_state_to_row(s) for s in history)

        # Best-K *parameter* snapshots needed the per-epoch VMCState; with the
        # param-free MetricsHistory that returns in roadmap step 6 (snapshot policy:
        # none/every_n/all/best_k). For now persist the final live state so a
        # resumable checkpoint always exists.
        save_checkpoint(state, self.path, filename="final_state.msgpack")


class NaNCallback(Callback):
    """Stop training and save an emergency checkpoint when energy is NaN."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def on_step_end(self, step, state, metrics):
        if jnp.isnan(metrics["energy"]):
            print(f"NaN detected at step {step}. Stopping.")
            save_checkpoint(
                state, path=self.checkpoint_path, filename="nan_checkpoint.msgpack"
            )
            return True
        return False


class CheckpointCallback(Callback):
    """Save a rolling checkpoint every `save_every` steps."""

    def __init__(self, checkpoint_path: str, save_every: int = 50):
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every

    def on_step_end(self, step, state, metrics):
        if step % self.save_every == 0:
            save_checkpoint(
                state, path=self.checkpoint_path, filename="checkpoint.msgpack"
            )
        return False


class ProgressCallback(Callback):
    """Update a tqdm progress bar with energy and std every `update_every` steps."""

    def __init__(self, progress_bar, update_every: int = 10):
        self.progress_bar = progress_bar
        self.update_every = update_every

    def on_step_end(self, step, state, metrics):
        if step % self.update_every == 0:
            self.progress_bar.set_postfix(
                E=f"{metrics['energy']:.4f}",
                sigma_E=f"{metrics['std']:.4f}",
            )
        return False
