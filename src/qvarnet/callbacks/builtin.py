import csv
import os

import jax.numpy as jnp

from qvarnet.callbacks.base import Callback
from qvarnet.core.serialization import save_checkpoint

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
    """Write the scalar history and the N best checkpoints when training ends.

    Added automatically by VMC with ``n=1, metric=["energy"]`` unless you pass your
    own instance. Writes ``history.csv`` (per-epoch scalars only, so it stays small
    whatever the model size) and ``checkpoints/best_<label>_<rank>.msgpack``
    (0-indexed, 0 = best).

    Args:
        n: how many best states to keep per metric.
        path: base output directory, usually TrainingConfig.checkpoint_path.
        metric: ranking criteria, same interface as ``TrainResult.best()`` --
            "energy", "std", or a callable ``(VMCState) -> float`` (lower is
            better), which is labelled ``custom_<index>``.
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
            save_checkpoint(state, path=self.checkpoint_path, filename="nan_checkpoint.msgpack")
            return True
        return False


class CheckpointCallback(Callback):
    """Save a rolling checkpoint every `save_every` steps."""

    def __init__(self, checkpoint_path: str, save_every: int = 50):
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every

    def on_step_end(self, step, state, metrics):
        if step % self.save_every == 0:
            save_checkpoint(state, path=self.checkpoint_path, filename="checkpoint.msgpack")
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
