import jax.numpy as jnp

from ..utils.checkpoint import save_checkpoint
from .base import Callback


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
