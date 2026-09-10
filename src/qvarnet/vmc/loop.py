"""The epoch loop: run the update, log one row, let the callbacks look.

Two performance invariants live here, and both are easy to break by accident:

* **One host sync per epoch.** Every scalar the loop needs is fetched in a single
  ``jax.device_get``. Adding a stray ``float(...)`` inside the loop adds a
  device->host round trip to every epoch.
* **The step size stays on device** and is threaded straight back into the next
  update; only a host copy is logged.

Optimisation diagnostics are computed here rather than inside the jitted update, so
the update's graph -- and therefore the energy trace -- is unaffected by logging.
"""

import signal
import time

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..callbacks import ProgressCallback
from .train_result import TrainResult

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")


class _StopOnSigint:
    """Ctrl-C finishes the epoch in flight, then stops. Restores the old handler."""

    def __init__(self):
        self.requested = False
        self._previous = None

    def __enter__(self):
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, signum, frame):
        self.requested = True
        print("\nSignal received, stopping after current step.")

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._previous)
        return False


def _optimisation_diagnostics(state, new_state, grads):
    """Gradient norm and relative parameter step. Logging only, never fed back."""
    grad_norm = optax.tree.norm(grads)
    old_flat = ravel_pytree(state.params)[0]  # state is still pre-update here
    new_flat = ravel_pytree(new_state.params)[0]
    theta_ratio = jnp.linalg.norm(new_flat - old_flat) / (jnp.linalg.norm(old_flat) + 1e-8)
    return grad_norm, theta_ratio


def run_loop(ctx, update_fn) -> TrainResult:
    """Run the training epochs and assemble the TrainResult."""
    progress_bar = tqdm(
        range(int(ctx.state.step), ctx.training_config.n_epochs),
        disable=not tqdm_available,
    )
    if tqdm_available:
        ctx.callbacks.append(ProgressCallback(progress_bar))

    try:
        with _StopOnSigint() as stopper:
            for step in progress_bar:
                if stopper.requested:
                    break

                t0 = time.perf_counter()
                (
                    new_state,
                    ctx.key,
                    ctx.positions,
                    E,
                    sigma_e,
                    E_chain,
                    error_of_mean,
                    acceptance_rate,
                    ctx.step_size,  # stays on device, threaded into the next update
                    grads,
                    cm_mean,
                    cm_std,
                    sr_info,
                ) = update_fn(
                    state=ctx.state,
                    key=ctx.key,
                    current_pos=ctx.positions,
                    prob_fn=ctx.prob_fn,
                    step_size=ctx.step_size,
                    hamiltonian=ctx.hamiltonian,
                    sampling_config=ctx.sampling_config,
                    training_config=ctx.training_config,
                )

                grad_norm, theta_ratio = _optimisation_diagnostics(
                    ctx.state, new_state, grads
                )
                ctx.state = new_state

                # The one host sync of the epoch.
                (
                    E_v, sigma_e_v, error_of_mean_v, E_chain_v, acceptance_rate_v,
                    cm_mean_v, cm_std_v, step_size_v, grad_norm_v, theta_ratio_v, sr_info_v,
                ) = jax.device_get(
                    (
                        E, sigma_e, error_of_mean, E_chain, acceptance_rate,
                        cm_mean, cm_std, ctx.step_size, grad_norm, theta_ratio, sr_info,
                    )
                )

                metrics = {
                    "step": step,
                    "energy": float(E_v),
                    "std": float(sigma_e_v),
                    "error_of_mean": float(error_of_mean_v),
                    "E_chain": E_chain_v,
                    "acceptance_rate": acceptance_rate_v,
                    "step_size": float(step_size_v),
                    "grad_norm": float(grad_norm_v),
                    "theta_ratio": float(theta_ratio_v),
                    "cm_mean": float(cm_mean_v),
                    "cm_std": float(cm_std_v),
                    "wall_time": time.perf_counter() - t0,
                }
                # SR guard diagnostics, empty unless use_qgt: which constraint shaped
                # the step (trust_scale < 1 => Fisher trust region; nat_grad_norm above
                # qgt_config.grad_clip_norm => the Euclidean clip).
                metrics.update({k: float(v) for k, v in sr_info_v.items()})
                ctx.history.append(metrics)

                # Callbacks also see the on-device raw (pre-QGT) gradient pytree;
                # history itself stays scalar-only so nothing pins VRAM per epoch.
                cb_metrics = {**metrics, "grads": grads}
                if any(cb.on_step_end(step, ctx.state, cb_metrics) for cb in ctx.callbacks):
                    break
    finally:
        for cb in ctx.callbacks:
            cb.on_train_end(ctx.state, ctx.history)

    return TrainResult(
        history=ctx.history,
        # One host copy of the final params; the best-k come from the snapshot policy.
        final_params=jax.device_get(ctx.state.params),
        snapshots=ctx.snapshot_cb.snapshots if ctx.snapshot_cb is not None else [],
        # Final sampler state, so a rerun can resume the walkers and the adapted step.
        final_positions=jax.device_get(ctx.positions),
        final_step_size=float(jax.device_get(ctx.step_size)),
    )
