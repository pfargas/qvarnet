class Callback:
    """Base class for training-loop hooks.

    on_step_end is called after every training step, outside of JIT.
    Return True to stop training early.

    metrics keys: "energy", "std", "acceptance_rate", "step_size", "cm_mean", "cm_std",
    plus any on-device extras the loop adds in ``cb_metrics`` (vmc/train.py) — currently
    "grads", the raw energy-gradient pytree (pre-QGT). Device-backed values are live JAX
    arrays: ``jax.device_get`` them if you keep them, and never retain them per-epoch
    (that pins device memory; see EarlyStopCallback for the filtering idiom).
    state is the full VMCState including current params (use it for checkpointing or
    model selection).
    """

    def on_step_end(self, step: int, state, metrics: dict) -> bool:
        return False

    def on_train_end(self, state, history: list) -> None:
        pass
