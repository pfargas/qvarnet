class Callback:
    """Base class for training-loop hooks.

    on_step_end is called after every training step, outside of JIT.
    Return True to stop training early.

    metrics keys: "energy", "std", "acceptance_rate", "step_size", "cm_mean", "cm_std"
    state is the full VMCState including current params (use it for checkpointing or
    model selection).
    """

    def on_step_end(self, step: int, state, metrics: dict) -> bool:
        return False

    def on_train_end(self, state, history: list) -> None:
        pass
