from flax.training import train_state


class VMCState(train_state.TrainState):
    """Live VMC training state — Flax ``TrainState`` (params, tx, opt_state, step).

    Per-epoch diagnostics (energy, std, acceptance_rate, step_size, grads, cm_*) used
    to live here, which meant ``state_history`` retained a full copy of params +
    gradients + optimizer state every epoch. Those metrics now live in
    ``MetricsHistory``; this object is purely the optimiser's live state, passed to
    callbacks for checkpointing / model selection.
    """
