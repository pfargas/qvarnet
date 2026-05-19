class AuxiliaryLoss:
    """Pluggable loss term added on top of the VMC energy loss.

    __call__ must be JAX-traceable: use jnp operations only, no Python
    control flow on array values.

    Args:
        params: model parameters (pytree).
        model_apply: model's apply function — callable(params, x) -> log|ψ|.
        batch: current MCMC batch, shape (n_chains, dof).

    Returns:
        Scalar JAX float to be added to the VMC loss.
    """

    def __call__(self, params, model_apply, batch):
        raise NotImplementedError
