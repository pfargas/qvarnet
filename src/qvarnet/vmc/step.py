"""The jitted per-epoch update: sample, adapt, then one optimizer step.

This is the hot path. ``make_update_fn`` closes over the run's fixed pieces
(sampler, QGT config, auxiliary losses) and returns a single jitted function, so
there is exactly one compiled graph and one launch per epoch. The frozen configs
are jit-static, which is why changing a config field triggers a retrace.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .training_step import compute_step


@jax.jit
def adapt_step_size(
    step_size, acceptance_rate, min_step, max_step, target_acc, adaptation_rate
):
    """Nudge the MH step toward ``target_acc``, clipped to [min_step, max_step]."""
    factor = 1.0 + adaptation_rate * (jnp.mean(acceptance_rate) - target_acc)
    return jnp.clip(step_size * factor, min_step, max_step)


def make_update_fn(ctx):
    """Build the jitted epoch update for ``ctx``.

    Returned signature:
        ``(state, key, current_pos, prob_fn, step_size, hamiltonian,
           sampling_config, training_config) -> tuple``
    """
    sampler = ctx.sampler
    qgt_config = ctx.qgt_config
    auxiliary_losses = ctx.auxiliary_losses
    box_L = ctx.box_L

    @partial(
        jax.jit,
        static_argnames=["prob_fn", "hamiltonian", "sampling_config", "training_config"],
    )
    def full_update(
        state,
        key,
        current_pos,
        prob_fn,
        step_size,
        hamiltonian,
        sampling_config,
        training_config,
    ):
        key, subkey, lap_key = jax.random.split(key, 3)
        n_chains, dof = current_pos.shape

        batch, new_pos, acceptance_rate = sampler.draw(
            key=subkey,
            prob_fn=prob_fn,
            prob_params=state.params,
            init_positions=current_pos,
            step_size=step_size,
            n_chains=n_chains,
            dof=dof,
            n_steps=sampling_config.chain_length,
            burn_in=sampling_config.thermalization_steps,
            thinning=sampling_config.thinning_factor,
            box_L=box_L,
        )

        cm = jnp.sum(new_pos, axis=1) / new_pos.shape[-1]
        cm_mean_val = jnp.mean(cm)
        cm_std_val = jnp.std(cm)

        if not training_config.warm_walkers:
            new_pos = current_pos

        if training_config.is_update_step_size:
            step_size = adapt_step_size(
                step_size,
                acceptance_rate,
                min_step=training_config.min_step,
                max_step=training_config.max_step,
                target_acc=training_config.target_acceptance,
                adaptation_rate=training_config.adaptation_rate,
            )

        new_state, E, sigma_e, E_chain, grads, sr_info = compute_step(
            state=state,
            batch=batch,
            hamiltonian=hamiltonian,
            n_chains=n_chains,
            use_qgt=training_config.use_qgt,
            qgt_config=qgt_config,
            auxiliary_losses=auxiliary_losses,
            key=lap_key,
        )

        # Naive MC error of the mean -- it ignores autocorrelation. See
        # docs/adr/0001-correlated-error-estimates.md.
        error_of_mean = sigma_e / jnp.sqrt(batch.shape[0])

        return (
            new_state,
            key,
            new_pos,
            E,
            sigma_e,
            E_chain,
            error_of_mean,
            acceptance_rate,
            step_size,
            grads,
            cm_mean_val,
            cm_std_val,
            sr_info,
        )

    return full_update
