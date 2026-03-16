import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from .vmc_state import VMCState
from .callbacks import *
from .samplers import mh_chain

import signal

from functools import partial

from .utils import (
    load_doc,
    save_checkpoint,
    load_checkpoint,
    numerical_parameter_gradients,
)

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")

from .qgt import (
    compute_natural_gradient,
    DEFAULT_QGT_CONFIG,
)

stop_requested = False


def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
    print("Signal received, will stop after current training step.")


signal.signal(signal.SIGINT, signal_handler)


def compute_local_energy(hamiltonian, params, samples, model_apply):
    return hamiltonian.local_energy(params, samples, model_apply).reshape(-1, 1)


@partial(jax.jit, static_argnames=["model_apply"])
def log_psi(x, params, model_apply):
    psi = model_apply(params, x)
    return jnp.log(jnp.abs(psi) + 1e-8).squeeze()  # Add small constant to avoid log(0)


def grad_log_psi(params, x, model_apply):
    """Compute the gradient of log(psi) with respect to parameters."""
    return jax.grad(lambda p: log_psi(x, p, model_apply), argnums=0)(params)


@partial(jax.jit, static_argnames=["model_apply"])
def energy_fn(hamiltonian, params, batch, model_apply):
    local_energy_per_point = compute_local_energy(
        hamiltonian, params, batch, model_apply
    )
    E = jnp.mean(local_energy_per_point)
    sigma_e = jnp.std(local_energy_per_point)
    return E, local_energy_per_point, sigma_e


@partial(jax.jit, static_argnames=["model_apply"])
def loss_and_grads(hamiltonian, params, batch, model_apply):
    E, E_loc, sigma_e = energy_fn(hamiltonian, params, batch, model_apply)
    tv = 5.0
    E_clipped = jnp.clip(E_loc, E - tv * sigma_e, E + tv * sigma_e)
    loss = lambda p: 2 * jnp.mean(
        jax.lax.stop_gradient(E_clipped - E)
        * log_psi(batch, p, model_apply).reshape(-1, 1)
    )
    grad_E = jax.grad(loss)(params)
    return E, sigma_e, grad_E


@jax.jit
def train_step(
    state, samples, hamiltonian, use_qgt=False, qgt_config=DEFAULT_QGT_CONFIG.to_dict()
):
    E, sigma_e, grads = loss_and_grads(
        hamiltonian, state.params, samples, state.apply_fn
    )
    if not use_qgt:
        new_state = state.apply_gradients(grads=grads)  # modifying the parameters!
    else:
        # Compute natural gradient using QGT
        natural_grad_flat, unravel_fn = compute_natural_gradient(
            state.params, samples, state.apply_fn, grads, qgt_config
        )

        # Apply natural gradient with learning rate
        learning_rate = qgt_config.get("learning_rate", 1e-3)
        new_params_flat = (
            ravel_pytree(state.params)[0] - learning_rate * natural_grad_flat
        )
        new_params = unravel_fn(new_params_flat)

        # Create new state
        new_state = state.replace(params=new_params)
    return new_state, E, sigma_e


@load_doc("train.txt")
def train(
    n_epochs,
    shape,  # n_chains, DoF -> shape of the input to the network
    model,
    optimizer,
    sampler_params,
    hamiltonian,
    rng_seed=0,
    checkpoint_path="./",
    save_checkpoints=False,
    init_positions="normal",
    warm_walkers=False,
    min_step=1e-5,
    max_step=5.0,
    is_update_step_size=False,
):
    """Train a VMC model using Metropolis-Hastings sampling.
    Docs loaded from _docs/train.txt
    """
    key = random.PRNGKey(rng_seed)

    # Vmap the sampler chain over the batch dimension (n_chains)
    sampler = jax.vmap(
        mh_chain,
        in_axes=(
            0,  # random_values (n_chains, n_steps, DoF + 1)
            None,  # PBC
            None,  # prob_fn
            None,  # prob_params
            0,  # init_position (n_chains, DoF)
            None,  # step_size
        ),
        out_axes=0,
    )

    params = model.init(key, jnp.ones(shape))  # Initialize parameters
    state = VMCState.create(apply_fn=model.apply, params=params, tx=optimizer)
    state = load_checkpoint(state, path=checkpoint_path, filename="checkpoint.msgpack")

    init_steps = state.n_step if hasattr(state, "n_step") else 0

    # Define prob_fn for the sampler
    @jax.jit
    def prob_fn(x, params):
        forward = model.apply(params, x).flatten()
        out = jnp.square(forward)
        return jnp.squeeze(out)

    # Use a standard list for history to avoid JAX array updates in loop
    state_history = []

    # Initialize walkers at 0 (or load from checkpoint if you had them)
    # This variable persists across loop iterations to keep chains "warm"
    if init_positions == "normal":
        current_positions = jax.random.normal(key, shape) * 0.5
    elif init_positions == "zeros":
        current_positions = jnp.zeros(shape)
    else:
        raise ValueError(f"Unknown init_positions: {init_positions}")

    step_size = sampler_params.get("step_size", 1.0)
    n_steps_sampler = sampler_params.get("chain_length", 500)
    burn_in_steps = sampler_params.get("thermalization_steps", 50)
    thinning_factor = sampler_params.get("thinning_factor", 5)
    PBC = sampler_params.get("PBC", 40.0)

    # --- JIT-COMPILED HELPER FUNCTIONS ---

    @jax.jit
    def update_step_size(
        step_size,
        acceptance_rate,
        target_acc=0.5,
        adaptation_rate=0.1,
        min_step=1e-5,
        max_step=5.0,
    ):
        """
        Adjusts step_size based on the last recorded acceptance rate.
        If acc > target, increase step_size.
        If acc < target, decrease step_size.
        """
        # Simple multiplicative update
        # We use jnp.clip to prevent the step size from exploding or hitting zero
        factor = 1.0 + adaptation_rate * (jnp.mean(acceptance_rate) - target_acc)
        new_step_size = jnp.clip(step_size * factor, min_step, max_step)
        return new_step_size

    @partial(
        jax.jit, static_argnames=["PBC", "n_steps", "burn_in", "thinning", "shape"]
    )
    def sample_and_process(
        key,
        params,
        init_pos,
        shape,  # n_chains, DoF -> shape of the input to the network
        step_size,
        PBC,
        n_steps,
        burn_in,
        thinning,
    ):
        """Runs the sampler and processes the batch on-device."""
        n_chains, DoF = shape
        rand_nums = jax.random.uniform(key, (n_chains, n_steps, DoF + 1))

        # Run sampler (returns shape: [n_chains, n_steps, DoF])
        raw_batch, acceptance_rates = sampler(
            rand_nums, PBC, prob_fn, params, init_pos, step_size
        )

        # 2. Process batch for training (Burn-in & Thinning)
        batch = raw_batch[:, burn_in:, :]
        batch = batch[:, ::thinning, :]
        last_positions = raw_batch[:, -1, :]
        batch_flat = batch.reshape(-1, DoF)

        return batch_flat, last_positions, acceptance_rates

    @partial(
        jax.jit,
        static_argnames=[
            "PBC",
            "n_steps",
            "burn_in",
            "thinning",
            "shape",
            "warm_walkers",
            "update_step_size",
        ],
    )
    def full_update(
        state,
        key,
        current_pos,
        shape,
        step_size,
        PBC,
        n_steps,
        burn_in,
        thinning,
        hamiltonian,
        warm_walkers=False,
        is_update_step_size=False,
    ):
        """Performs Sampling + Training + Best State Tracking in one compiled block."""
        key, subkey = jax.random.split(key)

        # 1. Sample (and get new walker positions)
        if warm_walkers:
            batch, current_pos, acceptance_rate = sample_and_process(
                subkey,
                state.params,
                current_pos,
                shape,
                step_size,
                PBC,
                n_steps,
                burn_in,
                thinning,
            )
        else:
            batch, _, acceptance_rate = sample_and_process(
                subkey,
                state.params,
                current_pos,
                shape,
                step_size,
                PBC,
                n_steps,
                burn_in,
                thinning,
            )

        # Update step size
        if is_update_step_size:
            step_size = update_step_size(
                step_size, acceptance_rate, min_step=min_step, max_step=max_step
            )

        # 2. Train
        new_state, E, sigma_e = train_step(state, batch, hamiltonian)

        return new_state, key, current_pos, E, sigma_e, acceptance_rate, step_size

    # --------------------------------------------
    # ---          TRAINING LOOP              ---
    # --------------------------------------------

    # _, current_positions, _ = sample_and_process(
    #     key,
    #     state.params,
    #     current_positions,
    #     shape,
    #     step_size,
    #     PBC,
    #     n_steps_sampler * 10,
    #     0,
    #     1,
    # )
    progress_bar = tqdm(range(init_steps, n_epochs), disable=not tqdm_available)

    for step in progress_bar:
        if stop_requested:
            break

        # execute the "Mega-Step"
        new_state, key, current_positions, E, sigma_e, acceptance_rate, step_size = (
            full_update(
                state=state,
                key=key,
                current_pos=current_positions,
                shape=shape,
                step_size=step_size,
                PBC=PBC,
                n_steps=n_steps_sampler,
                burn_in=burn_in_steps,
                thinning=thinning_factor,
                hamiltonian=hamiltonian,
                warm_walkers=warm_walkers,
                is_update_step_size=is_update_step_size,  # Enable adaptive step size adjustment
            )
        )

        # Append state with energy evaluated for its parameters
        state_history.append(
            state.replace(energy=E, std=sigma_e, acceptance_rate=acceptance_rate)
        )
        state = new_state

        # --- Logging & Checkpointing (Throttled) ---

        # Update progress bar only every 10 steps to reduce CPU-GPU sync overhead
        if tqdm_available and step % 10 == 0:
            progress_bar.set_postfix(E=f"{E:.2f}", sigma_E=f"{sigma_e:.2f}")

        # Save checkpoints rarely (e.g., every 50 steps)
        if save_checkpoints and step % 50 == 0:
            save_checkpoint(
                new_state, path=checkpoint_path, filename="checkpoint.msgpack"
            )
    return state_history
