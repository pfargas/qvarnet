import jax
import jax.numpy as jnp
from jax import random

from flax.training import train_state
from jax.scipy.integrate import trapezoid
from .callback import nan_callback, update_best_params
from .sampler import mh_chain

import signal

from functools import partial

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")


stop_requested = False


def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
    print("Signal received, will stop after current training step.")


signal.signal(signal.SIGINT, signal_handler)


# @partial(jax.jit, static_argnames=["func"])
def laplace_OLD(func, x):
    """Compute the Laplace operator of the model output with respect to inputs."""
    grad_fn = jax.grad(func)
    d2_dx2 = 0
    for i in range(x.shape[1]):
        d2_dx2 += jax.vmap(jax.grad(lambda xi: grad_fn(xi)[i]))(x)[:, i]
    return d2_dx2


@partial(jax.jit, static_argnames=["model_apply"])
def laplace_autodiff_new(params, xs, model_apply):
    """
    Computes Laplacian using Forward-over-Reverse AD.
    Memory efficient: O(DoF) instead of O(DoF^2).
    """

    def psi_fn(x):
        return model_apply(params, x.reshape(1, -1)).squeeze()

    def laplacian_single(x):
        # We want sum_i (d^2 psi / dx_i^2)
        # We can calculate this by looping over dimensions and projecting
        # the gradient onto the unit vector e_i.

        n_dims = x.shape[0]

        def body_fun(i, val):
            # Create unit vector e_i
            e_i = jnp.eye(n_dims)[i]

            # jvp(grad(psi), (primal,), (tangent,))
            # resulting tangent is (Hessian * e_i)
            # We take the i-th component of that vector.
            grad_dot_hessian = jax.jvp(jax.grad(psi_fn), (x,), (e_i,))[1]

            return val + grad_dot_hessian[i]

        return jax.lax.fori_loop(0, n_dims, body_fun, 0.0)

    return jax.vmap(laplacian_single)(xs)


def laplace_autodiff_FULL_HESSIAN(params, xs, model_apply):
    """
    Computes the Laplacian Δψ = ∇²ψ using JAX's Automatic Differentiation.
    This is shape-safe and significantly faster than central differences.
    """

    def psi_fn(x):
        # x is a single point (DoF,)
        return model_apply(params, x.reshape(1, -1)).squeeze()

    def laplacian_fn(x):  # memory throttle: DoF^2
        return jnp.trace(jax.hessian(psi_fn)(x))

    # Vectorize over the batch
    return jax.vmap(laplacian_fn)(xs)


def laplace_central_difference(params, xs, model_apply, h=1e-4):
    """
    Computes Laplacian using central difference, properly handling JAX batching.
    xs shape: (batch, DoF)
    """

    def psi_single(x_single):
        # x_single shape: (DoF,)
        # Reshape to (1, DoF) because Flax models expect a batch dimension
        return model_apply(params, x_single.reshape(1, -1)).squeeze()

    def single_point_laplacian(x):
        # x shape: (DoF,)
        d2psi = 0.0
        for i in range(x.shape[0]):
            # Create unit vector for coordinate i
            ei = jnp.eye(x.shape[0])[i]

            # Central difference formula: [f(x+h) - 2f(x) + f(x-h)] / h^2
            f_plus = psi_single(x + h * ei)
            f_main = psi_single(x)
            f_minus = psi_single(x - h * ei)

            d2psi += (f_plus - 2 * f_main + f_minus) / (h**2)
        return d2psi

    # Vectorize the single-point logic over the entire batch
    return jax.vmap(single_point_laplacian)(xs)


@partial(jax.jit, static_argnames=["model_apply", "h"])
def laplace_central_difference_scan(params, xs, model_apply, h=1e-4):
    """
    Computes Laplacian using jax.lax.scan.
    Memory: O(N * D) (Linear scaling)
    Speed: Single compiled kernel (Fast)
    """
    batch_size, n_dims = xs.shape

    # 1. Pre-calculate the center value once
    # f_main shape: (batch,)
    f_main = model_apply(params, xs).squeeze()

    def scan_body(carry, i):
        # i is the dimension index we are currently perturbing

        # Create unit vector e_i
        e_i = jnp.eye(n_dims)[i]  # Shape: (D,)

        # Perturb current dimension i for the whole batch
        # We broadcast e_i to (batch, D)
        x_plus = xs + h * e_i
        x_minus = xs - h * e_i

        # Evaluate model at perturbed points
        # These run sequentially inside the compiled kernel, saving memory
        psi_plus = model_apply(params, x_plus).squeeze()
        psi_minus = model_apply(params, x_minus).squeeze()

        # Finite difference for dimension i
        d2_dx2 = (psi_plus - 2 * f_main + psi_minus) / (h**2)

        # Accumulate the result (Laplacian is sum of d2_dx2)
        new_laplacian = carry + d2_dx2

        return new_laplacian, None  # We don't need to stack outputs

    # 2. Run the loop inside XLA
    # init_val is zeros of shape (batch,)
    # xs=jnp.arange(n_dims) gives us the loop indices 0..D-1
    final_laplacian, _ = jax.lax.scan(
        scan_body, init=jnp.zeros(batch_size), xs=jnp.arange(n_dims)
    )

    return final_laplacian


# @jax.jit
def V(x):
    """Harmonic oscillator potential."""
    return 0.5 * jnp.sum(x**2, axis=1)  # sum over dimensions


@partial(jax.jit, static_argnames=["model_apply"])
def local_energy_batch(params, xs, model_apply):
    # TODO: Look into jax.vmap for better performance
    # TODO: Consider using jax.jacfwd/jacrev for Hessian computation
    # BUG: Ensure that laplace doesn't blow the memory for large architectures/batches
    def psi_fn(x):
        # ensure input has shape (1,) as model expects last-dim features
        x = jnp.atleast_1d(x).reshape(1, -1)  # (1, DoF)
        return model_apply(params, x).squeeze()

    d2psi = laplace_autodiff_new(params, xs, model_apply)

    psi_vals = jax.vmap(lambda x: psi_fn(x))(xs)  # shape (batch,)

    psi_safe = psi_vals + 1e-12

    kinetic = -0.5 * (d2psi / psi_safe)  # shape (batch,)
    potential = V(xs)  # shape (batch,)
    return (kinetic + potential).reshape(-1, 1)  # keep your (batch,1) convention


# @partial(jax.jit, static_argnames=["model_apply"])
def log_psi(x, params, model_apply):
    psi = model_apply(params, x)
    return jnp.log(jnp.abs(psi) + 1e-8).squeeze()  # Add small constant to avoid log(0)


def grad_log_psi(params, x, model_apply):
    """Compute the gradient of log(psi) with respect to parameters."""
    return jax.grad(lambda p: log_psi(x, p, model_apply), argnums=0)(params)


# @partial(jax.jit, static_argnames=["model_apply"])
def energy_fn(params, batch, model_apply):
    local_energy_per_point = local_energy_batch(params, batch, model_apply)
    E = jnp.mean(local_energy_per_point)
    return E, local_energy_per_point


# @partial(jax.jit, static_argnames=["model_apply"])
def loss_and_grads(params, batch, model_apply):
    E, local_energy_per_point = energy_fn(params, batch, model_apply)
    loss = lambda p: 2 * jnp.mean(
        jax.lax.stop_gradient(local_energy_per_point - E)
        * log_psi(batch, p, model_apply).reshape(-1, 1)
    )
    grad_E = jax.grad(loss)(params)
    return E, grad_E


# ============================================================================
# QGT INTEGRATION POINTS - ADD QGT-BASED TRAINING METHODS HERE
# ============================================================================

# TODO: Add QGT-based training step function:
# def train_step_qgt(state, batch, qgt_config):
#     """Training step using Quantum Geometric Tensor preconditioning."""
#     # Import QGT functions: from .qgt import train_step_qgt, compute_qgt_statistics
#     E, grads = loss_and_grads(state.params, batch, state.apply_fn)
#
#     # Use train_step_qgt from qgt.py:
#     # new_state, energy = train_step_qgt(state, batch, qgt_config)
#     return new_state, E

# TODO: Extend main train() function to support QGT optimizer:
# - Add parameter: qgt_config=None
# - Add optimizer check: if optimizer == 'qgt':
# - Call qgt training step instead of standard train_step
# - Import: from .qgt import QGTConfig, DEFAULT_QGT_CONFIG

# TODO: Add QGT configuration parsing in CLI:
# - File: src/qvarnet/cli/run.py
# - Add --optimizer qgt option
# - Parse qgt_config from parameters file
# - Handle QGT-specific hyperparameters


@jax.jit
def train_step(state, batch):
    E, grads = loss_and_grads(state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state, E


def train(
    n_epochs,
    init_params,
    shape,
    model_apply,
    optimizer,
    sampler_params,
    rng_seed=0,
    split_sampler=False,
    hamiltonian_params=None,
):
    r"""Main function to optimize the wavefunction parameters using Variational Monte Carlo.

    The optimizer approach is based on gradient descent:

    .. math::

        \theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)

    Where $\mathcal{L}$ is the loss function defined as:

    .. math::

        \Psi

    .. math::

        \mathcal{L}(\theta) = 2
        E_{x \sim |\psi_\theta(x)|^2}\Big[
            (E_{\rm loc}(x) - \langle E \rangle)
            \log |\psi_\theta(x)|
        \Big]



    Args:
        n_epochs: Number of training epochs.
        init_params: Initial parameters of the wavefunction model.
        shape: Shape of the input data (batch_size, DoF).
    """

    if split_sampler:
        print("Using sampler_split module for sampling.")
        from .sampler_split import mh_chain

        sampler = jax.vmap(
            mh_chain,
            in_axes=(
                0,  # key (n_chains, )
                None,  # PBC
                None,  # prob_fn
                None,  # prob_params
                0,  # init_position (n_chains, DoF)
                None,  # step_size
                None,  # n_steps
            ),
            out_axes=0,
        )
    else:
        from .sampler import mh_chain

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

    state = train_state.TrainState.create(
        apply_fn=model_apply, params=init_params, tx=optimizer
    )

    def prob_fn(x, params):
        forward = model_apply(params, x).flatten()  # (batch,)
        out = jnp.square(forward)  # non-negative density probability
        return jnp.squeeze(out)  # scalar for scalar input, (batch,) for batch

    n_chains = shape[0]
    DoF = shape[1] if len(shape) > 1 else 1

    key = random.key(rng_seed)
    energy_history = jnp.zeros(n_epochs)
    init_position = jnp.zeros(shape)  # start all chains at 0
    best_energy = jnp.inf
    best_params = init_params
    step_size = sampler_params.get("step_size", 1.0)
    n_steps_sampler = sampler_params.get("chain_length", 500)
    PBC = sampler_params.get("PBC", 40.0)

    for step in tqdm(range(n_epochs)) if tqdm_available else range(n_epochs):
        if stop_requested:
            break

        # --------------------------------------------
        # ---            SAMPLING STEP             ---
        # --------------------------------------------
        if split_sampler:
            keys = random.split(key, n_chains)
            batch = sampler(
                keys,
                PBC,
                prob_fn,
                state.params,
                init_position,
                step_size,
                n_steps_sampler,
            )
        else:
            key, subkey = random.split(key)
            rand_nums = random.uniform(subkey, (n_chains, n_steps_sampler, DoF + 1))
            batch = sampler(
                rand_nums,
                PBC,
                prob_fn,
                state.params,
                init_position,
                step_size,
            )
        # combine n_chains and n_steps_sampler into one big batch
        batch = batch[:, 50:, :]  # burn-in
        batch = batch[:, ::5, :]  # Thinning to reduce correlations
        batch = batch.reshape(-1, DoF)  # (n_chains * n_steps_sampler, DoF)

        # check_size_batch(batch, step, n_chains, n_steps_sampler)

        # --------------------------------------------
        # ---          TRAINING STEP              ---
        # --------------------------------------------
        state, energy = train_step(state, batch)

        energy_history = energy_history.at[step].set(energy)

        if energy < best_energy:
            best_energy = energy
            best_params = state.params

    return state.params, energy_history, best_params, best_energy


def check_size_batch(batch, step, n_chains, n_steps_sampler):
    with open("batch_debug.txt", "a") as f:
        f.write(f"# STEP {step}\n")
        f.write("BATCH INFO\n")
        f.write(f"Batch shape: {batch.shape}\n")
        f.write(f"Batch data type: {batch.dtype}\n")
        f.write(f"Batch size in bytes: {batch.nbytes} bytes\n")

        num = batch[0]
        f.write("EXPLORING SINGLE SAMPLE\n")
        size_in_bytes = num.nbytes
        f.write(f"Sample size: {num.shape}, Size in bytes: {size_in_bytes} bytes\n")
        f.write(f"Sample data: {num}\n")
        # check the attributes of the single num
        f.write(f"Type: {type(num)}\n")
        for mini_num in num:
            f.write(f"  Mini value: {mini_num}, Type: {type(mini_num)}\n")

        f.write(
            f"computed size as the product of shape dimensions and itemsize: {n_chains * n_steps_sampler * size_in_bytes} bytes\n"
        )
        f.write("\n")
