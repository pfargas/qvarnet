import jax
import jax.numpy as jnp
from jax import random

from flax.training import train_state
from jax.scipy.integrate import trapezoid
from .callback import nan_callback

import numpy as np
import matplotlib.pyplot as plt
import os
from .sampler import mh_chain

import signal

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


def laplace(func, x):
    """Compute the Laplace operator of the model output with respect to inputs."""
    grad_fn = jax.grad(func)
    d2_dx2 = 0
    for i in range(x.shape[1]):
        d2_dx2 += jax.vmap(jax.grad(lambda xi: grad_fn(xi)[i]))(x)[:, i]
    return d2_dx2


# compute kinetic term with AD (correct)
def local_energy_batch(params, xs, model_apply):
    # xs: (batch, 1) or (batch,)
    print("In local_energy_batch:")
    print(params)

    # psi(x) -> scalar
    def psi_fn(x):
        # ensure input has shape (1,) as model expects last-dim features
        return model_apply(params, x).squeeze()

    # second derivative per sample via AD
    # d2psi_fn = jax.vmap(jax.jacfwd(jax.grad(psi_fn)))
    # d2psi = d2psi_fn(xs_flat)  # shape (batch,)
    d2psi = laplace(psi_fn, xs)

    psi_vals = jax.vmap(lambda x: psi_fn(x))(xs)  # shape (batch,)

    # avoid division by zero / small psi
    psi_safe = psi_vals + 1e-12

    kinetic = -0.5 * (d2psi / psi_safe)  # shape (batch,)
    # potential = 0.5 * (xs_flat**2)  # shape (batch,)
    potential = 0.5 * jnp.sum(xs**2, axis=1)
    return (kinetic + potential).reshape(-1, 1)  # keep your (batch,1) convention


def log_psi(x, params, model_apply):
    psi = model_apply(params, x)
    return jnp.log(jnp.abs(psi) + 1e-8).squeeze()  # Add small constant to avoid log(0)


grad_log_psi = jax.grad(
    lambda params, x, model_apply: log_psi(x, params, model_apply), argnums=0
)  # CAREFUL WITH THE DERIVATIVES: WE WANT THE GRADIENT WRT THE PARAMETERS, so we turn the order of the p and x

# now grad_log_psi is a function that takes (params, x, model_apply) and returns the gradient of log_psi wrt params


def energy_fn(params, batch, model_apply):
    local_energy_per_point = local_energy_batch(params, batch, model_apply)
    E = jnp.mean(local_energy_per_point)
    return E, local_energy_per_point


def energy_fn_trapezoidal(params, batch, model_apply):
    psi = jax.vmap(lambda x: model_apply(params, x))(batch)
    psi_squared = jnp.abs(psi) ** 2
    local_energy_per_point = local_energy_batch(params, batch, model_apply)

    energy_integrand = psi_squared * local_energy_per_point
    norm = trapezoid(psi_squared.squeeze(), batch.squeeze())
    integral = trapezoid(energy_integrand.squeeze(), batch.squeeze())
    return integral / norm


def tree_grad_log_psi(x, local_energy, mean_energy, params, model_apply):

    E_centered = (local_energy - mean_energy).squeeze()  # -> (N,)

    # per-sample grads: pytree where each leaf has leading batch dim N
    log_psi_grads = jax.vmap(lambda xx: grad_log_psi(params, xx, model_apply))(x)

    N = E_centered.shape[0]
    assert N == x.shape[0], "Batch size mismatch between E_centered and x"

    def multiply_and_mean(g):
        # g has shape (N, *leaf_shape)
        # build E_centered shaped (N, 1, 1, ..., 1) to broadcast safely
        trailing_singletons = (1,) * (g.ndim - 1)  # if g.ndim == 1, this is ()
        e_shape = (N,) + trailing_singletons
        e = E_centered.reshape(e_shape)  # (N, 1, 1, ...)
        # elementwise multiply then mean over batch axis=0
        return 2.0 * jnp.mean(e * g, axis=0)

    grad_tree = jax.tree.map(multiply_and_mean, log_psi_grads)
    return grad_tree


def loss_and_grads(params, batch, model_apply):
    E, local_energy_per_point = energy_fn(params, batch, model_apply)
    grad_E = tree_grad_log_psi(batch, local_energy_per_point, E, params, model_apply)
    return E, grad_E


@jax.jit
def train_step(state, batch):
    E, grads = loss_and_grads(state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state, E


def train(
    n_steps, init_params, shape, model_apply, optimizer, PBC=10, n_steps_sampler=500
):

    state = train_state.TrainState.create(
        apply_fn=model_apply, params=init_params, tx=optimizer
    )

    def prob_fn(x, params):
        # Ensure x has a batch dimension, run the MLP, and return non-negative per-input values.
        # x = jnp.atleast_1d(x).reshape(-1, 1)  # (batch, 1)
        print("In prob_fn:")
        print("x shape:", x.shape)
        print("params:", params)
        forward = model_apply(params, x).flatten()  # (batch,)
        out = jnp.square(forward)  # non-negative density probability
        return jnp.squeeze(out)  # scalar for scalar input, (batch,) for batch

    print("SHAPE:", shape)

    print("instantiate prob_fn")
    probability_fn = prob_fn
    print("applying prob_fn to test input")
    test_x = jnp.zeros((shape[0], shape[1]))
    print("test_x shape:", test_x.shape)
    test_out = probability_fn(test_x, state.params)
    print("test_out shape:", test_out.shape)
    print("====================\n\n")
    energy_history = []
    # batch = jnp.linspace(-5,5,1000).reshape(-1,1)
    sampler = jax.vmap(mh_chain, in_axes=(0, None, None, None, None, 0), out_axes=0)
    n_chains = shape[0]
    DoF = shape[1] if len(shape) > 1 else 1
    rng_keys = random.split(random.PRNGKey(872643), n_chains)
    # init_position = jax.random.normal(random.PRNGKey(0), (n_chains,)) * (
    #     PBC / 8
    # )  # TODO: This is susceptible to change
    init_position = jnp.zeros(shape)  # start all chains at 0
    print(f"Initial positions shape: {init_position.shape}\n====================\n\n")
    wf_hist = []
    best_energy = jnp.inf
    best_params = None
    debugSampling = False  # TODO: change this to argument

    os.makedirs("results", exist_ok=True)
    if debugSampling:
        # remove all images in results
        for f in os.listdir("results"):
            if f.endswith(".png") or f.endswith(".txt"):
                os.remove(os.path.join("results", f))

    for step in tqdm(range(n_steps)) if tqdm_available else range(n_steps):
        if stop_requested:
            break

        rng_keys = random.split(random.PRNGKey(step), n_chains)
        batch = sampler(
            rng_keys, n_steps_sampler, PBC, prob_fn, state.params, init_position
        )
        batch = batch.reshape(-1, 1)  # Flatten the chains into a single batch

        state, energy = train_step(state, batch)
        energy_history.append(energy)
        wf_hist.append(state.params)
        if energy < best_energy:
            best_energy = energy
            best_params = state.params
        if nan_callback(energy):
            print("NaN detected in energy, stopping training.")
            break
        init_position = batch
        # jax.debug.print("==============================\n\n")

        if step % 1000 == 0 and debugSampling:
            x = jnp.linspace(-PBC / 2, PBC / 2, 1000).reshape(-1, 1)
            plt.clf()
            grad_norm = jnp.sqrt(
                sum([p for p in jax.tree_util.tree_leaves(state.params)])
            )
            grad_norm = np.array(grad_norm)
            with open("results/grad_norms.txt", "a") as f:
                f.write(f"{step}\t{grad_norm}\n")
            plt.title(f"Batch size: {batch.shape[0]}")
            plt.hist(batch, bins=int(np.ceil(np.sqrt(batch.size))), density=True)
            plt.plot(
                x,
                prob_fn(x, state.params)
                / trapezoid(prob_fn(x, state.params), x.flatten()),
                color="red",
            )
            plt.savefig(f"results/wavefunction_step_{step}.png")
            plt.clf()

        if step % 100 == 0 and not tqdm_available:
            print(f"Step {step}, Energy: {energy}")
            # print("Current parameters:")
            # print(state.params)
            print("==============================")

    return state.params, energy_history, wf_hist, best_params, best_energy
