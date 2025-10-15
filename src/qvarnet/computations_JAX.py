import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import linen as nn
from jax.scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import os
import sys

import argparse

class MLP(nn.Module):
    architecture: list
    hidden_activation: callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.architecture) - 1):
            x = nn.Dense(features=self.architecture[i + 1])(x)
            if i < len(self.architecture) - 2:
                x = self.hidden_activation(x)
        return x
    
def local_energy_batch(params, xs, model_apply):

    # get ψ(x) for the whole batch
    psi = jax.vmap(model_apply, in_axes=(None, 0))(params, xs)

    # first and second derivative wrt x for the whole batch
    dpsi_dx = jnp.gradient(psi, xs.squeeze(), axis=0) # squeeze to remove last dim. xs is (N,1). Grad needs a 1D array in the second argument
    d2psi_dx2 = jnp.gradient(dpsi_dx, xs.squeeze(), axis=0)
    print(xs.shape)

    kinetic = -0.5 * d2psi_dx2 / psi
    potential = (0.5 * xs**2).reshape(-1,1)
    return kinetic + potential

def energy_fn(params, batch, model_apply):
    psi = jax.vmap(lambda x: model_apply(params, x))(batch)
    psi_squared = jnp.abs(psi)**2
    local_energy_per_point = local_energy_batch(params, batch, model_apply)

    energy_integrand = psi_squared * local_energy_per_point
    norm = trapezoid(psi_squared.squeeze(), batch.squeeze())
    integral = trapezoid(energy_integrand.squeeze(), batch.squeeze())
    return integral / norm 

def loss_and_grads(params, batch, model_apply):
    E = energy_fn(params, batch, model_apply)
    grad_E = jax.grad(energy_fn, argnums=0)(params, batch, model_apply)
    return E, grad_E

@jax.jit
def train_step(state, batch):
    E, grads = loss_and_grads(state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state, E

def train(n_steps, init_params, model_apply, optimizer, limits=(-5,5), N_samples=1000):

    state = train_state.TrainState.create(
        apply_fn=model_apply,
        params=init_params,
        tx=optimizer
    )

    energy_history = []
    wavefunction_history = []
    state_history = []
    batch = jnp.linspace(limits[0], limits[1], N_samples).reshape(-1, 1)

    for step in range(n_steps):
        # batch = sampler(state.params, state.apply_fn)
        state, energy = train_step(state, batch)
        energy_history.append(energy)
        wavefunction_history.append(state.params)
        state_history.append(state)

        if jnp.isnan(energy):
            print(f"NaN detected in energy at step {step}")
            print(f"parameters at NaN: {state.params}")
            return wavefunction_history[step-1], energy_history[:-1], wavefunction_history[:-1]

        if step % 100 == 0:
            print(f"Step {step}, Energy: {energy}")

    return state.params, energy_history, wavefunction_history

def run_training_pipeline(architecture, N_samples, N_epochs, lims, lr, rng_seed):
    model = MLP(architecture=architecture)
    rng = jax.random.PRNGKey(rng_seed)
    input_shape = (N_samples,1)
    params = model.init(rng, jnp.ones(input_shape)) # look details
    params_fin, energies, wavefun_history = train(N_epochs, params, model.apply, optax.adam(lr), limits=lims, N_samples=N_samples)
    os.makedirs("outputs", exist_ok=True)
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
    ax1.plot(energies)
    # Reconstruct wavefunction
    x = jnp.linspace(lims[0], lims[1],N_samples).reshape(-1,1)
    psi_approx = model.apply(params_fin, x)
    print(type(psi_approx))
    print(psi_approx.shape)
    norm = jnp.sqrt(trapezoid((psi_approx**2).squeeze(), x.squeeze()))
    print(f"Norm: {norm}")
    psi_approx = psi_approx / norm
    print(f"Norm after normalization: {jnp.sqrt(trapezoid((psi_approx**2).squeeze(), x.squeeze()))}")
    ax2.plot(x, psi_approx**2)
    ax2.plot(x, jnp.pi**(-0.5)*jnp.exp(-x**2), linestyle='dashed')
    plt.savefig(f"outputs/rng_seed_{rng_seed}.png")

if __name__ == "__main__":
    model = MLP(architecture=[1, 30, 1])
    rng = jax.random.PRNGKey(11)

    N_samples = 10_000

    N_epochs = 5000

    lims = (-5, 5)

    input_shape = (N_samples, 1)  # Batch size of 1000, input dimension
    params = model.init(rng, jnp.ones(input_shape))  # Initialize parameters
    params_fin, energy, wf = train(N_epochs, params, model.apply, optax.adam(1e-3), limits=lims, N_samples=N_samples)
    plt.plot(energy)
    plt.show()

    # Reconstruct wavefunction
    x = jnp.linspace(-5,5,1000).reshape(-1,1)
    psi_approx = model.apply(params_fin, x)
    print(type(psi_approx))
    print(psi_approx.shape)
    norm = jnp.sqrt(trapezoid((psi_approx**2).squeeze(), x.squeeze()))
    print(f"Norm: {norm}")
    psi_approx = psi_approx / norm
    print(f"Norm after normalization: {jnp.sqrt(trapezoid((psi_approx**2).squeeze(), x.squeeze()))}")
    plt.plot(x, psi_approx**2)
    plt.plot(x, jnp.pi**(-0.5)*jnp.exp(-x**2), linestyle='dashed')