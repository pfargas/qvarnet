# import sys
# sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt

from qvarnet.train import train
from qvarnet.models.exponential import LogExponentialMLPwithPenalty
from qvarnet.models.deep_set import DeepSet
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian

# ── System ────────────────────────────────────────────────────────────────────
# N_PARTICLES = 50
# DIM         = 1
# N_CHAINS    = 5_000
# DoF         = N_PARTICLES * DIM
# SHAPE       = (N_CHAINS, DoF)

# # ── Model: mlp-fourth-decay ───────────────────────────────────────────────────
# ARCHITECTURE = [DoF, 150, 1]   # [input, hidden..., output]
# IS_LOG_MODEL = True

# model = LogExponentialMLPwithPenalty(architecture=ARCHITECTURE, 
#                                      hidden_activation=jax.nn.tanh,
#                                      kernel_init=jax.nn.initializers.normal(stddev=0.01),
#                                      bias_init=jax.nn.initializers.normal(stddev=0.01),
#                                     )

# # ── Hamiltonian: Harmonic Oscillator ──────────────────────────────────────────
OMEGA = 1.0

# hamiltonian = HarmonicOscillatorHamiltonian(omega=OMEGA)

# # ── Optimizer ─────────────────────────────────────────────────────────────────
# LEARNING_RATE = 0.025

# optimizer = optax.adam(learning_rate=LEARNING_RATE)

N_PARTICLES = 50
DIM = 1
N_CHAINS = 5000
DoF = N_PARTICLES * DIM
SHAPE = (N_CHAINS, DoF)
CHAIN_LENGTH = 10

model = DeepSet(
    phi_hidden_architecture=[2],
    F_hidden_architecture=[1],
    hidden_internal_dimension=1,
    n_particles=N_PARTICLES,
    kernel_init=jax.nn.initializers.normal(stddev=0.01),
    bias_init=jax.nn.initializers.normal(stddev=0.01),
)
IS_LOG_MODEL = True  # Code works for log models.

hamiltonian = HarmonicOscillatorHamiltonian(omega=OMEGA)

optimizer = optax.adam(learning_rate=0.025)


# ── Sampler ───────────────────────────────────────────────────────────────────
STEP_SIZE            = 0.5
CHAIN_LENGTH         = 10
THERMALIZATION_STEPS = 10
THINNING_FACTOR      = 1
PBC                  = 40.0

sampler_params = {
    "step_size":            STEP_SIZE,
    "chain_length":         CHAIN_LENGTH + 1,
    "thermalization_steps": THERMALIZATION_STEPS,
    "thinning_factor":      THINNING_FACTOR,
    "PBC":                  PBC,
}

# ── Training ──────────────────────────────────────────────────────────────────
N_EPOCHS             = 5_000
RNG_SEED             = 42
WARM_WALKERS         = True
IS_UPDATE_STEP_SIZE  = True
MIN_STEP             = 1e-5
MAX_STEP             = 5.0
SAVE_CHECKPOINTS     = False
CHECKPOINT_PATH      = "./"

history, cm_mean, cm_std = train(
    n_epochs=N_EPOCHS,
    shape=SHAPE,
    model=model,
    optimizer=optimizer,
    sampler_params=sampler_params,
    hamiltonian=hamiltonian,
    rng_seed=RNG_SEED,
    warm_walkers=WARM_WALKERS,
    is_update_step_size=IS_UPDATE_STEP_SIZE,
    is_log_model=IS_LOG_MODEL,
    min_step=MIN_STEP,
    max_step=MAX_STEP,
    save_checkpoints=SAVE_CHECKPOINTS,
    checkpoint_path=CHECKPOINT_PATH,
)

# ── Extract history ───────────────────────────────────────────────────────────
energies   = np.array([s.energy for s in history]) / N_PARTICLES
stds       = np.array([s.std    for s in history]) / N_PARTICLES
acc_rates  = np.array([float(jnp.mean(s.acceptance_rate)) for s in history])
step_sizes = np.array([float(s.step_size) for s in history])
cm_mean_arr = np.array([float(x) for x in cm_mean])
cm_std_arr  = np.array([float(x) for x in cm_std])

steps   = np.arange(len(history))
E_EXACT = 0.5 * OMEGA  # ground state energy: 0.5 * hbar * omega (hbar=1)

# ── Summary ───────────────────────────────────────────────────────────────────
TAIL = 100
final_E   = energies[-TAIL:].mean()
final_std = stds[-TAIL:].mean()
error     = abs(final_E - E_EXACT)

print(f"Exact E₀             : {E_EXACT:.6f}")
print(f"Final E (last {TAIL}) : {final_E:.6f} ± {final_std:.6f}")
print(f"|E - E₀|             : {error:.6f}")
print(f"Relative error       : {error / E_EXACT * 100:.3f}%")
print(f"Final accept rate    : {acc_rates[-TAIL:].mean():.3f}")
print(f"Final step size      : {step_sizes[-1]:.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

ax = axes[0, 0]
ax.plot(steps, energies, lw=0.8, label='E')
ax.fill_between(steps, energies - stds, energies + stds, alpha=0.3, label='±σ')
ax.axhline(E_EXACT, color='red', ls='--', label=f'Exact E₀ = {E_EXACT}')
ax.set_xlabel('Epoch'); ax.set_ylabel('Energy'); ax.set_title('Energy'); ax.legend()

ax = axes[0, 1]
ax.semilogy(steps, np.abs(energies - E_EXACT), lw=0.8, color='C1')
ax.set_xlabel('Epoch'); ax.set_ylabel('|E - E₀|'); ax.set_title('Energy error (log scale)')

ax = axes[0, 2]
ax.semilogy(steps, stds, lw=0.8, color='C2')
ax.set_xlabel('Epoch'); ax.set_ylabel('σ(E)'); ax.set_title('Energy std (log scale)')

ax = axes[1, 0]
ax.plot(steps, acc_rates, lw=0.8, color='C3')
ax.axhline(0.5, color='red', ls='--', label='target 0.5')
ax.set_xlabel('Epoch'); ax.set_ylabel('Acceptance rate'); ax.set_title('MH acceptance rate'); ax.legend()

ax = axes[1, 1]
ax.plot(steps, step_sizes, lw=0.8, color='C4')
ax.set_xlabel('Epoch'); ax.set_ylabel('Step size'); ax.set_title('MH step size')

ax = axes[1, 2]
ax.plot(steps, cm_mean_arr, lw=0.8, color='C5', label='⟨R_cm⟩')
ax.fill_between(steps, cm_mean_arr - cm_std_arr, cm_mean_arr + cm_std_arr, alpha=0.3, label='±std')
ax.axhline(0.0, color='red', ls='--', label='expected 0')
ax.set_xlabel('Epoch'); ax.set_ylabel('R_cm'); ax.set_title('Centre of mass'); ax.legend()

plt.tight_layout()
# plt.savefig('training_results_deep_set_lecun_10000_epoch.png', dpi=150, bbox_inches='tight')
plt.show()

from plot_param_dashboard import compute_stats, plot_dashboard

stats = compute_stats(history, target="params")

plot_dashboard(stats)

stats_grad = compute_stats(history, target="grads")

plot_dashboard(stats_grad)
