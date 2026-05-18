"""
Demonstration that the E_L^2 cusp penalty does not help (and actively hurts)
for the Calogero-Sutherland model with a Jastrow+MLP ansatz.

Runs three short training sessions with identical seeds:
  A) No cusp penalty                  (baseline)
  B) Cusp penalty, normalised E_L^2   (current implementation)
  C) Cusp penalty, un-normalised      (to show gradient explosion without normalisation)

Produces cusp_demo.png.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np
import matplotlib.pyplot as plt
import optax

from qvarnet.train import train
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.config.coord_mode import LabCoords
from qvarnet.models.exponential import JastrowLogExponentialMLPwithPenalty
from qvarnet.hamiltonian.continuous import CalogeroSutherlandHamiltonian

# ── System ────────────────────────────────────────────────────────────────────
N_PARTICLES = 5
L_COUPLING  = 1.8
N_CHAINS    = 500          # small for speed
N_EPOCHS    = 800
SHAPE       = (N_CHAINS, N_PARTICLES)

E_EXACT = N_PARTICLES * (1 + L_COUPLING * (N_PARTICLES - 1)) / N_PARTICLES  # per particle

hamiltonian = CalogeroSutherlandHamiltonian(L=L_COUPLING, epsilon=1e-8)
optimizer   = optax.adam(learning_rate=1e-3)
sampler_params = {
    "step_size": 0.5, "chain_length": 11,
    "thermalization_steps": 10, "thinning_factor": 1, "PBC": 40.0,
}

def make_model():
    return JastrowLogExponentialMLPwithPenalty(
        architecture=[N_PARTICLES, 64, 1], lambda_init=0.5
    )

def make_cfg(**extra):
    return TrainingConfig(
        n_epochs=N_EPOCHS, rng_seed=42,
        warm_walkers=True, is_update_step_size=True,
        is_log_model=True, min_step=1e-5, max_step=5.0,
        **extra,
    )

def run(label, cfg):
    print(f"\n{'='*50}\nRunning: {label}\n{'='*50}")
    hist, _, _ = train(
        shape=SHAPE, model=make_model(), optimizer=optimizer,
        hamiltonian=hamiltonian, training_config=cfg,
        sampler_params=sampler_params, coord_mode=LabCoords(),
    )
    energies = np.array([float(s.energy) for s in hist]) / N_PARTICLES
    lambdas  = np.array([float(s.params["params"]["lam"]) for s in hist])
    tail_E   = energies[-100:].mean()
    tail_lam = lambdas[-1]
    print(f"  Final E/N : {tail_E:.4f}  (exact: {E_EXACT:.4f})")
    print(f"  Final λ   : {tail_lam:.4f}  (target: {L_COUPLING})")
    return energies, lambdas

# ── Run A: baseline ───────────────────────────────────────────────────────────
E_A, lam_A = run("A — no cusp penalty (baseline)", make_cfg())

# ── Run B: normalised E_L^2 cusp penalty ─────────────────────────────────────
E_B, lam_B = run(
    "B — normalised E_L^2 cusp penalty (alpha=1.0)",
    make_cfg(use_cusp_condition=True, cusp_alpha=1.0,
             cusp_epsilon=1e-2, cusp_n_configs_per_pair=3),
)

# ── Run C: small alpha, see if scale matters ───────────────────────────────────
E_C, lam_C = run(
    "C — normalised E_L^2 cusp penalty (alpha=0.1)",
    make_cfg(use_cusp_condition=True, cusp_alpha=0.1,
             cusp_epsilon=1e-2, cusp_n_configs_per_pair=3),
)

# ── Plot ───────────────────────────────────────────────────────────────────────
steps = np.arange(N_EPOCHS)
CLIP  = 3 * E_EXACT          # clip y-axis so explosions don't dominate the plot

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(
    f"Cusp penalty demo — CS model  N={N_PARTICLES}  L={L_COUPLING}  "
    f"λ_init=0.5  ({N_EPOCHS} epochs)",
    fontsize=12,
)

ax = axes[0]
for E, label, c in [
    (E_A, "A: no cusp (baseline)",      "C0"),
    (E_B, "B: cusp α=1.0",              "C1"),
    (E_C, "C: cusp α=0.1",              "C2"),
]:
    ax.plot(steps, np.clip(E, -CLIP, CLIP), lw=0.8, color=c, label=label)
ax.axhline(E_EXACT, color="red", ls="--", lw=1.2, label=f"Exact E₀/N={E_EXACT:.2f}")
ax.set_xlabel("Epoch"); ax.set_ylabel("E/N (clipped)")
ax.set_title("Energy convergence"); ax.legend(fontsize=9)
ax.set_ylim(E_EXACT - 1, CLIP)

ax = axes[1]
for lam, label, c in [
    (lam_A, "A: no cusp (baseline)", "C0"),
    (lam_B, "B: cusp α=1.0",         "C1"),
    (lam_C, "C: cusp α=0.1",         "C2"),
]:
    ax.plot(steps, lam, lw=0.8, color=c, label=label)
ax.axhline(L_COUPLING, color="red", ls="--", lw=1.2, label=f"Target λ={L_COUPLING}")
ax.set_xlabel("Epoch"); ax.set_ylabel("λ (Jastrow exponent)")
ax.set_title("Jastrow exponent convergence λ → L"); ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "cusp_demo.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()
