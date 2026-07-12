"""Does ParticleSubsetMove improve mixing on the real CS N=30 state?

Acceptance alone is a vanity metric — a subset move trivially accepts more because it
changes less. The honest comparison: adapt each proposal's step size to ~50%
acceptance on the trained |ψ|², then measure the integrated autocorrelation time
(IAT) of a physical observable (Σx², the trap energy) per MH step. Lower τ = more
independent samples per model evaluation (every proposal costs exactly one ψ eval
per step).

Setup: adam_train 2000 epochs (decent |ψ|², λ→~0.7), equilibrated walkers carried
from the run; per proposal: 8 rounds of proportional step adaptation, then a
4000-step chain history → IAT via samplers.diagnostics.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from qvarnet.boundaries import NoBoundary
from qvarnet.config.coord_mode import LabCoords
from qvarnet.hamiltonian.continuous import CalogeroSutherlandHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.envelopes import GaussianEnvelope
from qvarnet.models.jastrow import LogJastrow
from qvarnet.models.mlp import MLP
from qvarnet.recipes import adam_train
from qvarnet.samplers import (
    GaussianMove,
    ParticleSubsetMove,
    integrated_autocorr_time,
    sample_and_process,
)
from qvarnet.train import train
from qvarnet.vmc.probability import build_prob_fn

N, CS_L = 30, 0.8
N_CHAINS = 128  # small: the notebook kernel holds most of the GPU; history must fit
SHAPE = (N_CHAINS, N)

model = LogWavefunction(
    transform=NoBoundary(),
    network=MLP(hidden=[128]),
    envelope=GaussianEnvelope(),
    jastrow=LogJastrow(n_particles=N, lambda_init=CS_L),
)
ham = CalogeroSutherlandHamiltonian(L=CS_L, epsilon=1e-12)

t0 = time.time()
rA = train(shape=SHAPE, model=model, hamiltonian=ham, coord_mode=LabCoords(),
           **adam_train(n_epochs=2000, learning_rate=1e-2,
                        checkpoint_path="./probe5/adam"))
print(f"[adam] 2000 ep in {time.time() - t0:.0f}s | "
      f"E tail {np.mean(rA.history.get('energy')[-50:]):.2f}", flush=True)

prob_fn = build_prob_fn(model.apply)
params = rA.best_params()
walkers = jnp.asarray(rA.final_positions)

proposals = {
    "gaussian (all 30)": GaussianMove(),
    "subset n_move=1  ": ParticleSubsetMove(n_move=1),
    "subset n_move=3  ": ParticleSubsetMove(n_move=3),
    "subset n_move=10 ": ParticleSubsetMove(n_move=10),
}

for tag, prop in proposals.items():
    # adapt step to ~50% acceptance (proportional, 8 short rounds)
    step = float(rA.final_step_size)
    for i in range(8):
        _, _, acc = sample_and_process(
            key=jax.random.fold_in(jax.random.PRNGKey(1), i), prob_fn=prob_fn,
            prob_params=params, init_positions=walkers, step_size=step,
            n_chains=N_CHAINS, dof=N, n_steps=40, burn_in=39, thinning=1,
            proposal=prop,
        )
        step = float(np.clip(step * float(jnp.mean(acc)) / 0.5, 1e-3, 10.0))

    # long raw history (no thinning) → IAT of Σx² per chain
    n_steps = 3000
    raw, _, acc = sample_and_process(
        key=jax.random.PRNGKey(7), prob_fn=prob_fn, prob_params=params,
        init_positions=walkers, step_size=step, n_chains=N_CHAINS, dof=N,
        n_steps=n_steps, burn_in=0, thinning=1, proposal=prop,
    )
    hist = np.asarray(raw).reshape(N_CHAINS, n_steps, N)
    obs = np.sum(hist**2, axis=-1)  # Σx² per (chain, step)
    taus = np.array([float(integrated_autocorr_time(jnp.asarray(o))) for o in obs[:64]])
    print(
        f"[{tag}] step {step:6.3f} | acc {float(jnp.mean(acc)):.2f} | "
        f"IAT(Σx²) median {np.median(taus):7.1f}  p90 {np.percentile(taus, 90):7.1f} "
        f"| eff. samples/1000 steps ≈ {1000 / (2 * np.median(taus)):.1f}",
        flush=True,
    )
