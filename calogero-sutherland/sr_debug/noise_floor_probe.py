"""Why does SR *heat up* when started from a well-converged state?

Notebook observation (2026-07-11 evening): SR finetune from Adam-20k (σ_E 0.087)
first descends to σ_E 0.022 (beating Adam's best) then climbs to σ_E ~3.7 and sits
there. Hypothesis: constant-η SR has a stationary noise floor — near convergence the
sampled force is MC-noise-dominated and S⁻¹ amplifies it along flat directions;
η sets the equilibrium "temperature". Adam's m̂/√v̂ suppresses zero-mean noise, so
its floor is far lower.

Test: from the same converged Adam endpoint, run classic SR at η = 1e-3 / 3e-4 /
1e-4 (same trust region in *state* units via max_state_change, so the guard is not
the variable) + an Adam(1e-4) control. If the floor is a noise temperature, final
σ_E should scale ≈ linearly with η. Also log λ to rule out cusp drift.
"""

import time

import numpy as np
import optax

from qvarnet.boundaries import NoBoundary
from qvarnet.config.coord_mode import LabCoords
from qvarnet.config.training_setup import ChainInitAndWarmupConfig, TrainingConfig
from qvarnet.geometry.qgt import QGTConfig
from qvarnet.hamiltonian.continuous import CalogeroSutherlandHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.envelopes import GaussianEnvelope
from qvarnet.models.jastrow import LogJastrow
from qvarnet.models.mlp import MLP
from qvarnet.train import train

N, CS_L = 30, 0.8
N_CHAINS = 2048  # 4096 OOMs next to the live notebook kernel; η-scaling is internal
SHAPE = (N_CHAINS, N)
E0 = N * (1 + CS_L * (N - 1))


def model(lambda_init=CS_L):
    return LogWavefunction(
        transform=NoBoundary(),
        network=MLP(hidden=[128]),
        envelope=GaussianEnvelope(),
        jastrow=LogJastrow(n_particles=N, lambda_init=lambda_init),
    )


ham = CalogeroSutherlandHamiltonian(L=CS_L, epsilon=1e-12)
SAMPLER = {"step_size": 0.5, "chain_length": 21, "thermalization_steps": 20,
           "thinning_factor": 1}


def report(tag, r, t):
    h = r.history
    e, s = h.get("energy"), h.get("std")
    lam = float(r.final_params["params"]["jastrow"]["lambda"])
    n = len(e)
    q = n // 4
    sq = " → ".join(f"{s[i * q:(i + 1) * q].mean():.3f}" for i in range(4))
    best = float(np.min(s))
    print(f"[{tag}] {n} ep in {t:.0f}s | σ_E quarters {sq} | best σ_E {best:.4f} "
          f"| final E {e[-1]:.3f} | lam {lam:.4f}", flush=True)


# Stage A replicates the notebook's converged run: lambda_init=1.2, 20k epochs.
# (From lambda_init=L, Adam breaks the cusp early and 12k epochs was not enough —
# the previous stage-A attempt ended at lam=0.766, sigma_E ~39.)
t0 = time.time()
rA = train(shape=SHAPE, model=model(lambda_init=1.2), hamiltonian=ham,
           coord_mode=LabCoords(),
           optimizer=optax.adam(1e-2),
           training_config=TrainingConfig(
               n_epochs=20_000, rng_seed=0, warm_walkers=True,
               is_update_step_size=True, checkpoint_path="./probe6/adam",
               print_summary=False),
           sampler_params=SAMPLER,
           initial_chain_config=ChainInitAndWarmupConfig(
               init_positions="normal", init_position_params={"mean": 0.0, "std": 0.5},
               warmup_steps=300, warmup_adapt_step_size=True))
report("A adam 20k    ", rA, time.time() - t0)

EP = 2000
variants = [
    ("sr eta=1e-3  ", optax.sgd(1e-3), 1e-3, True),
    ("sr eta=3e-4  ", optax.sgd(3e-4), 3e-4, True),
    ("sr eta=1e-4  ", optax.sgd(1e-4), 1e-4, True),
    ("adam 1e-4    ", optax.adam(1e-4), 1e-3, False),
]
for tag, opt, eta, use_qgt in variants:
    t0 = time.time()
    r = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
              optimizer=opt,
              init_params=rA.best_params(),
              training_config=TrainingConfig(
                  n_epochs=EP, rng_seed=0, warm_walkers=True,
                  is_update_step_size=True, use_qgt=use_qgt,
                  checkpoint_path=f"./probe6/{tag.strip().replace(' ', '_').replace('=', '')}",
                  print_summary=False),
              # same max_state_change in STATE units for every eta — the trust
              # region is not the variable here
              qgt_config=QGTConfig(solver="auto", learning_rate=eta,
                                   regularization=1e-2, max_state_change=0.1),
              sampler_params=dict(SAMPLER, step_size=rA.final_step_size),
              initial_chain_config=ChainInitAndWarmupConfig(
                  init_positions=rA.final_positions, warmup_steps=100,
                  warmup_adapt_step_size=True))
    report(tag, r, time.time() - t0)
