"""M/P interpolation-threshold test, memory-light version.

Instead of raising M past P=4099 (OOMs next to the live notebook kernel), shrink the
model: MLP hidden [63] → P ≈ 2082. Then the SAME M sweep crosses the threshold:
  M=1024  M/P ≈ 0.49  (minSR, healthy underdetermined)
  M=2048  M/P ≈ 0.98  (minSR, AT the singularity — the notebook's regime)
  M=4096  M/P ≈ 1.97  (auto → cholesky, healthy overdetermined)
Same converged stage-A start (per-script; GPU nondeterminism makes stage A vary
between runs, so only within-script comparisons count), same classic SR η=1e-3,
2000 epochs. Prediction: the middle run heats up, the outer two don't.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

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
E0 = N * (1 + CS_L * (N - 1))
HIDDEN = [63]


def model(lambda_init=CS_L):
    return LogWavefunction(
        transform=NoBoundary(),
        network=MLP(hidden=HIDDEN),
        envelope=GaussianEnvelope(),
        jastrow=LogJastrow(n_particles=N, lambda_init=lambda_init),
    )


ham = CalogeroSutherlandHamiltonian(L=CS_L, epsilon=1e-12)
SAMPLER = {"step_size": 0.5, "chain_length": 21, "thermalization_steps": 20,
           "thinning_factor": 1}

P = ravel_pytree(model().init(jax.random.PRNGKey(0), jnp.ones((1, N))))[0].size
print(f"P = {P} parameters (hidden {HIDDEN})", flush=True)


def report(tag, r, t):
    h = r.history
    e, s = h.get("energy"), h.get("std")
    lam = float(r.final_params["params"]["jastrow"]["lambda"])
    n = len(e)
    q = n // 4
    sq = " → ".join(f"{s[i * q:(i + 1) * q].mean():.3f}" for i in range(4))
    print(f"[{tag}] {n} ep in {t:.0f}s | σ_E quarters {sq} | best σ_E "
          f"{float(np.min(s)):.4f} | final E {e[-1]:.3f} | lam {lam:.4f}", flush=True)


t0 = time.time()
rA = train(shape=(2048, N), model=model(lambda_init=1.2), hamiltonian=ham,
           coord_mode=LabCoords(),
           optimizer=optax.adam(1e-2),
           training_config=TrainingConfig(
               n_epochs=20_000, rng_seed=0, warm_walkers=True,
               is_update_step_size=True, checkpoint_path="./probe8/adam",
               print_summary=False),
           sampler_params=SAMPLER,
           initial_chain_config=ChainInitAndWarmupConfig(
               init_positions="normal", init_position_params={"mean": 0.0, "std": 0.5},
               warmup_steps=300, warmup_adapt_step_size=True))
report("A adam 20k       ", rA, time.time() - t0)

for M in (1024, 2048, 4096):
    if M <= 2048:
        walkers = jnp.asarray(rA.final_positions)[:M]
    else:
        walkers = jnp.tile(jnp.asarray(rA.final_positions), (M // 2048, 1))
    t0 = time.time()
    r = train(shape=(M, N), model=model(), hamiltonian=ham, coord_mode=LabCoords(),
              optimizer=optax.sgd(1e-3),
              init_params=rA.best_params(),
              training_config=TrainingConfig(
                  n_epochs=2000, rng_seed=0, warm_walkers=True,
                  is_update_step_size=True, use_qgt=True,
                  checkpoint_path=f"./probe8/sr_M{M}", print_summary=False),
              qgt_config=QGTConfig(solver="auto", learning_rate=1e-3,
                                   regularization=1e-2, max_state_change=0.1),
              sampler_params=dict(SAMPLER, step_size=rA.final_step_size),
              initial_chain_config=ChainInitAndWarmupConfig(
                  init_positions=walkers, warmup_steps=300,
                  warmup_adapt_step_size=True))
    report(f"sr M={M} M/P={M / P:.2f}", r, time.time() - t0)
