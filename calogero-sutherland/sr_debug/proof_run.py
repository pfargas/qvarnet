"""End-to-end proof: the stabilised SR stack on the real CS N=30 problem.

Stage A: adam_train from scratch.  Stage B: sr_train warm-started from A.
Stage C: sr_train from scratch (stability check — slow is fine, NaN is failure).
"""

import time

import numpy as np

from qvarnet.boundaries import NoBoundary
from qvarnet.config.coord_mode import LabCoords
from qvarnet.hamiltonian.continuous import CalogeroSutherlandHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.envelopes import GaussianEnvelope
from qvarnet.models.jastrow import LogJastrow
from qvarnet.models.mlp import MLP
from qvarnet.recipes import adam_train, sr_train
from qvarnet.train import train

N, CS_L = 30, 0.8
N_CHAINS = 2048
SHAPE = (N_CHAINS, N)
E0 = N * (1 + CS_L * (N - 1))  # 726


def model():
    return LogWavefunction(
        transform=NoBoundary(),
        network=MLP(hidden=[128]),
        envelope=GaussianEnvelope(),
        jastrow=LogJastrow(n_particles=N, lambda_init=CS_L),  # cusp-exact init
    )


ham = CalogeroSutherlandHamiltonian(L=CS_L, epsilon=1e-12)


def report(tag, result, t):
    e = np.array([float(s.energy) for s in result.history])
    tail = e[-50:]
    lam = float(result.final_params["params"]["jastrow"]["lambda"])
    alpha = float(result.final_params["params"]["envelope"]["alpha"])
    print(
        f"[{tag}] {len(e)} epochs in {t:.1f}s ({len(e) / t:.1f} it/s) | "
        f"E_tail = {tail.mean():.3f} ± {tail.std():.3f} (exact {E0}) | "
        f"finite: {np.all(np.isfinite(e))} | lam={lam:.4f} alpha={alpha:.4f} | "
        f"final step={result.final_step_size:.4f}",
        flush=True,
    )
    return e


t0 = time.time()
rA = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
           **adam_train(n_epochs=3000, learning_rate=1e-2,
                        checkpoint_path="./proof/adam"))
eA = report("A adam-scratch", rA, time.time() - t0)

t0 = time.time()
rB = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
           **sr_train(n_epochs=500, prev_result=rA, checkpoint_path="./proof/sr"))
eB = report("B sr-finetune ", rB, time.time() - t0)

t0 = time.time()
rC = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
           **sr_train(n_epochs=500, checkpoint_path="./proof/sr_scratch"))
eC = report("C sr-scratch  ", rC, time.time() - t0)

print(f"\nA last-10: {np.round(eA[-10:], 3)}")
print(f"B last-10: {np.round(eB[-10:], 3)}")
print(f"C last-10: {np.round(eC[-10:], 3)}")
