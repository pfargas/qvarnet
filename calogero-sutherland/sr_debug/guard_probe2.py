"""Probe 2: with the clip gone (new defaults), does SR descend below Adam?

While the trust region binds, the update is max_state_change · δ/‖δ‖_S regardless of
learning_rate — η only matters once ‖δ‖_S falls under Δ = max_state_change/η. So the
real knobs are max_state_change (per-epoch Fisher budget) and η (end-game step).

Stage A: adam_train 3000 from scratch (same as guard_probe.py, same seed).
Continuations, 2000 epochs each from the same endpoint:
  adam-cont     : control (lr 1e-3)
  sr-msc0.1     : sr_train defaults (η 1e-3, Δ_state 0.1, reg 1e-2, no clip)
  sr-msc0.1-lr2 : η 1e-2, Δ_state 0.1 — same while trust binds, 10× bigger end-game
  sr-msc0.3     : η 1e-3, Δ_state 0.3 — 3× the per-epoch Fisher budget
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
        jastrow=LogJastrow(n_particles=N, lambda_init=CS_L),
    )


ham = CalogeroSutherlandHamiltonian(L=CS_L, epsilon=1e-12)


def report(tag, result, t):
    h = result.history
    e = h.get("energy")
    lam = float(result.final_params["params"]["jastrow"]["lambda"])
    alpha = float(result.final_params["params"]["envelope"]["alpha"])
    n = len(e)
    q = n // 4
    quarters = " → ".join(f"{e[i * q:(i + 1) * q].mean():.2f}" for i in range(4))
    tail = e[-200:]
    print(
        f"[{tag}] {n} ep in {t:.1f}s | quarters {quarters} | "
        f"tail200 {tail.mean():.3f}±{tail.std():.3f} (exact {E0}) | "
        f"finite {np.all(np.isfinite(e))} | lam {lam:.4f} alpha {alpha:.4f}",
        flush=True,
    )
    if "fisher_norm" in h[0].keys():
        ts = h.get("trust_scale")
        ok = h.get("solve_ok")
        # binding fraction per quarter — does the trust region release near convergence?
        binds = [float(np.mean(ts[i * q:(i + 1) * q] < 0.999)) for i in range(4)]
        print(
            f"        trust binds/quarter {' '.join(f'{b:.0%}' for b in binds)} | "
            f"failed solves {int((1 - ok).sum())}",
            flush=True,
        )


t0 = time.time()
rA = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
           **adam_train(n_epochs=3000, learning_rate=1e-2,
                        checkpoint_path="./probe2/adam"))
report("A adam-scratch", rA, time.time() - t0)

EP = 2000
variants = {
    "adam-cont    ": adam_train(n_epochs=EP, learning_rate=1e-3, prev_result=rA,
                                checkpoint_path="./probe2/adam_cont"),
    "sr-msc0.1    ": sr_train(n_epochs=EP, prev_result=rA,
                              checkpoint_path="./probe2/sr01"),
    "sr-msc0.1-lr2": sr_train(n_epochs=EP, prev_result=rA, learning_rate=1e-2,
                              checkpoint_path="./probe2/sr01lr2"),
    "sr-msc0.3    ": sr_train(n_epochs=EP, prev_result=rA, max_state_change=0.3,
                              checkpoint_path="./probe2/sr03"),
}

for tag, kwargs in variants.items():
    t0 = time.time()
    r = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
              **kwargs)
    report(tag, r, time.time() - t0)
