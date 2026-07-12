"""Which guard binds in sr_train — Euclidean grad clip vs Fisher trust region?

Proof run 3 left SR stable but over-damped (finetune holds, doesn't descend).
Suspects, in order: (a) grad_clip_norm=10 binds before the trust region,
(b) learning_rate=1e-3 too small (total state-change budget), (c) reg=1e-2 over-damps.

Stage A: adam_train from scratch (same as proof_run.py).
Then, from the SAME stage-A endpoint, 600-epoch continuations:
  adam-cont : Adam control — the descent rate SR must at least match
  sr-base   : sr_train defaults (lr 1e-3, clip 10, max_state_change 0.1, reg 1e-2)
  sr-noclip : grad_clip_norm=None                       → tests suspect (a)
  sr-lr1e-2 : learning_rate 1e-2 (same max_state_change) → tests suspect (b)
  sr-reg1e-3: regularization 1e-3                        → tests suspect (c)

Per-epoch guard diagnostics (new): fisher_norm (pre-trust √δᵀSδ), trust_scale
(<1 ⇒ trust region bound), nat_grad_norm (Euclid, pre-clip; > clip ⇒ clip bound),
solve_ok (0 ⇒ zero-step epoch).
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


def report(tag, result, t, clip=None):
    h = result.history
    e = h.get("energy")
    lam = float(result.final_params["params"]["jastrow"]["lambda"])
    alpha = float(result.final_params["params"]["envelope"]["alpha"])
    n = len(e)
    tail = e[-100:]
    # descent: robust slope over the run (median of per-epoch diffs is ~0; use ends)
    head = e[:100]
    print(
        f"[{tag}] {n} ep in {t:.1f}s ({n / t:.1f} it/s) | "
        f"E head100 {head.mean():.3f}±{head.std():.3f} → tail100 {tail.mean():.3f}±{tail.std():.3f} "
        f"(exact {E0}) | finite {np.all(np.isfinite(e))} | lam {lam:.4f} alpha {alpha:.4f}",
        flush=True,
    )
    if "fisher_norm" in h[0].keys():
        fn = h.get("fisher_norm")
        ts = h.get("trust_scale")
        gn = h.get("nat_grad_norm")
        ok = h.get("solve_ok")
        trust_binds = float(np.mean(ts < 0.999))
        clip_binds = float(np.mean(gn > clip)) if clip else 0.0
        both = float(np.mean((ts < 0.999) & (gn > clip))) if clip else 0.0
        print(
            f"        fisher_norm med {np.median(fn):.3g} p90 {np.percentile(fn, 90):.3g} max {fn.max():.3g} | "
            f"natgrad_norm med {np.median(gn):.3g} p90 {np.percentile(gn, 90):.3g} max {gn.max():.3g}",
            flush=True,
        )
        print(
            f"        trust binds {trust_binds:.1%} | clip({clip}) binds {clip_binds:.1%} "
            f"| both {both:.1%} | failed solves {int((1 - ok).sum())}",
            flush=True,
        )
    return e


t0 = time.time()
rA = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
           **adam_train(n_epochs=3000, learning_rate=1e-2,
                        checkpoint_path="./probe/adam"))
report("A adam-scratch", rA, time.time() - t0)

EP = 600
variants = {
    "adam-cont ": (adam_train(n_epochs=EP, learning_rate=1e-3, prev_result=rA,
                              checkpoint_path="./probe/adam_cont"), None),
    "sr-base   ": (sr_train(n_epochs=EP, prev_result=rA,
                            checkpoint_path="./probe/sr_base"), 10.0),
    "sr-noclip ": (sr_train(n_epochs=EP, prev_result=rA, grad_clip_norm=None,
                            checkpoint_path="./probe/sr_noclip"), None),
    "sr-lr1e-2 ": (sr_train(n_epochs=EP, prev_result=rA, learning_rate=1e-2,
                            checkpoint_path="./probe/sr_lr"), 10.0),
    "sr-reg1e-3": (sr_train(n_epochs=EP, prev_result=rA, regularization=1e-3,
                            checkpoint_path="./probe/sr_reg"), 10.0),
}

for tag, (kwargs, clip) in variants.items():
    t0 = time.time()
    r = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
              **kwargs)
    report(tag, r, time.time() - t0, clip=clip)
