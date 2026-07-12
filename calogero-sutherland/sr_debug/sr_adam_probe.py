"""SR-preconditioned Adam vs classic SR (SGD), now that train() honours the optimizer.

From the same 3000-epoch Adam endpoint, 1000-epoch finetunes:
  sr-sgd      : sr_train defaults — classic SR (control)
  sr-adam1e-3 : same qgt_config, optimizer overridden to optax.adam(1e-3)
  sr-adam1e-4 : optax.adam(1e-4)
"""

import time

import numpy as np
import optax

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
SHAPE = (2048, N)
E0 = N * (1 + CS_L * (N - 1))


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
    n = len(e)
    q = n // 4
    quarters = " → ".join(f"{e[i * q:(i + 1) * q].mean():.2f}" for i in range(4))
    print(
        f"[{tag}] {n} ep in {t:.0f}s | E {quarters} | tail100 {e[-100:].mean():.2f}"
        f"±{e[-100:].std():.2f} (exact {E0}) | finite {np.all(np.isfinite(e))} | lam {lam:.4f}",
        flush=True,
    )


t0 = time.time()
rA = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
           **adam_train(n_epochs=3000, learning_rate=1e-2,
                        checkpoint_path="./probe4/adam"))
report("A adam-scratch", rA, time.time() - t0)

EP = 1000
for tag, opt in [("sr-sgd     ", None),
                 ("sr-adam1e-3", optax.adam(1e-3)),
                 ("sr-adam1e-4", optax.adam(1e-4))]:
    kwargs = sr_train(n_epochs=EP, prev_result=rA,
                      checkpoint_path=f"./probe4/{tag.strip()}")
    if opt is not None:
        kwargs["optimizer"] = opt  # SR-preconditioned Adam: honoured by train() now
    t0 = time.time()
    r = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
              **kwargs)
    report(tag, r, time.time() - t0)
