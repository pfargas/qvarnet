"""Why does SR-from-scratch CLIMB? (notebook cell 22 rerun: 2477 → 7050 over 5000 ep)

Guard-probe 2 validated max_state_change=0.3 for FINETUNE only; the scratch path was
never revalidated after the clip removal. Here: scratch SR at the notebook's exact
settings (4096 chains, lambda_init=L), msc 0.3 vs 0.1, tracking E, lambda, alpha and
the guard stats per quarter — plus an Adam-scratch control at the same chain count.

The suspect mechanism: from the wide init (alpha=0.1) the walkers lag behind the
state; if SR keeps widening the envelope (alpha falling), E rises without bound and
the walkers never equilibrate — a feedback Adam escapes but natural-gradient
step-allocation might not.
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
from qvarnet.callbacks import SnapshotCallback
from qvarnet.recipes import adam_train, sr_train
from qvarnet.train import train

N, CS_L = 30, 0.8
N_CHAINS = 4096  # notebook value — also flips auto solver: P=4099 > M=4096 → minSR
SHAPE = (N_CHAINS, N)
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
    n = len(e)
    q = n // 4
    lam_traj = np.array(
        [float(s["params"]["params"]["jastrow"]["lambda"]) for s in result.snapshots]
    )
    alp_traj = np.array(
        [float(s["params"]["params"]["envelope"]["alpha"]) for s in result.snapshots]
    )
    quarters = " → ".join(f"{e[i * q:(i + 1) * q].mean():.1f}" for i in range(4))
    print(
        f"[{tag}] {n} ep in {t:.0f}s | E {quarters} | tail100 {e[-100:].mean():.2f}"
        f"±{e[-100:].std():.2f} (exact {E0})",
        flush=True,
    )
    # trajectory every n/8 epochs
    idx = np.linspace(0, len(lam_traj) - 1, 9).astype(int)
    print(f"        lam   {' '.join(f'{lam_traj[i]:.3f}' for i in idx)}", flush=True)
    print(f"        alpha {' '.join(f'{alp_traj[i]:.3f}' for i in idx)}", flush=True)
    if "fisher_norm" in h[0].keys():
        ts, ok = h.get("trust_scale"), h.get("solve_ok")
        binds = [float(np.mean(ts[i * q:(i + 1) * q] < 0.999)) for i in range(4)]
        print(
            f"        trust binds/quarter {' '.join(f'{b:.0%}' for b in binds)} | "
            f"failed {int((1 - ok).sum())}",
            flush=True,
        )


EP = 1500
runs = {
    "adam-scratch": adam_train(n_epochs=EP, learning_rate=1e-2,
                               checkpoint_path="./probe3/adam"),
    "sr-msc0.3   ": sr_train(n_epochs=EP, max_state_change=0.3,
                             checkpoint_path="./probe3/sr03"),
    "sr-msc0.1   ": sr_train(n_epochs=EP, checkpoint_path="./probe3/sr01"),
}

for tag, kwargs in runs.items():
    t0 = time.time()
    r = train(shape=SHAPE, model=model(), hamiltonian=ham, coord_mode=LabCoords(),
              callbacks=[SnapshotCallback(policy="all", k=EP, metric="std")],
              **kwargs)
    report(tag, r, time.time() - t0)
