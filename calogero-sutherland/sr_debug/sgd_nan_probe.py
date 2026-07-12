"""Instrumented bare-SGD reproduction of the CS NaN.

Mirrors cs-single-point-exploration.ipynb (N=30, L=0.8, mlp_jastrow) but drives the
training loop manually so we can log, per epoch, which quantity explodes first:
E_loc tails, min pair distance, per-group grads (lambda / alpha / network), and the
actual SGD update magnitudes.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qvarnet.boundaries import NoBoundary
from qvarnet.config.coord_mode import LabCoords
from qvarnet.hamiltonian.continuous import CalogeroSutherlandHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.envelopes import GaussianEnvelope
from qvarnet.models.jastrow import LogJastrow
from qvarnet.models.mlp import MLP
from qvarnet.samplers import sample_and_process
from qvarnet.vmc.probability import build_prob_fn
from qvarnet.vmc.training_step import energy_and_grads

N, L, EPS = 30, 0.8, 1e-12
N_CHAINS, DOF = 4096, 30
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
N_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
LAMBDA_INIT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
SEED = 0

model = LogWavefunction(
    transform=NoBoundary(),
    network=MLP(hidden=[128]),
    envelope=GaussianEnvelope(),
    jastrow=LogJastrow(n_particles=N, lambda_init=LAMBDA_INIT),
)
ham = CalogeroSutherlandHamiltonian(L=L, epsilon=EPS).replace(coord_mode=LabCoords())
E_exact = N * (1 + L * (N - 1))

key = jax.random.PRNGKey(SEED)
params = model.init(key, jnp.ones((N_CHAINS, DOF)))
prob_fn = build_prob_fn(model.apply)

CLIP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # 0 = no update-norm control
if CLIP > 0:
    tx = optax.chain(optax.clip_by_global_norm(CLIP), optax.sgd(LR))
else:
    tx = optax.sgd(LR)
opt_state = tx.init(params)

# --- walkers: same init + warmup as the notebook (normal(0,0.5), 300 steps @ 0.5)
pos = jax.random.normal(key, (N_CHAINS, DOF)) * 0.5
pos = sample_and_process(
    key=key, prob_fn=prob_fn, prob_params=params, init_positions=pos,
    step_size=0.5, n_chains=N_CHAINS, dof=DOF, n_steps=300, burn_in=299,
    thinning=1, block_size=0, box_L=0.0,
)[1]

step_size = 0.5


@jax.jit
def diagnostics(batch, E_loc, grads, params):
    # min pairwise distance across the whole batch + the E_loc tail
    diffs = batch[:, :, None] - batch[:, None, :]                # (B, N, N)
    iu = jnp.triu_indices(DOF, k=1)
    pair_d = jnp.abs(diffs[:, iu[0], iu[1]])                     # (B, n_pairs)
    min_d_per_sample = jnp.min(pair_d, axis=1)
    p = params["params"]
    g = grads["params"]
    net_g = jax.flatten_util.ravel_pytree(g["network"])[0]
    net_p = jax.flatten_util.ravel_pytree(p["network"])[0]
    return dict(
        min_dist=jnp.min(min_d_per_sample),
        argmin_sample_min_d=jnp.argmin(min_d_per_sample),
        eloc_min=jnp.min(E_loc), eloc_max=jnp.max(E_loc),
        eloc_absmax_idx=jnp.argmax(jnp.abs(E_loc)),
        lam=p["jastrow"]["lambda"], lam_g=g["jastrow"]["lambda"],
        alpha=p["envelope"]["alpha"], alpha_g=g["envelope"]["alpha"],
        net_norm=jnp.linalg.norm(net_p), net_gnorm=jnp.linalg.norm(net_g),
        net_gmax=jnp.max(jnp.abs(net_g)),
    )


print(f"bare SGD lr={LR}  lam_init={LAMBDA_INIT}  N={N} L={L}  exact E0={E_exact:.3f}", flush=True)
hdr = ("ep      E          sigma      acc    step   min_d     eloc_min      eloc_max     "
       "lam      lam_g       alpha    alpha_g     |net|    |g_net|    max|g_net|")
print(hdr, flush=True)

rows = []
for ep in range(N_EPOCHS):
    key, skey, lkey = jax.random.split(key, 3)
    batch, pos, acc = sample_and_process(
        key=skey, prob_fn=prob_fn, prob_params=params, init_positions=pos,
        step_size=step_size, n_chains=N_CHAINS, dof=DOF, n_steps=21, burn_in=20,
        thinning=1, block_size=0, box_L=0.0,
    )
    E, sigma, E_loc, grads = energy_and_grads(ham, params, batch, model.apply, key=lkey)
    d = diagnostics(batch, E_loc, grads, params)
    acc_m = float(jnp.mean(acc))

    vals = {k: float(v) for k, v in jax.device_get(d).items()}
    E_v, s_v = float(E), float(sigma)
    print(f"{ep:3d} {E_v:12.4f} {s_v:10.4f} {acc_m:6.3f} {float(step_size):6.3f} "
          f"{vals['min_dist']:.2e} {vals['eloc_min']:13.4e} {vals['eloc_max']:13.4e} "
          f"{vals['lam']:8.4f} {vals['lam_g']:11.4e} {vals['alpha']:8.4f} "
          f"{vals['alpha_g']:10.3e} {vals['net_norm']:8.3f} {vals['net_gnorm']:10.3e} "
          f"{vals['net_gmax']:10.3e}", flush=True)
    rows.append(dict(ep=ep, E=E_v, sigma=s_v, acc=acc_m, step=float(step_size), **vals))

    if not np.isfinite(E_v):
        print(f"--- NaN/inf energy at epoch {ep}; stopping ---", flush=True)
        break

    # step-size adaptation, identical to train()'s _update_step_size
    step_size = float(np.clip(step_size * (1.0 + 0.1 * (acc_m - 0.5)), 1e-5, 5.0))

    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

import csv, os
out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"sgd_probe_lr{LR}_lam{LAMBDA_INIT}.csv"
)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("saved", out, flush=True)
