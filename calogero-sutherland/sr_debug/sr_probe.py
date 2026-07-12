"""Instrumented SR (stochastic reconfiguration) probe for the CS model.

Reproduces notebook cell 22 (SR from scratch) but manually, logging per epoch:
energy, descent check <delta, F>, Fisher norm of the step sqrt(d^T S d), raw |d|,
lambda/alpha trajectories, and wall-time breakdown (sampling / grads / QGT).

usage: sr_probe.py LR N_EPOCHS LAMBDA_INIT TRUST [SOLVER]
  TRUST  = 0 -> plain SR step; >0 -> rescale step so sqrt(d^T S d) <= TRUST
"""

import os
import sys
import time

import jax

if os.environ.get("QVN_F64"):
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from qvarnet.boundaries import NoBoundary
from qvarnet.config.coord_mode import LabCoords
from qvarnet.geometry.qgt import QGTConfig, compute_log_derivatives, compute_qgt
from qvarnet.hamiltonian.continuous import CalogeroSutherlandHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.envelopes import GaussianEnvelope
from qvarnet.models.jastrow import LogJastrow
from qvarnet.models.mlp import MLP
from qvarnet.samplers import sample_and_process
from qvarnet.vmc.probability import build_prob_fn
from qvarnet.vmc.training_step import energy_and_grads

N, L, EPS = 30, 0.8, 1e-12
N_CHAINS, DOF = int(os.environ.get("QVN_CHAINS", 4096)), 30
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
N_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LAMBDA_INIT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
TRUST = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
SOLVER = sys.argv[5] if len(sys.argv) > 5 else "direct"
SEED = 0
REG = 1e-2

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
qcfg = QGTConfig(solver=SOLVER, learning_rate=LR, regularization=REG)

pos = jax.random.normal(key, (N_CHAINS, DOF)) * 0.5
pos = sample_and_process(
    key=key, prob_fn=prob_fn, prob_params=params, init_positions=pos,
    step_size=0.5, n_chains=N_CHAINS, dof=DOF, n_steps=300, burn_in=299,
    thinning=1, block_size=0, box_L=0.0,
)[1]
step_size = 0.5


@jax.jit
def sr_step(params, batch, grads, e_loc):
    """One SR step with diagnostics. Returns (new_params, diag)."""
    flat_p, unravel = ravel_pytree(params)
    flat_g, _ = ravel_pytree(grads)
    if SOLVER == "minsr":
        # M×M Gram dual: d = (2/M)·Ō^T (T+εI)^{-1} e
        O = compute_log_derivatives(
            flat_p, batch, lambda p, x: model.apply(unravel(p), x)
        )
        M = O.shape[0]
        O_bar = O - jnp.mean(O, axis=0)
        e = e_loc - jnp.mean(e_loc)
        T = (O_bar @ O_bar.T) / M + REG * jnp.eye(M, dtype=O.dtype)
        y = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(T), e)
        d = (2.0 / M) * (O_bar.T @ y)
        dSd = jnp.sum((O_bar @ d) ** 2) / M + REG * jnp.sum(d**2)
        resid = jnp.float32(0.0)
    else:
        S, _ = compute_qgt(flat_p, batch, lambda p, x: model.apply(unravel(p), x), REG)
        if SOLVER == "cholesky":
            d = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(S), flat_g)
        else:
            d = jnp.linalg.solve(S, flat_g)
        dSd = d @ (S @ d)                  # Fisher norm^2 of the raw solve
        # residual of the linear solve (float32 quality check)
        resid = jnp.linalg.norm(S @ d - flat_g) / (jnp.linalg.norm(flat_g) + 1e-30)
    desc = d @ flat_g                      # descent check: must be > 0
    scale = 1.0
    if TRUST > 0:
        # trust region on the *update* delta = lr*d in the Fisher metric
        fisher_norm_update = LR * jnp.sqrt(jnp.maximum(dSd, 0.0))
        scale = jnp.minimum(1.0, TRUST / (fisher_norm_update + 1e-30))
    new_flat = flat_p - LR * scale * d
    return unravel(new_flat), dict(
        d_norm=jnp.linalg.norm(d), dSd=dSd, desc=desc, scale=scale,
        g_norm=jnp.linalg.norm(flat_g), resid=resid,
        lam=params["params"]["jastrow"]["lambda"],
        alpha=params["params"]["envelope"]["alpha"],
    )


print(f"SR lr={LR} lam_init={LAMBDA_INIT} trust={TRUST} solver={SOLVER} reg={REG}  "
      f"exact E0={E_exact:.3f}", flush=True)
print("ep      E          sigma      acc    step    |F|       |d|      sqrt(dSd)   "
      "d.F(>0)   resid    scale   lam     alpha   [t_samp t_grad t_sr ms]", flush=True)

t_samp = t_grad = t_sr = 0.0
for ep in range(N_EPOCHS):
    key, skey, lkey = jax.random.split(key, 3)
    t0 = time.perf_counter()
    batch, pos, acc = sample_and_process(
        key=skey, prob_fn=prob_fn, prob_params=params, init_positions=pos,
        step_size=step_size, n_chains=N_CHAINS, dof=DOF, n_steps=21, burn_in=20,
        thinning=1, block_size=0, box_L=0.0,
    )
    batch.block_until_ready(); t1 = time.perf_counter()
    E, sigma, E_loc, grads = energy_and_grads(ham, params, batch, model.apply, key=lkey)
    E.block_until_ready(); t2 = time.perf_counter()
    params, diag = sr_step(params, batch, grads, E_loc)
    jax.block_until_ready(params); t3 = time.perf_counter()
    t_samp, t_grad, t_sr = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3

    E_v, s_v, acc_m = float(E), float(sigma), float(jnp.mean(acc))
    dg = {k: float(v) for k, v in jax.device_get(diag).items()}
    if ep < 10 or ep % 10 == 0 or not np.isfinite(E_v):
        print(f"{ep:3d} {E_v:12.4f} {s_v:10.4f} {acc_m:6.3f} {step_size:6.3f} "
              f"{dg['g_norm']:9.2e} {dg['d_norm']:9.2e} {np.sqrt(max(dg['dSd'],0)):9.2e} "
              f"{dg['desc']:9.2e} {dg['resid']:8.1e} {dg['scale']:6.3f} "
              f"{dg['lam']:7.4f} {dg['alpha']:7.4f}  "
              f"[{t_samp:5.1f} {t_grad:5.1f} {t_sr:6.1f}]", flush=True)
    if not np.isfinite(E_v):
        print(f"--- NaN/inf at epoch {ep} ---", flush=True)
        break
    step_size = float(np.clip(step_size * (1.0 + 0.1 * (acc_m - 0.5)), 1e-5, 5.0))

print("done", flush=True)
