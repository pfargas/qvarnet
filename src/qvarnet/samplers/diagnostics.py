"""MCMC diagnostics — autocorrelation and effective sample size.

All functions accept JAX arrays and are JIT-compilable.  They operate on
**raw chain histories** (before thinning), which you can get from `mh_chain`
directly or by running a short diagnostic chain.

Typical usage::

    from qvarnet.samplers import mh_chain, integrated_autocorr_time, chain_stats
    import jax, jax.numpy as jnp

    # run one chain for n_steps steps
    rand = jax.random.normal(key, (n_steps, dof + 1))
    positions, _ = mh_chain(rand, prob_fn, params, init_pos, step_size)
    # positions: (n_steps, dof)

    tau = integrated_autocorr_time(positions[:, 0])   # IAT on first coord
    ess = effective_sample_size(positions[:, 0])

    # or summarise all chains at once
    taus, ess_all = chain_stats(all_positions)         # (n_chains, n_steps, dof)
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("max_lag",))
def autocorr(chain, max_lag=None):
    """Normalised autocorrelation function via FFT.

    chain:   (n_steps,) — 1-D time series of a real-valued observable
    max_lag: int (default n_steps // 4); number of lags to return

    returns: (max_lag,) with ρ_0 = 1, ρ_1, ..., ρ_{max_lag-1}
    """
    n = chain.shape[0]
    if max_lag is None:
        max_lag = n // 4
    x = chain - chain.mean()
    # Zero-pad to 2n to avoid circular wrap-around
    xf = jnp.fft.rfft(x, n=2 * n)
    acf_full = jnp.fft.irfft(xf * jnp.conj(xf), n=2 * n)[:n].real
    return acf_full[:max_lag] / (acf_full[0] + 1e-12)


@partial(jax.jit, static_argnames=("max_lag",))
def integrated_autocorr_time(chain, max_lag=None):
    """Integrated autocorrelation time (IAT).

    Uses a fixed-window estimator:
        τ_int = 1 + 2 Σ_{t=1}^{max_lag} ρ_t

    A good rule of thumb: thinning by τ_int produces near-independent samples.

    chain:   (n_steps,)
    max_lag: int (default n_steps // 4)

    returns: scalar τ_int ≥ 1
    """
    acf = autocorr(chain, max_lag=max_lag)
    return 1.0 + 2.0 * jnp.sum(acf[1:])


@partial(jax.jit, static_argnames=("max_lag",))
def effective_sample_size(chain, max_lag=None):
    """Effective sample size = n_steps / τ_int.

    chain:   (n_steps,)
    max_lag: int (default n_steps // 4)

    returns: scalar ESS
    """
    n = chain.shape[0]
    tau = integrated_autocorr_time(chain, max_lag=max_lag)
    return n / tau


@partial(jax.jit, static_argnames=("max_lag",))
def chain_stats(chains, max_lag=None):
    """Per-chain IAT and ESS, averaged over all coordinates (DoF).

    chains:  (n_chains, n_steps, dof) — raw chain positions (before thinning)
    max_lag: int (default n_steps // 4)

    returns:
        taus: (n_chains,) — mean IAT across coordinates per chain
        ess:  (n_chains,) — mean ESS across coordinates per chain
    """
    n_steps = chains.shape[1]

    def per_coord(x):          # (n_steps,) -> scalar
        return integrated_autocorr_time(x, max_lag=max_lag)

    def per_chain(chain):      # (n_steps, dof) -> scalar
        # vmap over the dof axis (axis=1 of chain, i.e. in_axes=1)
        return jnp.mean(jax.vmap(per_coord, in_axes=1)(chain))

    taus = jax.vmap(per_chain)(chains)   # (n_chains,)
    ess = n_steps / taus
    return taus, ess
