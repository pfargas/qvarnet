# MCMC diagnostics: autocorrelation, IAT, and ESS — from the ground up

This document explains the diagnostics computed in `src/qvarnet/samplers/diagnostics.py`
(`autocorr`, `integrated_autocorr_time`, `effective_sample_size`, `chain_stats`), and
how they're used in `TrainedWavefunction.sample()` to decide how much to thin the raw
Metropolis-Hastings chains before using them for estimators.

## 1. The problem: MCMC samples are correlated, not independent

MCMC gives you a sequence of samples $x_1, x_2, \dots, x_n$ drawn (asymptotically) from
your target distribution $|\psi|^2$. But consecutive samples are **not independent** —
each Metropolis-Hastings step is a small perturbation of the previous configuration, so
neighboring samples in the chain are correlated with each other.

If you naively treat all $n$ samples as independent draws when estimating an observable
(energy, density, pair correlation, etc.), you will **underestimate your statistical
error**, because the true amount of independent information in the chain is less than
$n$. The diagnostics in this file answer the question: *how many truly independent
samples is this batch of $n$ correlated samples worth?*

## 2. The autocorrelation function (ACF)

For a scalar time series (think: one coordinate of one particle, tracked over MCMC
steps), the lag-$k$ autocorrelation is defined as

$$
\rho_k \;=\; \frac{\operatorname{Cov}(x_t, x_{t+k})}{\operatorname{Var}(x_t)}
\;=\; \frac{\mathbb{E}\big[(x_t-\mu)(x_{t+k}-\mu)\big]}{\sigma^2}
$$

where $\mu = \mathbb{E}[x_t]$ and $\sigma^2 = \operatorname{Var}(x_t)$ (assuming the chain
is stationary, i.e. these don't depend on $t$).

By definition, $\rho_0 = 1$ (a sample is perfectly correlated with itself). For a
well-mixing chain, $\rho_k \to 0$ as $k$ grows — samples far apart in the chain become
statistically independent. For a poorly-mixing chain (too small a step size, a sticky
region of configuration space, etc.), $\rho_k$ decays slowly, meaning you need many
steps before two samples are effectively independent.

### How `autocorr()` computes this

Computing the ACF directly from its definition is an $O(n^2)$ operation (a sum over all
pairs of time points, for every lag $k$). Instead, `autocorr()` uses the
**Wiener–Khinchin theorem**: for a stationary process, the autocovariance sequence is
the inverse Fourier transform of the power spectral density $|\hat{x}(\omega)|^2$. This
lets the whole ACF be computed in $O(n \log n)$ via FFT.

```python
def autocorr(chain, max_lag=None):
    n = chain.shape[0]
    if max_lag is None:
        max_lag = n // 4
    x = chain - chain.mean()
    xf = jnp.fft.rfft(x, n=2 * n)
    acf_full = jnp.fft.irfft(xf * jnp.conj(xf), n=2 * n)[:n].real
    return acf_full[:max_lag] / (acf_full[0] + 1e-12)
```

Step by step:

- **`x = chain - chain.mean()`** — center the series. Autocovariance is defined about
  the mean, so this subtraction has to happen before anything else.

- **`xf = jnp.fft.rfft(x, n=2 * n)`** — zero-pad the series to twice its length before
  taking the FFT. This matters because FFT-based multiplication computes a *circular*
  (periodic, wrap-around) correlation, not the *linear* correlation we actually want.
  Padding with zeros to at least double the original length ensures the circular
  correlation the FFT computes agrees with the linear correlation for every lag we care
  about — without the padding, the correlation at lag $k$ would spuriously include
  contributions from the far end of the series wrapping around to meet the near end.

- **`xf * jnp.conj(xf)`** — this is the power spectrum, $|\hat{x}(\omega)|^2$.

- **`jnp.fft.irfft(..., n=2*n)[:n]`** — the inverse FFT of the power spectrum gives the
  (unnormalized) autocovariance at each lag — this is the Wiener–Khinchin step, and it's
  what makes the whole computation $O(n \log n)$ instead of $O(n^2)$. Only the first $n$
  entries are kept (the rest, from the zero-padding, are not meaningful lags).

- **`/ (acf_full[0] + 1e-12)`** — `acf_full[0]` is the lag-0 autocovariance, i.e. just
  $n\sigma^2$. Dividing by it converts raw autocovariance into a normalized
  autocorrelation, so that $\rho_0 = 1$ and $\rho_k \in [-1, 1]$ for $k > 0$. The
  `1e-12` is a numerical safety epsilon, guarding against division by zero if a chain
  is completely frozen ($\sigma^2 = 0$, e.g. a walker that never accepted a move).

- **`max_lag = n // 4` (default)** — the sum is truncated to the first `max_lag` lags
  rather than going out to $n-1$. This matters because the estimate of $\rho_k$ uses
  only the $n-k$ overlapping pairs available at that lag, so it gets noisier as $k$
  approaches $n$. Summing all the way out would just accumulate noise from lags where
  the estimate is unreliable. `n/4` is a standard fixed-window heuristic. (More
  sophisticated approaches exist — e.g. Geyer's initial-monotone-sequence estimator,
  which adaptively picks the window — but this codebase uses the simpler fixed-window
  version, as noted in its own docstring.)

## 3. Integrated autocorrelation time (IAT), $\tau_{\text{int}}$

```python
def integrated_autocorr_time(chain, max_lag=None):
    acf = autocorr(chain, max_lag=max_lag)
    return 1.0 + 2.0 * jnp.sum(acf[1:])
```

$$
\tau_{\text{int}} \;=\; 1 \;+\; 2\sum_{k=1}^{\text{max\_lag}} \rho_k
$$

### Where this formula comes from

Consider the sample mean of the chain, $\bar{x} = \frac{1}{n}\sum_{t=1}^n x_t$. For an
**independent** sample, $\operatorname{Var}(\bar x) = \sigma^2/n$. But for a
**correlated** series, the variance of the sum picks up cross terms:

$$
\operatorname{Var}(\bar x) \;=\; \frac{1}{n^2}\sum_{t=1}^n\sum_{s=1}^n \operatorname{Cov}(x_t, x_s)
$$

Grouping this double sum by lag $k = s - t$, and using stationarity
($\operatorname{Cov}(x_t, x_{t+k})$ doesn't depend on $t$), this becomes, for $n$ large
relative to the correlation length:

$$
\operatorname{Var}(\bar x) \;\approx\; \frac{\sigma^2}{n}\left(1 + 2\sum_{k=1}^{\infty}\rho_k\right) \;=\; \frac{\sigma^2}{n}\,\tau_{\text{int}}
$$

Two things to note about the formula's structure:

- The **"1"** is the $k=0$ diagonal term, where $\rho_0 = 1$ by definition.
- The **"2"** accounts for the fact that both $+k$ and $-k$ off-diagonal terms
  contribute equally (a stationary process has $\operatorname{Cov}(x_t, x_{t+k}) =
  \operatorname{Cov}(x_t, x_{t-k})$), so instead of summing over all nonzero integer
  lags separately, we sum over positive lags once and double it.

### What $\tau_{\text{int}}$ means intuitively

Compare to the i.i.d. case: if the samples were independent, $\rho_k = 0$ for all
$k \geq 1$, so $\tau_{\text{int}} = 1$, and $\operatorname{Var}(\bar x) = \sigma^2/n$
exactly — the textbook formula.

Whenever $\tau_{\text{int}} > 1$, the correlated chain's sample mean is *noisier* than
an i.i.d. sample of the same size $n$ would be — specifically, by a factor of
$\tau_{\text{int}}$ in variance. It's as if you had only drawn $n / \tau_{\text{int}}$
independent samples instead of $n$ correlated ones. That ratio is exactly the effective
sample size, defined next.

$\tau_{\text{int}}$ can also be read directly as a **timescale**: it's roughly the
number of MCMC steps you need to advance before a sample becomes statistically
independent of where you started. That's why it's the natural thinning interval (see
§5 below).

## 4. Effective sample size (ESS)

```python
def effective_sample_size(chain, max_lag=None):
    n = chain.shape[0]
    tau = integrated_autocorr_time(chain, max_lag=max_lag)
    return n / tau
```

$$
\text{ESS} \;=\; \frac{n}{\tau_{\text{int}}}
$$

This is literally the number of independent samples that would give an estimator the
same variance as your $n$ correlated ones. It's the number you'd plug into a Monte
Carlo standard error formula,

$$
\text{SE}(\bar x) \;=\; \frac{\sigma}{\sqrt{\text{ESS}}}
$$

instead of the (wrong, too optimistic) $\sigma/\sqrt{n}$.

## 5. Aggregating over chains and coordinates: `chain_stats`

```python
def chain_stats(chains, max_lag=None):
    n_steps = chains.shape[1]

    def per_coord(x):          # (n_steps,) -> scalar
        return integrated_autocorr_time(x, max_lag=max_lag)

    def per_chain(chain):      # (n_steps, dof) -> scalar
        return jnp.mean(jax.vmap(per_coord, in_axes=1)(chain))

    taus = jax.vmap(per_chain)(chains)   # (n_chains,)
    ess = n_steps / taus
    return taus, ess
```

`chains` here is `(n_chains, n_steps, dof)` — the raw, per-chain, per-step positions
(all particle coordinates), *before* burn-in cropping/thinning/flattening.

- `per_coord`: computes $\tau_{\text{int}}$ for a single coordinate's time series
  (e.g. particle 3's x-position over the whole chain).
- `per_chain`: vmaps `per_coord` over all degrees of freedom (`in_axes=1`, i.e. over the
  `dof` axis), then averages the resulting per-coordinate $\tau_{\text{int}}$ values —
  giving one scalar IAT per chain.
- The outer `jax.vmap(per_chain)` runs this over all chains, giving `taus`, one IAT
  per chain, shape `(n_chains,)`.
- `ess = n_steps / taus` then converts each chain's IAT into an ESS, using that
  chain's own `n_steps` (the number of steps in the array passed in — see §6, this is
  important for the post-thinning check).

In `TrainedWavefunction.sample()`, these per-chain values are further averaged across
chains (`jnp.mean(taus)`, `jnp.mean(ess)`) to report one summary IAT/ESS for the whole
sampling run.

## 6. Why we thin by IAT (and how to sanity-check it)

```python
thin = int(np.ceil(self._iat))
processed = cropped[:, ::thin, :]
```

Since $\rho_k \to 0$ roughly on the timescale of $\tau_{\text{int}}$, keeping only every
$\lceil \tau_{\text{int}} \rceil$-th sample gives a subsequence with $\rho_{\text{thin}}
\approx 0$ between consecutive retained samples — i.e., close to independent. That's the
assumption most downstream estimators (density, pair correlation, OBDM, condensate
fraction, ...) implicitly rely on.

**Self-consistency check.** If the thinning actually worked, then recomputing
`chain_stats` on the *thinned* array should give an IAT close to **1** — because by
construction, consecutive retained samples should now be roughly independent. And since
$\text{ESS} = n_{\text{steps}}/\tau_{\text{int}}$, an IAT of $\approx 1$ on the thinned
array means

$$
\text{ESS}_{\text{thinned}} \;\approx\; n_{\text{steps, thinned}}
$$

i.e. **the effective sample size of the thinned chain should be close to the actual
number of samples per chain after thinning.** This is the meaningful version of "ESS
should equal the number of samples" — it only holds *after* thinning by IAT, not on the
raw chain (where IAT is often well above 1, and that's expected, not a bug: a Gaussian
random-walk Metropolis proposal on a many-body wavefunction is inherently correlated
step-to-step).

This check is easy to add: call `chain_stats(processed)` (the post-thin, pre-flatten
`(n_chains, n_steps_thinned, dof)` array) right after computing `processed`, and verify
its IAT is close to 1 (e.g. `assert jnp.mean(taus_thinned) < 1.5` or similar, with some
tolerance since it's a noisy estimate on a shorter series). If it's *not* close to 1,
that's a sign the fixed-window ACF estimate was noisy, or `ceil(IAT)` under-thinned
because the mean IAT across chains/coordinates hid a slower-mixing outlier.
