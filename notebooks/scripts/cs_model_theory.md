# Calogero–Sutherland Model: Exact Wavefunction, Variational Energy, and Monte Carlo

## 1. The Hamiltonian and Exact Ground State

### Code convention

Throughout, $\hbar^2/m = 1$ (so the kinetic operator is $-\sum_i \partial^2/\partial x_i^2$, not $-\frac{1}{2}\sum_i \partial^2/\partial x_i^2$). All particles are in 1D.

$$H = -\sum_i \frac{\partial^2}{\partial x_i^2} + \sum_i x_i^2 + 2L(L-1)\sum_{i<j} \frac{1}{(x_i - x_j)^2}$$

### Exact ground state

$$\psi_0(x) = C \prod_{i<j} |x_i - x_j|^L \cdot \exp\!\left(-\tfrac{1}{2}\sum_i x_i^2\right)$$

$$\log|\psi_0| = L \sum_{i<j} \log|x_i - x_j| - \tfrac{1}{2}\sum_i x_i^2$$

$$E_0 = N\bigl(1 + L(N-1)\bigr), \qquad E_0/N = 1 + L(N-1)$$

---

## 2. One-Parameter Family of Trial States

We parametrise a family of trial wavefunctions by a single Jastrow exponent $\lambda$:

$$\log|\psi_\lambda| = \lambda \sum_{i<j} \log|x_i - x_j| - \tfrac{1}{2}\sum_i x_i^2
\equiv \lambda\, J(x) - \tfrac{1}{2}B(x)$$

where $J(x) = \sum_{i<j}\log|x_i - x_j|$ and $B(x) = \sum_i x_i^2$.

The exact ground state corresponds to $\lambda = L$. The goal of training is to find this $\lambda$ by minimising $E_\text{var}(\lambda)$.

---

## 3. Derivation of the Local Energy

The local energy is $E_\text{loc}(x;\lambda) = H\psi_\lambda(x)/\psi_\lambda(x)$. For a log-model the CS kinetic energy is

$$T = -\!\left(\Delta\log|\psi_\lambda| + |\nabla\log|\psi_\lambda||^2\right)$$

(the extra factor of 2 relative to the standard convention is absorbed into the CS Hamiltonian coefficient).

### Step 1: First and second derivatives

$$\frac{\partial}{\partial x_k}\log|\psi_\lambda| = \lambda\sum_{j\neq k}\frac{1}{x_k - x_j} - x_k$$

$$\frac{\partial^2}{\partial x_k^2}\log|\psi_\lambda| = -\lambda\sum_{j\neq k}\frac{1}{(x_k - x_j)^2} - 1$$

### Step 2: Laplacian

$$\Delta\log|\psi_\lambda| = \sum_k \frac{\partial^2}{\partial x_k^2}\log|\psi_\lambda| = -2\lambda A(x) - N$$

where $A(x) = \sum_{i<j} \frac{1}{(x_i - x_j)^2}$ and we used $\sum_k\sum_{j\neq k}(x_k-x_j)^{-2} = 2A$.

### Step 3: Squared gradient

$$|\nabla\log|\psi_\lambda||^2 = \sum_k\!\left[\lambda\sum_{j\neq k}\frac{1}{x_k-x_j} - x_k\right]^2$$

**Three-body sum vanishes.** For three distinct indices $k,j,l$:
$$\frac{1}{(x_k-x_j)(x_k-x_l)} + \frac{1}{(x_j-x_k)(x_j-x_l)} + \frac{1}{(x_l-x_k)(x_l-x_j)} = 0$$
(partial fractions identity). Therefore $\sum_k\!\left[\sum_{j\neq k}\frac{1}{x_k-x_j}\right]^2 = 2A$.

**Cross term.** 
$$\sum_k x_k \sum_{j\neq k}\frac{1}{x_k-x_j} = \sum_{i<j}\left[\frac{x_i}{x_i-x_j}+\frac{x_j}{x_j-x_i}\right] = \sum_{i<j}1 = \frac{N(N-1)}{2}$$

Collecting:

$$|\nabla\log|\psi_\lambda||^2 = 2\lambda^2 A - \lambda N(N-1) + B$$

### Step 4: Kinetic energy

$$T = -\!\left(\Delta\log|\psi_\lambda| + |\nabla\log|\psi_\lambda||^2\right)
= -\!\left(-2\lambda A - N + 2\lambda^2 A - \lambda N(N-1) + B\right)$$

$$\boxed{T = 2\lambda(1-\lambda)\,A + N + \lambda N(N-1) - B}$$

### Step 5: Potential energy (Hamiltonian $H$ with coupling $L$)

$$V = \sum_i x_i^2 + 2L(L-1)\sum_{i<j}\frac{1}{(x_i-x_j)^2} = B + 2L(L-1)\,A$$

### Step 6: Local energy — the $B$ terms cancel

$$E_\text{loc}(x;\lambda) = T + V = 2\lambda(1-\lambda)A + N + \lambda N(N-1) \cancel{- B} + 2L(L-1)A + \cancel{B}$$

$$\boxed{E_\text{loc}(x;\lambda) = \bigl[2\lambda(1-\lambda) + 2L(L-1)\bigr]\,A(x) + N + \lambda N(N-1)}$$

The $\sum x_i^2$ dependence drops out entirely. The local energy is a **linear function of $A(x)$ only**.

**Sanity check at $\lambda = L$:**
$$2L(1-L) + 2L(L-1) = 0 \implies E_\text{loc} = N + LN(N-1) = N(1+L(N-1)) = E_0 \checkmark$$
The local energy is constant everywhere — zero variance, as expected for an exact eigenstate.

---

## 4. Analytical Variational Energy via Hellmann–Feynman

Since the local energy is linear in $A(x)$, the variational energy $E_\text{var}(\lambda) = \langle E_\text{loc}\rangle_\lambda$ only requires $\langle A\rangle_\lambda$.

### The trial Hamiltonian

The state $\psi_\lambda$ is the **exact ground state** of the trial Hamiltonian with matching coupling:

$$H(\lambda) = -\sum_i\partial_i^2 + \sum_i x_i^2 + 2\lambda(\lambda-1)\sum_{i<j}(x_i-x_j)^{-2}$$

with exact eigenvalue $E(\lambda) = N(1 + \lambda(N-1))$.

### Hellmann–Feynman theorem

$$\frac{dE(\lambda)}{d\lambda} = \left\langle\frac{dH(\lambda)}{d\lambda}\right\rangle_\lambda, \qquad \frac{dH(\lambda)}{d\lambda} = 2(2\lambda-1)\,A$$

$$N(N-1) = 2(2\lambda-1)\,\langle A\rangle_\lambda$$

$$\boxed{\langle A\rangle_\lambda = \frac{N(N-1)}{2(2\lambda-1)}, \qquad \lambda > \tfrac{1}{2}}$$

### Exact variational energy

$$\boxed{\frac{E_\text{var}(\lambda)}{N} = \frac{(N-1)\bigl[2\lambda(1-\lambda)+2L(L-1)\bigr]}{2(2\lambda-1)} + 1 + \lambda(N-1), \qquad \lambda > \tfrac{1}{2}}$$

This is an **exact, closed-form, noise-free** expression. No Monte Carlo needed.

**Checks:**
- At $\lambda = L$: numerator $= 2L(1-L)+2L(L-1) = 0$, so $E_\text{var}(L)/N = 1+L(N-1) = E_0/N$. ✓  
- At $\lambda = 1$ ($N=5, L=2$): $E_\text{var}(1)/N = 4\cdot\frac{20}{2}/5 + 5 = 13$. This is the fermionic ground state of $N$ independent harmonic oscillators. ✓  
- For all $\lambda > \tfrac{1}{2}$: $E_\text{var}(\lambda) \geq E_0$ (variational principle). ✓

---

## 5. The Monte Carlo Approach

Monte Carlo estimates $E_\text{var}(\lambda)$ by:

1. Sampling configurations $\{x^{(i)}\}$ from $|\psi_\lambda|^2$ via Metropolis–Hastings.
2. Evaluating $E_\text{loc}(x^{(i)};\lambda)$ at each sample using automatic differentiation through the model.
3. Reporting $\hat{E} = \frac{1}{N_\text{chains}}\sum_i E_\text{loc}(x^{(i)};\lambda)$.

This is statistically consistent with $E_\text{var}(\lambda)$ whenever the mean exists and the variance is finite.

The advantage over the analytical formula: MC works for **any** wavefunction ansatz, not just the analytic one.

---

## 6. Why Monte Carlo Fails for $\lambda < \tfrac{1}{2}$

### The singularity structure

Near a particle coincidence $r = x_i - x_j \to 0$:

$$|\psi_\lambda|^2 \sim r^{2\lambda} \quad\text{(Jastrow suppression)}$$
$$E_\text{loc} \sim \frac{C}{r^2}, \quad C = 2\lambda(1-\lambda) + 2L(L-1) \neq 0 \text{ for } \lambda \neq L$$

The contribution to the mean from this region:

$$\int_0^\epsilon |\psi_\lambda|^2\, E_\text{loc}\, dr \;\sim\; C\int_0^\epsilon r^{2\lambda-2}\,dr = \frac{C\,\epsilon^{2\lambda-1}}{2\lambda-1}$$

| $\lambda$ | Integral | Consequence |
|-----------|----------|-------------|
| $\lambda > \tfrac{1}{2}$ | Converges | $E_\text{var}(\lambda)$ finite, MC converges |
| $\lambda = \tfrac{1}{2}$ | $\sim \log\epsilon \to \infty$ | Mean diverges logarithmically |
| $\lambda < \tfrac{1}{2}$ | $\to \infty$ | Mean is $+\infty$ |

### What Monte Carlo actually does

Every individual sample gives a **finite** $E_\text{loc}(x;\lambda)$ — the formula is analytic and well-defined at any non-coincident configuration. But as more samples are drawn, the rare near-coincidence samples (weight $\sim r^{2\lambda}$, value $\sim 1/r^2$) contribute increasingly large values that push the running mean upward without bound. The huge standard deviation is the correct statistical signal: it indicates the estimator is tracking a divergent quantity.

This is not a numerical artefact. More samples, smaller $\varepsilon$, or better sampling cannot fix it — the underlying integral does not converge.

### Why the exact solution has no such problem

In $H(\lambda)$, the kinetic and potential singularities at coincidences cancel **identically**:

$$\underbrace{2\lambda(1-\lambda)}_{\text{kinetic}} + \underbrace{2\lambda(\lambda-1)}_{\text{potential of }H(\lambda)} = 0$$

So $E_\text{loc}(x;\lambda)$ with $H(\lambda)$ is a constant $= E(\lambda) = N(1+\lambda(N-1))$ with **zero variance**, for any configuration, for any $\lambda > 0$.

The energy scan uses $H(L)$ with fixed $L$, not $H(\lambda)$. The residual:

$$2\lambda(1-\lambda) + 2L(L-1) \neq 0 \quad (\lambda \neq L)$$

creates an uncompensated $1/r^2$ singularity. The integral diverges for $\lambda < \tfrac{1}{2}$.

### Summary table

| Configuration | What is computed | Finite for all $\lambda > 0$? |
|---|---|---|
| $\psi_\lambda$ in $H(\lambda)$, MC | $E(\lambda) = N(1+\lambda(N-1))$, zero variance | **Yes** |
| $\psi_\lambda$ in $H(L)$, analytical | $E_\text{var}(\lambda)$ via Hellmann–Feynman | Only for $\lambda > \tfrac{1}{2}$ |
| $\psi_\lambda$ in $H(L)$, MC | Same $E_\text{var}(\lambda)$, estimated stochastically | Only for $\lambda > \tfrac{1}{2}$ |

---

## 7. The Pole at $\lambda = \tfrac{1}{2}$

The Hellmann–Feynman formula $\langle A\rangle_\lambda = N(N-1)/[2(2\lambda-1)]$ diverges at $\lambda = \tfrac{1}{2}$.

Physically: the coupling $2\lambda(\lambda-1)$ at $\lambda=\tfrac{1}{2}$ equals $-\tfrac{1}{4}$, the critical value of the attractive inverse-square potential in 1D (the von Neumann threshold). Below this, the trial Hamiltonian $H(\lambda)$ loses a well-defined ground state in the standard $L^2$ sense, and the formal eigenvalue $E(\lambda) = N(1+\lambda(N-1))$ is no longer reliable. Consequently, the Hellmann–Feynman derivation of $\langle A\rangle_\lambda$ also breaks down.

This is the same threshold from a different angle: the integrability condition $\lambda > \tfrac{1}{2}$ and the pole in the analytical formula are the **same phenomenon**.

---

## 8. Implementation Notes

### Analytical curve (zero noise)
```python
lam = np.linspace(0.55, 2.0 * L, 400)   # avoid pole at λ = 0.5
A_mean     = N * (N - 1) / (2 * (2 * lam - 1))
E_analytic = (
    (2 * lam * (1 - lam) + 2 * L * (L - 1)) * A_mean
    + N + lam * N * (N - 1)
) / N
```

### Monte Carlo scan (valid for $\lambda > \tfrac{1}{2}$)
```python
for lv in lam_values:            # lam_values starting from ~0.6
    p = {'params': {'lam': jnp.array(float(lv))}}
    batch, x_scan, _ = sample_and_process(...)   # resample from |ψ_λ|²
    E_loc = hamiltonian.local_energy(p, batch, model.apply, is_log_model=True)
    E_scan[i] = float(jnp.mean(E_loc)) / N_PARTICLES
    # std grows and mean diverges for lv < 0.5 — expected, not a bug
```

### Exact zero-variance computation at any $\lambda$ (uses $H(\lambda)$, not $H(L)$)
```python
for lv in lam_values:            # works for ANY lv > 0
    p = {'params': {'lam': jnp.array(float(lv))}}
    ham_lv = CalogeroSutherlandHamiltonian(L=lv, epsilon=1e-12)
    ham_lv = ham_lv.replace(coord_mode=LabCoords())
    E_loc = ham_lv.local_energy(p, batch, model.apply, is_log_model=True)
    # E_loc is constant = N*(1 + lv*(N-1)), variance ≈ 0
```
This recovers the exact CS spectrum but is **not** the variational energy in $H(L)$.

---

## 9. How Can the Mean Diverge If Every Sample Is Finite?

This is the sharpest conceptual point. The confusion arises from conflating two separate questions:
- Is $\psi_\lambda$ normalizable? (Is the distribution valid?)
- Is $\mathbb{E}[E_\text{loc}]$ finite?

These are answered by **different integrals** with different convergence conditions.

### A concrete example with no physics

Take one particle with distribution $p(x) = 2x$ on $(0, 1)$. This is a perfectly valid distribution — it integrates to 1, every sample is a finite number in $(0,1)$.

Ask: what is $\mathbb{E}[1/x^2]$?

$$\mathbb{E}\!\left[\frac{1}{x^2}\right] = \int_0^1 2x \cdot \frac{1}{x^2}\,dx = 2\int_0^1 \frac{1}{x}\,dx = 2\ln x\Big|_0^1 = \infty$$

Every single sample gives a finite number. The distribution is valid. The mean is still infinite.

**Why does Monte Carlo show this?**

```
x = 0.5    → 1/x² = 4
x = 0.1    → 1/x² = 100
x = 0.01   → 1/x² = 10,000
x = 0.001  → 1/x² = 1,000,000
```

Each draw is finite. But occasionally $x$ lands very close to 0 and the value is enormous. Those rare enormous values pull the running mean upward. It never settles — the more samples you draw, the more likely you hit a very small $x$, and the mean keeps climbing. That is the huge standard deviation.

### Back to the CS model

Near a coincidence $r = x_i - x_j \to 0$:

$$|\psi_\lambda|^2 \sim r^{2\lambda}, \qquad E_\text{loc} \sim \frac{C}{r^2}$$

Two completely separate questions with different answers:

| Question | Integral | Converges when |
|---|---|---|
| Is $\psi_\lambda$ normalizable? | $\displaystyle\int r^{2\lambda}\,dr$ | $\lambda > -\tfrac{1}{2}$ — always satisfied |
| Is $\mathbb{E}[E_\text{loc}]$ finite? | $\displaystyle\int r^{2\lambda} \cdot \frac{1}{r^2}\,dr = \int r^{2\lambda-2}\,dr$ | $\lambda > \tfrac{1}{2}$ — fails below |

The first integral converges easily for any physical $\lambda$. The second multiplies by $1/r^2$ from $E_\text{loc}$, shifting the exponent by $-2$ and making convergence strictly harder.

Having an analytic formula for the integrand does not help. $2\int_0^1 dx/x$ is a perfectly explicit analytic expression — it is still $\infty$.
