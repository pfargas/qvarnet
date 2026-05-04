# Why CM=off beats CM=on when ω_trap = ω_int = 1

## The Setup

System: N=5 particles in 1D with Hamiltonian (ℏ = m = 1):

$$H = -\frac{1}{2}\sum_{i=1}^{N}\frac{\partial^2}{\partial x_i^2} + \frac{\omega_{\rm trap}^2}{2}\sum_{i=1}^{N}x_i^2 + \frac{\omega_{\rm int}^2}{2}\sum_{i=1}^{N}(x_i - x_{i+1})^2$$

Two regimes under investigation:
- **Case A**: $\omega_{\rm trap}=0.1$, $\omega_{\rm int}=1$ → CM=on wins
- **Case B**: $\omega_{\rm trap}=\omega_{\rm int}=1$ → CM=off wins

---

## Step 1: Normal Mode Decomposition

Both cases are a quadratic Hamiltonian, so we can diagonalise exactly.

The potential matrix $K$ (where $V = \frac{1}{2}\mathbf{x}^T K \mathbf{x}$) with PBC is the **circulant matrix**:

$$K_{ij} = \begin{cases}
\omega_{\rm trap}^2 + 2\omega_{\rm int}^2 & i = j \\
-\omega_{\rm int}^2 & |i-j| = 1 \pmod{N} \\
0 & \text{otherwise}
\end{cases}$$

Its eigenvalues are (circulant matrix with diagonal $d=\omega_{\rm trap}^2+2\omega_{\rm int}^2$ and off-diagonal $c=-\omega_{\rm int}^2$):

$$\boxed{\lambda_k = \omega_{\rm trap}^2 + 2\omega_{\rm int}^2\left(1 - \cos\frac{2\pi k}{N}\right), \quad k = 0,1,\ldots,N-1}$$

The normal mode frequencies are $\omega_k = \sqrt{\lambda_k}$ and the exact ground state energy is:

$$E_0 = \frac{1}{2}\sum_{k=0}^{N-1}\omega_k$$

### Case A: $\omega_{\rm trap}=0.1$, $\omega_{\rm int}=1$, $N=5$

| $k$ | $\lambda_k$ | $\omega_k$ | mode |
|-----|-------------|------------|------|
| 0 | 0.010 | **0.100** | CM |
| 1,4 | 1.392 | 1.180 | relative |
| 2,3 | 3.628 | 1.905 | relative |

$$E_0 = \tfrac{1}{2}(0.1 + 2\times1.180 + 2\times1.905) = 3.135$$

**The CM mode is 10× softer than the relative modes.**

### Case B: $\omega_{\rm trap}=\omega_{\rm int}=1$, $N=5$

| $k$ | $\lambda_k$ | $\omega_k$ | mode |
|-----|-------------|------------|------|
| 0 | 1.000 | **1.000** | CM |
| 1,4 | 2.382 | 1.543 | relative |
| 2,3 | 4.618 | 2.149 | relative |

$$E_0 = \tfrac{1}{2}(1.0 + 2\times1.543 + 2\times2.149) = 4.192$$

**All modes are within a factor of ~2 of each other.**

---

## Step 2: The Ground State Wavefunction Is Separable

For any quadratic Hamiltonian, the ground state factorises exactly into CM and relative parts:

$$\Psi_0(\mathbf{x}) = \psi_{\rm CM}(X_{\rm CM}) \cdot \psi_{\rm rel}(\mathbf{q}_1,\ldots,\mathbf{q}_{N-1})$$

where $X_{\rm CM} = \frac{1}{N}\sum_i x_i$ and $\mathbf{q}_k$ are the $N-1$ relative normal modes.

The CM sub-system behaves as a harmonic oscillator with **effective mass $N$** and **spring constant $N\omega_{\rm trap}^2$**, so the CM frequency is $\omega_{\rm CM} = \omega_{\rm trap}$.

The CM ground state:

$$\psi_{\rm CM}(X_{\rm CM}) \propto \exp\!\left(-\frac{N\,\omega_{\rm trap}}{2}\,X_{\rm CM}^2\right)$$

The 1σ width of the CM distribution:

$$\sigma_{\rm CM} = \frac{1}{\sqrt{N\,\omega_{\rm trap}}}$$

| Case | $\sigma_{\rm CM}$ |
|------|-------------------|
| A ($\omega_{\rm trap}=0.1$) | $1/\sqrt{5\times0.1} = \sqrt{2} \approx 1.41$ |
| B ($\omega_{\rm trap}=1$) | $1/\sqrt{5\times1} = 1/\sqrt{5} \approx 0.45$ |

**Case B has a very tight CM Gaussian.** The CM is well-localised near zero.

---

## Step 3: What CM Subtraction Actually Does to the Sampled Distribution

When `use_cm_coords=True`, the code wraps the model as:

```python
def effective_apply(params, x):
    return base_apply(params, x - X_CM(x))
```

The ansatz is $\log|\psi_\theta(\mathbf{x})| = f_\theta(\mathbf{x} - X_{\rm CM})$. The model **only receives the CM-subtracted coordinates**.

The `LogExponentialMLPwithPenalty` output:

$$\log|\psi_\theta(\mathbf{x})| = {\rm MLP}\!\left(\mathbf{x} - X_{\rm CM}\right) - \alpha\sum_{i=1}^{N}(x_i - X_{\rm CM})^4$$

**Crucially**, both terms depend only on $(x_i - X_{\rm CM})$, i.e., only on the relative coordinates. Therefore:

$$\frac{\partial}{\partial X_{\rm CM}}\log|\psi_\theta(\mathbf{x})| = 0 \quad \forall\,\theta$$

**The model probability $|\psi_\theta|^2$ is exactly flat in $X_{\rm CM}$, for every possible parameter value $\theta$.** No amount of training can fix this — it is structural.

The MCMC samples from $|\psi_\theta(\mathbf{x})|^2$. In the CM direction, this is:

$$P_{\rm model}(X_{\rm CM}) = \text{const} \quad \text{within the PBC box } [-L/2,\, L/2]$$

The **true** distribution is:

$$P_{\rm true}(X_{\rm CM}) \propto \exp\!\left(-N\,\omega_{\rm trap}\,X_{\rm CM}^2\right)$$

### CM acceptance is always 100%

The MH kernel accepts a proposal $\mathbf{x}'$ with probability:

$$A = \min\!\left(1,\; \frac{|\psi(\mathbf{x}'-X'_{\rm CM})|^2}{|\psi(\mathbf{x}-X_{\rm CM})|^2}\right)$$

Consider a **pure CM shift**: $x'_i = x_i + \delta$ for all $i$. Then $X'_{\rm CM} = X_{\rm CM} + \delta$, so:

$$\mathbf{x}' - X'_{\rm CM} = \mathbf{x} - X_{\rm CM} \implies |\psi(\mathbf{x}' - X'_{\rm CM})|^2 = |\psi(\mathbf{x}-X_{\rm CM})|^2 \implies A = 1$$

**Every move in the CM direction is accepted with probability 1.** The CM performs an unimpeded random walk with no restoring force at all.

---

## Step 4: The CM Random Walk — Quantitative Drift

The proposal in `mh_kernel_log` is:

```python
proposal = position + step_size * uniform_random_numbers[:-1]  # shape (DoF,)
```

Each $\delta x_i \sim \mathcal{N}(0, \text{step\_size}^2)$, so:

$$\delta X_{\rm CM} = \frac{1}{N}\sum_i \delta x_i \sim \mathcal{N}\!\left(0, \frac{\text{step\_size}^2}{N}\right)$$

The CM accepted step size per MCMC step is (CM direction has 100% acceptance):

$$\sigma_{\rm step}^{\rm CM} = \frac{\text{step\_size}}{\sqrt{N}} = \frac{0.5}{\sqrt{5}} \approx 0.224$$

With `chain_length=11`, `burn_in=10` per epoch, the chain runs 11 steps per epoch. Over $n_{\rm epoch}=3000$ epochs:

$$\sigma_{\rm drift} = 0.224 \times \sqrt{11 \times 3000} \approx 0.224 \times 182 \approx 40$$

The CM diffuses to the **PBC boundary** ($L/2 = 20$) within the span of training.

### How long until the CM leaves its physical 1σ region?

The CM leaves $\sigma_{\rm CM}$ after $M^*$ steps where $M^* \cdot \sigma_{\rm step}^2 = \sigma_{\rm CM}^2$:

$$M^* = \frac{\sigma_{\rm CM}^2}{\sigma_{\rm step}^2} = \frac{N\,\sigma_{\rm CM}^2}{\text{step\_size}^2}$$

| Case | $\sigma_{\rm CM}$ | $M^*$ (steps) | epochs |
|------|-------------------|----------------|--------|
| A ($\omega_{\rm trap}=0.1$) | 1.41 | $5\times2.0/0.25 = 40$ | ~4 |
| B ($\omega_{\rm trap}=1$) | 0.45 | $5\times0.2/0.25 = 4$ | <1 |

In **Case B**, the CM is already outside its physical 1σ width **within the very first epoch**.

---

## Step 5: The VMC Energy Estimator Is Broken

The VMC energy estimate is:

$$E_{\rm VMC} = \left\langle E_{\rm loc}(\mathbf{x})\right\rangle_{|\psi_\theta|^2}$$

where the local energy decomposes as:

$$E_{\rm loc}(\mathbf{x}) = T_{\rm loc}(\mathbf{x}) + V_{\rm trap}(\mathbf{x}) + V_{\rm int}(\mathbf{x})$$

The trapping potential decomposes exactly:

$$V_{\rm trap}(\mathbf{x}) = \frac{\omega_{\rm trap}^2}{2}\sum_i x_i^2 = \underbrace{\frac{N\,\omega_{\rm trap}^2}{2}\,X_{\rm CM}^2}_{V_{\rm trap}^{\rm CM}} + \underbrace{\frac{\omega_{\rm trap}^2}{2}\sum_i (x_i - X_{\rm CM})^2}_{V_{\rm trap}^{\rm rel}}$$

Since $|\psi_\theta|^2$ is flat in $X_{\rm CM}$ within the PBC box $[-L/2, L/2]$:

$$\left\langle V_{\rm trap}^{\rm CM}\right\rangle_{|\psi_\theta|^2} = \frac{N\,\omega_{\rm trap}^2}{2}\cdot\frac{1}{L}\int_{-L/2}^{L/2} X_{\rm CM}^2\,dX_{\rm CM} = \frac{N\,\omega_{\rm trap}^2}{2}\cdot\frac{L^2}{12}$$

With $N=5$, $L=40$ (so $L^2/12 \approx 133$):

| Case | $\langle V_{\rm trap}^{\rm CM}\rangle$ | True $E_{\rm CM} = \omega_{\rm trap}/2$ | Ratio |
|------|----------------------------------------|------------------------------------------|-------|
| A ($\omega_{\rm trap}=0.1$) | $5\times0.01/2\times133 = 3.3$ | 0.05 | 67× |
| B ($\omega_{\rm trap}=1$) | $5\times1/2\times133 = \mathbf{333}$ | 0.5 | **666×** |

### What the optimizer sees in Case B

The VMC energy reported during training is:

$$E_{\rm VMC}^{\rm (CM\,on)} \approx E_{\rm rel} + 333 \gg E_0 = 4.19$$

The model parameters $\theta$ only affect $E_{\rm rel}$ (through $\psi_{\rm rel}$). The CM contribution of **333** is irreducible — no gradient update can reduce it, because the gradients are:

$$\frac{\partial E_{\rm VMC}}{\partial\theta} = 2\left\langle (E_{\rm loc} - E_{\rm VMC})\,\nabla_\theta \log|\psi_\theta(\mathbf{x}-X_{\rm CM})|\right\rangle$$

And $\nabla_\theta \log|\psi_\theta|$ does not depend on $X_{\rm CM}$. So the gradient signal from the 333 CM-energy contribution is **pure noise**: it cancels to zero in expectation, but adds enormous variance to every gradient estimate.

The signal-to-noise ratio for the relative modes is:

$${\rm SNR} = \frac{|E_{\rm rel} - E_{\rm rel}^*|}{\sigma_{E_{\rm loc}}} \approx \frac{\delta E_{\rm rel}}{333}$$

which is essentially zero. The optimizer cannot learn the relative-mode structure because the CM noise drowns it completely.

---

## Step 6: Why Case A (ω_trap=0.1) Survives Despite the Same Problem

The argument above applies to both cases. The difference is **scale**:

| Quantity | Case A | Case B |
|----------|--------|--------|
| CM energy waste $\langle V_{\rm trap}^{\rm CM}\rangle$ | 3.3 | 333 |
| True ground state energy $E_0$ | 3.14 | 4.19 |
| Relative error in estimator | ~100% | ~8000% |
| Steps to exit $1\sigma_{\rm CM}$ | 40 | 4 |

In Case A, the CM noise is the **same order** as the signal. The optimizer still converges (slowly, imprecisely) because the relative mode gradients are not completely buried. It gets to the right answer because for 4 epochs (40 steps) the walkers are still in the physical CM region, and there are 3000 epochs total.

In Case B, the CM noise is **three orders of magnitude** larger than the signal. The optimizer is blind.

---

## Step 7: The Kinetic Energy Contribution from CM is Also Wrong

For a wavefunction that is flat in $X_{\rm CM}$:

$$\frac{\partial^2 \psi}{\partial X_{\rm CM}^2} = 0 \implies T_{\rm loc}^{\rm CM} = -\frac{1}{2N}\frac{\partial^2 \psi/\partial X_{\rm CM}^2}{\psi} = 0$$

So the kinetic energy contribution from the CM mode is **zero**, whereas the true value is:

$$\langle T_{\rm CM}\rangle_{\rm true} = \frac{\omega_{\rm CM}}{4} = \frac{\omega_{\rm trap}}{4}$$

The model with CM=on misestimates **both** kinetic and potential energy in the CM mode:
- Gets $T_{\rm CM} = 0$ instead of $\omega_{\rm trap}/4$
- Gets $\langle V_{\rm CM}\rangle = N\omega_{\rm trap}^2 L^2/24$ instead of $\omega_{\rm trap}/4$

The net loss from the CM mode in the VMC estimator:

$$\Delta E_{\rm CM} = \underbrace{0}_{T_{\rm CM}} + \underbrace{\frac{N\omega_{\rm trap}^2 L^2}{24}}_{V_{\rm CM}} - \underbrace{\frac{\omega_{\rm trap}}{2}}_{E_{\rm CM}^{\rm true}}$$

For Case B: $\Delta E_{\rm CM} = 0 + 333 - 0.5 \approx 332.5$

---

## Step 8: Why CM=off Works Well for Case B

With `use_cm_coords=False`, the model receives raw particle coordinates $(x_1,\ldots,x_N)$.

The ansatz:
$$\log|\psi_\theta(\mathbf{x})| = {\rm MLP}(x_1,\ldots,x_N) - \alpha\sum_i x_i^4$$

**The envelope $\sum_i x_i^4$ now confines the CM.** When $X_{\rm CM} = R$ is large:

$$\sum_i x_i^4 \approx N\,R^4 \quad (\text{for pure CM shift})$$

So the model probability decays as $\exp(-2\alpha N R^4)$ in the CM direction. The walkers are confined.

Moreover, in Case B, the problem is **well-conditioned**: all 5 normal mode frequencies span a factor of only ~2.15 (from 1.0 to 2.149). The function space the MLP needs to learn is roughly isotropic in the 5-dimensional input space. The Σxᵢ⁴ envelope has the right order of magnitude for ALL modes, not just the relative ones.

In Case A ($\omega_{\rm trap}=0.1$), the problem was ill-conditioned: the CM mode is 10× softer than the relative modes. The envelope Σxᵢ⁴ grows as $NR^4$ in CM (too confining — mismatches the true $\exp(-0.5R^2)$ Gaussian) and as $\sum_i(x_i-\bar{x})^4$ in relative (appropriate). This mismatch caused the catastrophic explosion.

| Property | Case A (0.1/1) | Case B (1/1) |
|----------|----------------|--------------|
| Frequency ratio $\omega_{\rm max}/\omega_{\rm min}$ | 1.905/0.1 = **19** | 2.149/1.0 = **2.1** |
| Condition number of $K$ | 362 | 4.6 |
| Envelope appropriate? | ✗ (wrong CM scale) | ✓ (all scales similar) |
| CM=off result | explosion | converges |

---

## Summary

The failure of CM=on when $\omega_{\rm trap}=\omega_{\rm int}=1$ is not a bug — it is a fundamental **structural mismatch** between the ansatz and the physics:

1. The CM subtraction makes $|\psi_\theta|^2$ **identically flat** in $X_{\rm CM}$ for all $\theta$.
2. The MH sampler accepts all CM moves with probability 1 → CM performs an unimpeded random walk.
3. After just ~4 MCMC steps, the CM has left its physical $1\sigma$ region ($\sigma_{\rm CM}=0.45$).
4. The trapping potential at displaced CM generates spurious energy $\sim 333$, vs. the true ground state energy $4.19$ → the VMC estimator is off by a factor of ~80.
5. All gradient signal is buried under this CM noise → the model cannot learn.

CM subtraction is only valid when the Hamiltonian is **translationally invariant** (or approximately so), meaning $V$ depends only on differences $x_i - x_j$ and there is **no external trap**. The moment there is a trap, the CM has a physical frequency $\omega_{\rm CM}=\omega_{\rm trap}$ and a physical Gaussian width $\sigma_{\rm CM}=1/\sqrt{N\omega_{\rm trap}}$. The model must be able to see and learn that CM Gaussian. Removing the CM from the model input destroys this ability.

| Regime | CM subtraction | Reason |
|--------|----------------|--------|
| $\omega_{\rm trap}=0$, any $\omega_{\rm int}$ | ✓ correct | CM is truly free, $V$ is TI |
| $\omega_{\rm trap}\ll\omega_{\rm int}$ | ✓ approximately OK | CM noise is small vs. signal |
| $\omega_{\rm trap}\sim\omega_{\rm int}$ | ✗ wrong | CM noise equals or dominates signal |
| $\omega_{\rm trap}>\omega_{\rm int}$ | ✗ very wrong | CM is the tightest mode; removing it is catastrophic |
