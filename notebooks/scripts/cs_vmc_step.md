# One Analytic VMC Step for the Calogero–Sutherland Jastrow Ansatz

We redo the entire VMC machinery with clean notation:

- $\lambda$ — fixed physical coupling in the Hamiltonian (the "truth")
- $\theta$ — variational parameter to be optimised (the "weight")

The goal is to show that gradient descent on the variational energy drives $\theta \to \lambda$.

---

## 1. Setup

### Hamiltonian (fixed)

$$H = -\sum_i \partial_i^2 + \underbrace{\sum_i x_i^2}_{B(x)} + 2\lambda(\lambda-1)\underbrace{\sum_{i<j}\frac{1}{(x_i-x_j)^2}}_{A(x)}$$

with $\hbar^2/m = 1$.  The exact ground-state energy is $E_0 = N(1+\lambda(N-1))$.

### Ansatz (parametrised by $\theta$)

$$\log|\psi_\theta|(x)
= \theta \underbrace{\sum_{i<j}\log|x_i-x_j|}_{J(x)}
  - \tfrac{1}{2}\underbrace{\sum_i x_i^2}_{B(x)/2}$$

At $\theta = \lambda$ this is the exact ground state.

---

## 2. Local Energy

The local energy is $E_\text{loc}(x;\theta) = H\psi_\theta(x)/\psi_\theta(x)$.

### Kinetic term

From the log-model identity $T = -(|\nabla u|^2 + \Delta u)$ with $u = \log|\psi_\theta|$
(derived in `cs_kinetic_derivation.md`):

$$\Delta u = -2\theta A - N$$

$$|\nabla u|^2 = 2\theta^2 A - \theta N(N-1) + B$$

$$T(\theta) = 2\theta(1-\theta)\,A + N + \theta N(N-1) - B$$

### Potential term

$$V = B + 2\lambda(\lambda-1)\,A$$

### Local energy

$$E_\text{loc}(x;\theta) = T(\theta) + V$$

$$= 2\theta(1-\theta)A + N + \theta N(N-1) - \cancel{B} + \cancel{B} + 2\lambda(\lambda-1)A$$

$$\boxed{E_\text{loc}(x;\theta)
= \underbrace{\bigl[2\theta(1-\theta)+2\lambda(\lambda-1)\bigr]}_{C(\theta,\lambda)}\,A(x)
+ N + \theta N(N-1)}$$

Two observations:
1. $E_\text{loc}$ depends on $x$ only through $A(x)$ — it is linear in $A$.
2. $C(\lambda,\lambda) = 2\lambda(1-\lambda)+2\lambda(\lambda-1) = 0$, so at $\theta=\lambda$ the local energy is constant $= E_0$ everywhere. Zero variance.

---

## 3. Variational Energy

$$E_\text{var}(\theta) = \langle E_\text{loc}\rangle_\theta = C(\theta,\lambda)\,\langle A\rangle_\theta + N + \theta N(N-1)$$

We need $\langle A\rangle_\theta$, the expectation of $A$ under $|\psi_\theta|^2$.

### Computing $\langle A\rangle_\theta$ via Hellmann–Feynman

$\psi_\theta$ is the exact ground state of the auxiliary Hamiltonian

$$H(\theta) = -\sum_i\partial_i^2 + B + 2\theta(\theta-1)A$$

with eigenvalue $E(\theta) = N(1+\theta(N-1))$.

The Hellmann–Feynman theorem applied to $H(\theta)$ (where **both** $H$ and $\psi_\theta$ move with $\theta$) gives

$$\frac{dE(\theta)}{d\theta} = \left\langle\frac{dH(\theta)}{d\theta}\right\rangle_\theta = 2(2\theta-1)\langle A\rangle_\theta$$

The left-hand side is $N(N-1)$ from the explicit eigenvalue, so:

$$\boxed{\langle A\rangle_\theta = \frac{N(N-1)}{2(2\theta-1)}, \qquad \theta > \tfrac{1}{2}}$$

> **Note:** this step differentiates $H(\theta)$, not the physical $H$.
> It is a trick to extract $\langle A\rangle_\theta$ analytically and has nothing to do with the VMC gradient.

### Closed-form variational energy

$$\boxed{E_\text{var}(\theta)
= \frac{C(\theta,\lambda)\,N(N-1)}{2(2\theta-1)} + N + \theta N(N-1), \qquad \theta > \tfrac{1}{2}}$$

**Check at $\theta=\lambda$:** $C(\lambda,\lambda) = 0$, so $E_\text{var}(\lambda) = N + \lambda N(N-1) = N(1+\lambda(N-1)) = E_0$. ✓

---

## 4. The VMC Gradient

We differentiate $E_\text{var}(\theta) = \langle\psi_\theta|H|\psi_\theta\rangle$ at **fixed** physical $H$.

### Derivation

$$\frac{dE_\text{var}}{d\theta}
= \frac{d}{d\theta}\int|\psi_\theta|^2 E_\text{loc}\,dx
= \underbrace{\int\frac{d|\psi_\theta|^2}{d\theta}E_\text{loc}\,dx}_{\text{(I)}}
+ \underbrace{\int|\psi_\theta|^2\frac{\partial E_\text{loc}}{\partial\theta}\,dx}_{\text{(II)}}$$

**Term (II) vanishes** by the Hermiticity of $H$. Writing $O_\theta = \partial_\theta\log|\psi_\theta| = J$:

$$\int|\psi_\theta|^2\frac{\partial E_\text{loc}}{\partial\theta}dx
= \int\psi_\theta\cdot H(\partial_\theta\psi_\theta)\,dx
  - \int|\psi_\theta|^2 E_\text{loc}\,O_\theta\,dx
\overset{H^\dagger=H}{=} 0$$

**Term (I)** uses $d|\psi_\theta|^2/d\theta = 2|\psi_\theta|^2(O_\theta - \langle O_\theta\rangle)$:

$$\text{(I)} = 2\int|\psi_\theta|^2(J-\langle J\rangle_\theta)E_\text{loc}\,dx = 2\,\text{Cov}_\theta(J,\,E_\text{loc})$$

### Result

$$\boxed{\frac{dE_\text{var}}{d\theta} = 2\,\text{Cov}_\theta(J,\,E_\text{loc})}$$

### Simplification

Since $E_\text{loc} = C(\theta,\lambda)\,A + \text{const}$ and constants drop out of covariances:

$$\frac{dE_\text{var}}{d\theta} = 2\,C(\theta,\lambda)\,\text{Cov}_\theta(J,\,A)$$

---

## 5. Finding the Optimal $\theta$

### Stationary condition

$$\frac{dE_\text{var}}{d\theta} = 0
\;\iff\;
C(\theta,\lambda) = 0
\quad\text{or}\quad
\text{Cov}_\theta(J,A) = 0$$

**Solving $C(\theta,\lambda) = 0$:**

$$2\theta(1-\theta) + 2\lambda(\lambda-1) = 0$$

$$\theta(1-\theta) = \lambda(1-\lambda)$$

$$\theta - \theta^2 = \lambda - \lambda^2$$

$$\theta^2 - \theta - \lambda^2 + \lambda = 0$$

$$(\theta^2 - \lambda^2) - (\theta - \lambda) = 0$$

$$(\theta - \lambda)\underbrace{(\theta + \lambda - 1)}_{\neq\, 0\text{ generically}} = 0$$

The two roots are $\theta = \lambda$ and $\theta = 1 - \lambda$.

### The two roots and the MC-valid domain

The restriction $\theta > \tfrac{1}{2}$ is on the **variational parameter** $\theta$, not on the
physical coupling $\lambda$.  The CS model is defined for all $\lambda > -\tfrac{1}{2}$
(normalizability condition $|\psi_\lambda|^2 \sim r^{2\lambda}$ integrable at coincidences).
The two cases are:

**Case $\lambda > \tfrac{1}{2}$.**
The true minimum $\theta = \lambda$ is inside the MC-valid domain $\theta > \tfrac{1}{2}$, and
the second root $\theta = 1-\lambda < \tfrac{1}{2}$ is outside it.  MC gradient descent
converges to the correct answer:

$$\boxed{\theta^* = \lambda}$$

**Case $\lambda < \tfrac{1}{2}$.**
Now $\theta = \lambda < \tfrac{1}{2}$ is **outside** the MC-valid domain, and
$\theta = 1-\lambda > \tfrac{1}{2}$ is **inside** it.  Gradient descent within the accessible
region converges to the wrong stationary point $\theta^* = 1-\lambda \neq \lambda$.  The MC
reports $E_\text{var}(1-\lambda) = N(1+(1-\lambda)(N-1)) > E_0$, a valid upper bound but not
the ground state.  This is a deeper failure than variance divergence: the optimisation itself
is attracted to the wrong answer by the domain constraint.

### Confirming the minimum for $\lambda > \tfrac{1}{2}$: variational principle

For any normalised $\psi_\theta$:

$$E_\text{var}(\theta) = \frac{\langle\psi_\theta|H|\psi_\theta\rangle}{\langle\psi_\theta|\psi_\theta\rangle} \geq E_0$$

with equality if and only if $\psi_\theta \propto \psi_0$, i.e., $\theta = \lambda$.  Since
$E_\text{var}(\lambda) = E_0$, $\theta = \lambda$ is the **global minimum** of $E_\text{var}$
over all $\theta$, and the unique minimum within $\theta > \tfrac{1}{2}$ when $\lambda > \tfrac{1}{2}$.

---

## 6. One Full VMC Step

Given current parameter $\theta^{(t)}$:

### Step 1 — Sample

Draw $M$ configurations $\{x^{(m)}\}_{m=1}^M$ from $|\psi_{\theta^{(t)}}|^2$ via Metropolis–Hastings.

### Step 2 — Evaluate local energy at each sample

$$E_\text{loc}^{(m)} = C(\theta^{(t)},\lambda)\,A(x^{(m)}) + N + \theta^{(t)} N(N-1)$$

where $A(x) = \sum_{i<j}(x_i-x_j)^{-2}$ and $C(\theta,\lambda) = 2\theta(1-\theta)+2\lambda(\lambda-1)$.

### Step 3 — Evaluate the log-derivative at each sample

$$O^{(m)} = \partial_\theta\log|\psi_\theta|(x^{(m)})\big|_{\theta=\theta^{(t)}} = J(x^{(m)}) = \sum_{i<j}\log|x_i^{(m)}-x_j^{(m)}|$$

### Step 4 — Estimate the gradient

$$\hat{g} = 2\!\left(\frac{1}{M}\sum_m O^{(m)} E_\text{loc}^{(m)} - \frac{1}{M}\sum_m O^{(m)}\cdot\frac{1}{M}\sum_m E_\text{loc}^{(m)}\right)
= 2\,\widehat{\mathrm{Cov}}(J,\, E_\text{loc})$$

### Step 5 — Update

$$\theta^{(t+1)} = \theta^{(t)} - \eta\,\hat{g}$$

### Exact gradient at each step

Replacing the MC estimate with the analytical expression:

$$g(\theta) = 2\,C(\theta,\lambda)\,\text{Cov}_\theta(J,A)$$

Since $J$ and $A$ are negatively correlated under $|\psi_\theta|^2$ (larger pair separations increase $J$
but decrease $A = \sum(x_i-x_j)^{-2}$), $\text{Cov}_\theta(J,A) < 0$ for all $\theta > \tfrac{1}{2}$.
The sign of the gradient is therefore controlled entirely by $C(\theta,\lambda) = -2(\theta-\lambda)(\theta+\lambda-1)$.

**For $\lambda > \tfrac{1}{2}$** (both stationary points straddle the domain boundary):

$$\theta > \lambda \;\Rightarrow\; C < 0 \;\Rightarrow\; g > 0 \;\Rightarrow\; \theta \text{ decreases toward } \lambda$$
$$\tfrac{1}{2} < \theta < \lambda \;\Rightarrow\; C > 0 \;\Rightarrow\; g < 0 \;\Rightarrow\; \theta \text{ increases toward } \lambda$$

Gradient descent is monotonically attracted to $\theta^* = \lambda$. $\blacksquare$

**For $\lambda < \tfrac{1}{2}$** (the true minimum lies outside the MC-valid domain):

$$\theta > 1-\lambda \;\Rightarrow\; C < 0 \;\Rightarrow\; g > 0 \;\Rightarrow\; \theta \text{ decreases toward } 1-\lambda$$
$$\tfrac{1}{2} < \theta < 1-\lambda \;\Rightarrow\; C > 0 \;\Rightarrow\; g < 0 \;\Rightarrow\; \theta \text{ increases toward } 1-\lambda$$

Gradient descent converges to $\theta^* = 1-\lambda \neq \lambda$.
The MC optimisation is blind to the true ground state.

---

## 7. Summary

| Quantity | Expression | At $\theta = \lambda$ |
|---|---|---|
| Local energy | $C(\theta,\lambda)\,A(x) + N + \theta N(N-1)$ | $E_0$ everywhere (zero variance) |
| $\langle A\rangle_\theta$ | $N(N-1)/[2(2\theta-1)]$, valid for $\theta > \tfrac{1}{2}$ | $N(N-1)/[2(2\lambda-1)]$ |
| $E_\text{var}(\theta)$ | $C\cdot N(N-1)/[2(2\theta-1)] + N + \theta N(N-1)$ | $E_0$ (global minimum) |
| VMC gradient | $2\,C(\theta,\lambda)\,\text{Cov}_\theta(J,A)$ | $0$ (since $C=0$) |
| MC convergence target | $\theta^* = \lambda$ if $\lambda > \tfrac{1}{2}$; $\quad\theta^* = 1-\lambda$ if $\lambda < \tfrac{1}{2}$ | correct iff $\lambda > \tfrac{1}{2}$ |
