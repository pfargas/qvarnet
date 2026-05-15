# VMC Gradient for the Jastrow–Gaussian Ansatz

This note clarifies the role of $\lambda$ in the CS scan, distinguishes the
Hellmann–Feynman trick used in `cs_model_theory.md` from the actual VMC gradient,
and derives the VMC gradient from scratch.

---

## 1. What $\lambda$ is — and is not

$\lambda$ is the **Jastrow exponent**, the single variational parameter of the ansatz:

$$\log|\psi_\lambda|(x) = \lambda \underbrace{\sum_{i<j}\log|x_i-x_j|}_{J(x)} - \tfrac{1}{2}\underbrace{\sum_i x_i^2}_{B(x)/2}$$

Scanning $\lambda$ traces the variational energy curve $E_\text{var}(\lambda)$.
"Training" this model means finding the $\lambda$ that minimises $E_\text{var}(\lambda)$.

$\lambda$ is **not** a gradient. It is the parameter with respect to which one takes gradients.

---

## 2. What the Hellmann–Feynman step actually computes

The HF step in `cs_model_theory.md` differentiates

$$E(\lambda) = \langle \psi_\lambda \,|\, H(\lambda) \,|\, \psi_\lambda \rangle$$

where **both** $\psi_\lambda$ and $H(\lambda)$ depend on $\lambda$:

$$H(\lambda) = -\sum_i\partial_i^2 + B + 2\lambda(\lambda-1)A$$

By HF: $\dfrac{dE(\lambda)}{d\lambda} = \left\langle \dfrac{dH(\lambda)}{d\lambda}\right\rangle_\lambda$

The derivative of the Hamiltonian is:

$$\frac{dH(\lambda)}{d\lambda} = \frac{d}{d\lambda}\bigl[2\lambda(\lambda-1)\bigr] A = 2(2\lambda-1)\,A$$

And the left-hand side is known analytically from $E(\lambda) = N(1+\lambda(N-1))$:

$$\frac{dE(\lambda)}{d\lambda} = N(N-1)$$

Equating: $N(N-1) = 2(2\lambda-1)\langle A\rangle_\lambda$, and solving:

$$\langle A\rangle_\lambda = \frac{N(N-1)}{2(2\lambda-1)}$$

**This is the sole purpose of the HF step: to get $\langle A\rangle_\lambda$ analytically.**
It is not the VMC gradient.

---

## 3. Why the two derivatives are different

The VMC gradient is the derivative of $E_\text{var}(\lambda) = \langle\psi_\lambda|H(L)|\psi_\lambda\rangle$
at **fixed** $H(L)$, as $\lambda$ changes only through the ansatz. The HF derivative is:

$$\frac{d}{d\lambda}\langle\psi_\lambda|H(\lambda)|\psi_\lambda\rangle
= \frac{d}{d\lambda}\langle\psi_\lambda|H(L)|\psi_\lambda\rangle
+ \frac{d}{d\lambda}\langle\psi_\lambda|\underbrace{H(\lambda)-H(L)}_{[2\lambda(\lambda-1)-2L(L-1)]A}|\psi_\lambda\rangle$$

$$= \underbrace{\frac{dE_\text{var}}{d\lambda}}_{\text{VMC gradient}}
  + \frac{d}{d\lambda}\!\Bigl[\bigl(2\lambda(\lambda-1)-2L(L-1)\bigr)\langle A\rangle_\lambda\Bigr]$$

The HF step computes the **left-hand side** (which equals $N(N-1)$).
The VMC gradient is the **first term on the right** — a different object.

---

## 4. Deriving the VMC gradient

We want $\dfrac{d}{d\lambda}E_\text{var}(\lambda)$ where $E_\text{var}(\lambda) = \langle\psi_\lambda|H(L)|\psi_\lambda\rangle$ and $H(L)$ is fixed.

### 4a. Splitting the derivative

$$\frac{dE_\text{var}}{d\lambda}
= \frac{d}{d\lambda}\int |\psi_\lambda(x)|^2\,\underbrace{\frac{H(L)\,\psi_\lambda(x)}{\psi_\lambda(x)}}_{E_\text{loc}(x;\lambda)}\,dx$$

$$= \underbrace{\int \frac{d|\psi_\lambda|^2}{d\lambda}\,E_\text{loc}\,dx}_{\text{(I) measure changes}}
  + \underbrace{\int |\psi_\lambda|^2\,\frac{\partial E_\text{loc}}{\partial\lambda}\,dx}_{\text{(II) local energy changes explicitly}}$$

### 4b. The explicit term (II) vanishes

This is the non-obvious part. The explicit $\lambda$-dependence of $E_\text{loc}$ comes only from the kinetic term (since $H(L)$ is fixed, the potential does not depend on $\lambda$). We can write:

$$\frac{\partial E_\text{loc}(x;\lambda)}{\partial\lambda}
= \frac{\partial}{\partial\lambda}\frac{H(L)\psi_\lambda(x)}{\psi_\lambda(x)}
= \frac{H(L)(\partial_\lambda\psi_\lambda)}{\psi_\lambda} - \frac{H(L)\psi_\lambda}{\psi_\lambda}\frac{\partial_\lambda\psi_\lambda}{\psi_\lambda}
= \frac{H(L)\,\partial_\lambda\psi_\lambda}{\psi_\lambda} - E_\text{loc}\cdot O_\lambda$$

where $O_\lambda(x) = \dfrac{\partial_\lambda\psi_\lambda}{\psi_\lambda} = \partial_\lambda\log|\psi_\lambda| = J(x)$.

Then:

$$\int |\psi_\lambda|^2 \frac{\partial E_\text{loc}}{\partial\lambda}\,dx
= \int\psi_\lambda\cdot H(L)(\partial_\lambda\psi_\lambda)\,dx - \int|\psi_\lambda|^2 E_\text{loc}\,O_\lambda\,dx$$

Since $H(L)$ is **Hermitian**:

$$\int\psi_\lambda\cdot H(L)(\partial_\lambda\psi_\lambda)\,dx = \int(H(L)\psi_\lambda)\cdot\partial_\lambda\psi_\lambda\,dx
= \int|\psi_\lambda|^2 E_\text{loc}\,O_\lambda\,dx$$

The two terms cancel exactly:

$$\boxed{\int |\psi_\lambda|^2 \frac{\partial E_\text{loc}}{\partial\lambda}\,dx = 0}$$

This holds for **any** variational parameter in **any** Hermitian Hamiltonian.
The $\partial_\lambda E_\text{loc}$ term never contributes to the VMC gradient.

### 4c. The measure term (I): the score function estimator

For the **normalised** $|\psi_\lambda|^2$:

$$\frac{d|\psi_\lambda(x)|^2}{d\lambda}
= 2|\psi_\lambda(x)|^2\Bigl(\partial_\lambda\log|\psi_\lambda(x)| - \langle\partial_\lambda\log|\psi_\lambda|\rangle_\lambda\Bigr)
= 2|\psi_\lambda|^2\bigl(O_\lambda(x) - \langle O_\lambda\rangle_\lambda\bigr)$$

where the subtraction of $\langle O_\lambda\rangle$ enforces that $\int d|\psi_\lambda|^2/d\lambda\,dx = 0$.

For this ansatz: $O_\lambda(x) = \partial_\lambda\log|\psi_\lambda| = J(x) = \sum_{i<j}\log|x_i-x_j|$.

Substituting into (I):

$$\text{(I)} = 2\int|\psi_\lambda|^2\bigl(J(x)-\langle J\rangle_\lambda\bigr)E_\text{loc}(x;\lambda)\,dx
= 2\text{Cov}_\lambda(J,\,E_\text{loc})$$

### 4d. The VMC gradient

Since (II) = 0:

$$\boxed{\frac{dE_\text{var}}{d\lambda} = 2\,\text{Cov}_\lambda\!\left(J,\,E_\text{loc}\right)
= 2\!\left(\langle J\cdot E_\text{loc}\rangle_\lambda - \langle J\rangle_\lambda\langle E_\text{loc}\rangle_\lambda\right)}$$

where $J(x) = \sum_{i<j}\log|x_i-x_j|$ is the only variational "direction" for this one-parameter ansatz.

---

## 5. Simplifying with the known local energy

From the kinetic energy derivation, $E_\text{loc}$ is linear in $A(x)$:

$$E_\text{loc}(x;\lambda) = \underbrace{\bigl[2\lambda(1-\lambda)+2L(L-1)\bigr]}_{C(\lambda)} A(x) + N + \lambda N(N-1)$$

Since $N+\lambda N(N-1)$ is a constant (no $x$-dependence), the covariance only picks up the $A$ term:

$$\frac{dE_\text{var}}{d\lambda} = 2\,C(\lambda)\,\text{Cov}_\lambda(J,\,A)$$

This is the simplest possible form for the gradient: a scalar prefactor $C(\lambda)$ times one
covariance between the Jastrow log-derivative $J$ and the pair-distance sum $A$.

---

## 6. Sanity check: gradient is zero at the minimum $\lambda = L$

At $\lambda = L$: $C(L) = 2L(1-L) + 2L(L-1) = 0$, so immediately:

$$\frac{dE_\text{var}}{d\lambda}\bigg|_{\lambda=L} = 2 \cdot 0 \cdot \text{Cov}_L(J,A) = 0 \checkmark$$

This is also consistent with zero-variance: when $C(\lambda)=0$ the local energy is constant
everywhere ($= E_0$), so every estimator has zero variance and the gradient is identically zero
regardless of what $\text{Cov}(J,A)$ is.

---

## 7. Summary: the three different $\lambda$-derivatives

| Derivative | What changes with $\lambda$ | What it equals | Used for |
|---|---|---|---|
| $dE(\lambda)/d\lambda$ | both $\psi_\lambda$ and $H(\lambda)$ | $N(N-1)$ (exact) | **Deriving $\langle A\rangle_\lambda$ via HF** |
| $dE_\text{var}(\lambda)/d\lambda$ | only $\psi_\lambda$, $H(L)$ fixed | $2C(\lambda)\,\text{Cov}_\lambda(J,A)$ | **VMC optimisation gradient** |
| $\partial E_\text{loc}/\partial\lambda$ at fixed $x$ | explicit $\lambda$ in kinetic formula | $(2-4\lambda)A(x)+N(N-1)$ | **Cancels exactly** (Hermiticity), contributes nothing |
