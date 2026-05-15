# Kinetic Energy of the Jastrow–Gaussian Ansatz: Full Derivation

This note derives, step by step, the kinetic contribution to the local energy for the one-parameter
trial state introduced in `cs_model_theory.md`.  Notation follows `cs_vmc_step.md`:

- $\theta$ — variational parameter (Jastrow exponent)
- $\lambda$ — fixed physical coupling in the Hamiltonian

$$\log|\psi_\theta|(x) = \underbrace{\theta \sum_{i<j}\log|x_i - x_j|}_{\theta\, J(x)} \underbrace{- \tfrac{1}{2}\sum_i x_i^2}_{-\tfrac{1}{2}B(x)}$$

with the Hamiltonian convention $\hbar^2/m = 1$, so $T = -\sum_i \partial^2/\partial x_i^2$.

---

## 0. From wavefunction to log-model: the kinetic identity

Let $u(x) = \log|\psi_\theta(x)|$.  Because $|\psi_\theta| = e^u$:

$$\frac{\partial |\psi_\theta|}{\partial x_k} = e^u \frac{\partial u}{\partial x_k} = |\psi_\theta|\, \partial_k u$$

Differentiating again:

$$\frac{\partial^2 |\psi_\theta|}{\partial x_k^2}
= \frac{\partial}{\partial x_k}\!\left(|\psi_\theta|\,\partial_k u\right)
= \left(\partial_k|\psi_\theta|\right)\partial_k u + |\psi_\theta|\,\partial_k^2 u
= |\psi_\theta|\left[(\partial_k u)^2 + \partial_k^2 u\right]$$

Dividing by $|\psi_\theta|$ and summing over all $k$:

$$\boxed{T = -\frac{\Delta|\psi_\theta|}{|\psi_\theta|} = -\!\left(\,|\nabla u|^2 + \Delta u\right)}$$

This identity is exact for any positive wavefunction; it is the starting point used throughout
the CS notebook.

---

## 1. First derivative of $u$

$$u(x) = \theta \sum_{i < j} \log|x_i - x_j| - \tfrac{1}{2}\sum_i x_i^2$$

Differentiating with respect to $x_k$.  The log-pair term $\log|x_i - x_j|$ depends on $x_k$
only when $k = i$ or $k = j$.  For the case $k = i$ (the argument is $x_k - x_j$ with $j < k$,
or $x_j - x_k$ with $j > k$; in both cases $\partial_{x_k}\log|x_k - x_j| = 1/(x_k - x_j)$):

$$\frac{\partial}{\partial x_k}\,\theta\sum_{i<j}\log|x_i - x_j|
= \theta \sum_{j \neq k} \frac{1}{x_k - x_j}$$

(both the $i=k$ and $j=k$ cases contribute exactly $1/(x_k - x_j)$, accumulated into $j \neq k$).

The Gaussian term gives simply $-x_k$.  Therefore:

$$\boxed{\partial_k u = \theta \sum_{j \neq k} \frac{1}{x_k - x_j} - x_k}$$

---

## 2. Second derivative of $u$

Differentiating $\partial_k u$ with respect to $x_k$:

- The Jastrow part: $\partial_{x_k}[1/(x_k - x_j)] = -1/(x_k-x_j)^2$.
- The Gaussian part: $\partial_{x_k}(-x_k) = -1$.

$$\boxed{\partial_k^2 u = -\theta\sum_{j\neq k}\frac{1}{(x_k - x_j)^2} - 1}$$

---

## 3. Laplacian $\Delta u$

Sum $\partial_k^2 u$ over all $k = 1, \ldots, N$.

**Gaussian contribution:** $\sum_k (-1) = -N$.

**Jastrow contribution:** each unordered pair $\{i,j\}$ appears exactly twice in
$\sum_k \sum_{j\neq k} (x_k - x_j)^{-2}$ (once as $(k,j) = (i,j)$ and once as $(k,j) = (j,i)$):

$$\sum_k \sum_{j \neq k} \frac{1}{(x_k - x_j)^2} = 2 \underbrace{\sum_{i<j}\frac{1}{(x_i-x_j)^2}}_{A(x)}$$

Collecting:

$$\boxed{\Delta u = -2\theta\, A(x) - N}$$

---

## 4. Squared gradient $|\nabla u|^2$

$$|\nabla u|^2 = \sum_k (\partial_k u)^2
= \sum_k \!\left(\theta\sum_{j\neq k}\frac{1}{x_k - x_j} - x_k\right)^{\!2}$$

Expand the square into three sums:

$$|\nabla u|^2
= \theta^2 \underbrace{\sum_k\!\left(\sum_{j\neq k}\frac{1}{x_k-x_j}\right)^{\!2}}_{\text{(I) Jastrow}^2}
  - 2\theta \underbrace{\sum_k x_k \sum_{j\neq k}\frac{1}{x_k - x_j}}_{\text{(II) cross}}
  + \underbrace{\sum_k x_k^2}_{B(x)}$$

### 4a. Term (I): Jastrow squared

$$\text{(I)} = \sum_k \left[\,\underbrace{\sum_{j\neq k}\frac{1}{(x_k-x_j)^2}}_{\text{diagonal}} + \underbrace{\sum_{\substack{j\neq k\\l\neq k\\l\neq j}}\frac{1}{(x_k-x_j)(x_k-x_l)}}_{\text{three-body}}\right]$$

**Diagonal part** counts each pair twice: equals $2A(x)$.

**Three-body part vanishes.**  For every unordered triple $\{a,b,c\}$ the contribution is

$$2\left[\frac{1}{(a-b)(a-c)} + \frac{1}{(b-a)(b-c)} + \frac{1}{(c-a)(c-b)}\right]$$

(the factor 2 comes from the two orderings of $j,l$ in the double sum).
Using a common denominator $(a-b)(a-c)(b-c)$ [with $(b-a)=-(a-b)$ etc.]:

$$\frac{1}{(a-b)(a-c)} - \frac{1}{(a-b)(b-c)} + \frac{1}{(a-c)(b-c)}
= \frac{(b-c) - (a-c) + (a-b)}{(a-b)(a-c)(b-c)} = \frac{0}{(\cdots)} = 0$$

Numerator: $(b-c) - (a-c) + (a-b) = b - c - a + c + a - b = 0$.  $\square$

Therefore:

$$\text{(I)} = 2A(x)$$

### 4b. Term (II): cross term

Pair each ordered term $(k,j)$ with its reverse $(j,k)$:

$$\frac{x_k}{x_k - x_j} + \frac{x_j}{x_j - x_k}
= \frac{x_k}{x_k-x_j} - \frac{x_j}{x_k - x_j}
= \frac{x_k - x_j}{x_k - x_j} = 1$$

Every unordered pair $\{k,j\}$ with $k<j$ contributes exactly $1$.  There are $N(N-1)/2$ such pairs:

$$\text{(II)} = \frac{N(N-1)}{2}$$

### 4c. Collecting

$$\boxed{|\nabla u|^2 = 2\theta^2 A(x) - \theta N(N-1) + B(x)}$$

---

## 5. Kinetic energy

Substitute $\Delta u$ (Section 3) and $|\nabla u|^2$ (Section 4) into the log-model identity:

$$T = -\left(\Delta u + |\nabla u|^2\right)$$

$$= -\Big(\underbrace{-2\theta A - N}_{\Delta u}
        + \underbrace{2\theta^2 A - \theta N(N-1) + B}_{|\nabla u|^2}\Big)$$

$$= -\left[(2\theta^2 - 2\theta)A + (-N - \theta N(N-1)) + B\right]$$

$$\boxed{T = 2\theta(1-\theta)\,A(x) + N + \theta N(N-1) - B(x)}$$

The three pieces have clear physical origins:
- $2\theta(1-\theta)\,A$: kinetic contribution from the Jastrow correlations.
- $N + \theta N(N-1)$: zero-point energy from the Gaussian envelope and its cross-coupling with the Jastrow.
- $-B(x) = -\sum_i x_i^2$: kinetic cost from the spatial variation of the Gaussian envelope; it exactly cancels the trapping potential (see below).

---

## 6. Sanity checks on the kinetic term alone

**No Jastrow ($\theta = 0$).**  The wavefunction is a pure Gaussian $e^{-B/2}$.

For $\theta=0$: $\partial_k u = -x_k$, $(\partial_k u)^2 = x_k^2$, $\partial_k^2 u = -1$.

$$T = -\sum_k\left[x_k^2 + (-1)\right] = -B + N = N - B \checkmark$$

Consistent with the formula: $T|_{\theta=0} = 0 + N + 0 - B = N - B$. ✓

**Exact eigenstate ($\theta = \lambda$).**  The full local energy must equal $E_0 = N(1+\lambda(N-1))$.
Adding the potential $V = B + 2\lambda(\lambda-1)A$ to $T$:

$$E_\text{loc} = 2\lambda(1-\lambda)A + N + \lambda N(N-1) - \cancel{B} + \cancel{B} + 2\lambda(\lambda-1)A$$

$$= \left[2\lambda(1-\lambda) + 2\lambda(\lambda-1)\right]A + N + \lambda N(N-1) = 0 + N(1 + \lambda(N-1)) = E_0 \checkmark$$

The $-B$ from the kinetic term and the $+B$ from the trap cancel exactly.
The Jastrow kinetic term $2\lambda(1-\lambda)A$ and the Jastrow potential $2\lambda(\lambda-1)A$ cancel exactly.
The local energy is constant — zero variance.

---

## 7. The VMC gradient with respect to $\theta$

### 7.1 What $\theta$ is — and is not

$\theta$ is the Jastrow exponent, the single variational parameter.  Scanning $\theta$ traces
the variational energy curve $E_\text{var}(\theta) = \langle\psi_\theta|H|\psi_\theta\rangle$
where $H$ has fixed physical coupling $\lambda$.
$\theta$ is **not** a gradient; it is the parameter with respect to which we take the gradient.

### 7.2 What the Hellmann–Feynman step actually computes

The HF step differentiates

$$E(\theta) = \langle\psi_\theta\,|\,H(\theta)\,|\,\psi_\theta\rangle$$

where **both** $\psi_\theta$ and $H(\theta) = -\sum_i\partial_i^2 + B + 2\theta(\theta-1)A$ change with $\theta$.
This is not the VMC gradient.  The relationship between the two is:

$$\underbrace{\frac{d}{d\theta}\langle\psi_\theta|H(\theta)|\psi_\theta\rangle}_{\text{HF step} = N(N-1)}
= \underbrace{\frac{d}{d\theta}\langle\psi_\theta|H|\psi_\theta\rangle}_{\text{VMC gradient}}
+ \frac{d}{d\theta}\Bigl[\underbrace{\bigl(2\theta(\theta-1)-2\lambda(\lambda-1)\bigr)}_{\text{difference in couplings}}\langle A\rangle_\theta\Bigr]$$

The HF step computes the left-hand side — knowing $dE(\theta)/d\theta = N(N-1)$ analytically —
and uses it only to extract $\langle A\rangle_\theta = N(N-1)/[2(2\theta-1)]$.
The VMC gradient is the first term on the right, a different quantity.

### 7.3 Deriving the VMC gradient

We want $dE_\text{var}/d\theta$ where $H$ is fixed and only $\psi_\theta$ changes.

**Splitting the total derivative:**

$$\frac{dE_\text{var}}{d\theta}
= \frac{d}{d\theta}\int|\psi_\theta|^2\,E_\text{loc}(x;\theta)\,dx
= \underbrace{\int\frac{d|\psi_\theta|^2}{d\theta}\,E_\text{loc}\,dx}_{\text{(I) measure term}}
+ \underbrace{\int|\psi_\theta|^2\,\frac{\partial E_\text{loc}}{\partial\theta}\,dx}_{\text{(II) explicit term}}$$

**Term (II) vanishes by Hermiticity.**  Writing $E_\text{loc} = H\psi_\theta/\psi_\theta$ and
letting $O_\theta = \partial_\theta\log|\psi_\theta|$:

$$\frac{\partial E_\text{loc}}{\partial\theta}
= \frac{H\,\partial_\theta\psi_\theta}{\psi_\theta} - E_\text{loc}\cdot O_\theta$$

Substituting into (II):

$$\text{(II)}
= \int\psi_\theta\cdot H(\partial_\theta\psi_\theta)\,dx
  - \int|\psi_\theta|^2 E_\text{loc}\,O_\theta\,dx$$

Since $H$ is Hermitian, $\int\psi_\theta \cdot H(\partial_\theta\psi_\theta)\,dx = \int(H\psi_\theta)\cdot\partial_\theta\psi_\theta\,dx = \int|\psi_\theta|^2 E_\text{loc}\,O_\theta\,dx$.
The two pieces cancel:

$$\boxed{\text{(II)} = 0}$$

This holds for any Hermitian $H$ and any variational parameter — the explicit change of
$E_\text{loc}$ with the parameter never contributes to the VMC gradient.

**Term (I): the score-function estimator.**  For the normalised $|\psi_\theta|^2$:

$$\frac{d|\psi_\theta|^2}{d\theta} = 2|\psi_\theta|^2\bigl(O_\theta - \langle O_\theta\rangle_\theta\bigr),
\qquad O_\theta(x) = \partial_\theta\log|\psi_\theta| = J(x) = \sum_{i<j}\log|x_i-x_j|$$

Therefore:

$$\text{(I)} = 2\int|\psi_\theta|^2\bigl(J - \langle J\rangle_\theta\bigr)E_\text{loc}\,dx
= 2\,\text{Cov}_\theta(J,\,E_\text{loc})$$

**Full VMC gradient:**

$$\boxed{\frac{dE_\text{var}}{d\theta} = 2\,\text{Cov}_\theta\!\left(J,\,E_\text{loc}\right)}$$

### 7.4 Closed form using the known local energy

Since $E_\text{loc}(x;\theta) = C(\theta)\,A(x) + N + \theta N(N-1)$ with $C(\theta) = 2\theta(1-\theta)+2\lambda(\lambda-1)$,
and the additive constant drops out of a covariance:

$$\frac{dE_\text{var}}{d\theta} = 2\,C(\theta)\,\text{Cov}_\theta(J,\,A)$$

**Check at $\theta = \lambda$:** $C(\lambda) = 0$, so the gradient is zero regardless of $\text{Cov}(J,A)$. ✓
This is consistent with $\theta = \lambda$ being the minimum and with zero variance at that point
($E_\text{loc}$ is constant, so every covariance with it is zero).

### 7.5 Summary: three different $\theta$-derivatives

| Derivative | What changes with $\theta$ | Result |
|---|---|---|
| $dE(\theta)/d\theta$ | both $\psi_\theta$ and $H(\theta)$ | $N(N-1)$ — used only to get $\langle A\rangle_\theta$ |
| $dE_\text{var}(\theta)/d\theta$ | only $\psi_\theta$, physical $H$ fixed | $2C(\theta)\,\text{Cov}_\theta(J,A)$ — the true VMC gradient |
| $\partial E_\text{loc}/\partial\theta$ at fixed $x$ | explicit $\theta$ in kinetic formula | cancels exactly by Hermiticity, contributes nothing |
