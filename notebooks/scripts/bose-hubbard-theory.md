# Bose-Hubbard Physics: Theory Notes

These notes explain the full chain from the continuous-space Hamiltonian to the
Bose-Hubbard model, including every energy term and where each parameter comes from.

---

## 1. The continuous Hamiltonian

We simulate $N$ bosons in 1D with natural units $\hbar = m = 1$:

$$H = -\frac{1}{2}\sum_{i=1}^N \frac{\partial^2}{\partial x_i^2}
    + V_0 \sum_{i=1}^N \sin^2\!\left(\frac{\pi x_i}{a}\right)
    + g_{1D} \sum_{i<j} \delta(x_i - x_j)$$

- $a$ — lattice spacing
- $V_0$ — lattice depth
- $g_{1D}$ — contact interaction strength (the 1D coupling constant)

The lattice potential $V_0 \sin^2(\pi x/a)$ has its minima at $x = 0, a, 2a, \ldots$
(the lattice sites) and its maxima halfway between them.

The $\delta$-function interaction is approximated in the simulation by a narrow Gaussian
of width $\sigma \ll a$:
$$g_{1D}\,\delta(x) \;\approx\; \frac{g_{1D}}{\sigma\sqrt{2\pi}}\,e^{-x^2/2\sigma^2}$$

---

## 2. Single-particle physics: Bloch bands

Before adding interactions, consider a single particle in the lattice.
The Schrödinger equation is:

$$\left[-\frac{1}{2}\frac{d^2}{dx^2} + V_0\sin^2\!\left(\frac{\pi x}{a}\right)\right]\psi_k(x) = \varepsilon_k\,\psi_k(x)$$

This is the **Mathieu equation**. By Bloch's theorem the solutions are labelled by a
crystal momentum $k$:
$$\psi_k(x) = e^{ikx}\,u_k(x), \qquad u_k(x+a) = u_k(x)$$

The energy $\varepsilon_k$ is periodic in $k$ with period $2\pi/a$, forming **bands**.
For $N_\text{sites}$ sites with periodic boundary conditions, the allowed momenta in the
first Brillouin zone are:
$$k_m = \frac{2\pi m}{N_\text{sites}\,a}, \qquad m = 0, 1, \ldots, N_\text{sites}-1$$

**Computing $\varepsilon_k$ numerically (plane-wave method):**
Write $V_0\sin^2(\pi x/a) = \frac{V_0}{2} - \frac{V_0}{4}e^{i2\pi x/a} - \frac{V_0}{4}e^{-i2\pi x/a}$.
In a plane-wave basis $G_n = 2\pi n/a$, the Hamiltonian matrix is:

$$(H_k)_{mn} = \frac{1}{2}(k+G_m)^2\,\delta_{mn} + \tilde{V}_{m-n}$$

with $\tilde{V}_0 = V_0/2$, $\tilde{V}_{\pm 1} = -V_0/4$, and all other $\tilde{V}_n = 0$.

---

## 3. Wannier functions

Bloch states $\psi_k(x)$ are spread over the whole lattice. The equivalent localised basis
is the **Wannier functions**, one centred on each site $j$:

$$w_j(x) = \frac{1}{\sqrt{N_\text{sites}}} \sum_{k} e^{-ikja}\,\psi_k(x)$$

where the sum runs over all $N_\text{sites}$ Bloch momenta in the lowest band.

Key properties:
- $w_j(x) = w_0(x - ja)$ — all Wannier functions are the same shape, just shifted.
- $\int |w_j(x)|^2\,dx = 1$ — normalised.
- $\langle w_i | w_j \rangle = \delta_{ij}$ — orthogonal.
- For deep lattices ($V_0/E_R \gg 1$): $w_0(x) \approx$ Gaussian with width
  $\sigma_w \approx \frac{a}{\pi}\!\left(\frac{E_R}{V_0}\right)^{1/4}$.

The **recoil energy** $E_R = \frac{\pi^2}{2a^2}$ (with $\hbar=m=1$) sets the natural
energy scale for the lattice.

---

## 4. Tight-binding expansion and the Hubbard parameters

Any single-particle state can be expanded in the Wannier basis. The single-particle
Hamiltonian projected onto the lowest band becomes:

$$H_{sp} = \sum_{ij} t_{ij}\,a^\dagger_i a_j,
\qquad t_{ij} = \langle w_i | H_{sp} | w_j \rangle$$

The matrix elements:

| Term | Expression | Physical meaning |
|---|---|---|
| $t_{ii} \equiv \varepsilon_0$ | $\frac{1}{N_\text{sites}}\sum_k \varepsilon_k$ | on-site energy (mean of lowest band) |
| $t_{i,i\pm 1} \equiv -t$ | $\frac{1}{N_\text{sites}}\sum_k e^{\pm ika}\varepsilon_k$ | nearest-neighbour hopping |
| $t_{i,i\pm 2},\ldots$ | $\approx 0$ for deep lattice | longer-range hoppings (small) |

**How to extract $t$ from the band:**
For a pure nearest-neighbour tight-binding band $\varepsilon_k = \varepsilon_0 - 2t\cos(ka)$:
$$t = \frac{\varepsilon_{k=\pi/a} - \varepsilon_{k=0}}{4} = \frac{\text{bandwidth}}{4}$$

**Deep-lattice approximation:**
$$t \approx \frac{4}{\sqrt{\pi}}\,E_R\!\left(\frac{V_0}{E_R}\right)^{3/4} e^{-2\sqrt{V_0/E_R}}$$

The exponential makes $t$ extremely sensitive to $V_0$.

For the interaction projected onto the Wannier basis (keeping only the on-site term):
$$U = g_{1D} \int_0^{N_\text{sites}\cdot a} |w_0(x)|^4\,dx$$

**Deep-lattice approximation** (Gaussian Wannier):
$$U \approx \frac{g_{1D}}{\sigma_w\sqrt{2\pi}}
  \approx \frac{g_{1D}\,\pi}{a\sqrt{2\pi}}\!\left(\frac{V_0}{E_R}\right)^{1/4}$$

---

## 5. The Bose-Hubbard model

Combining the above, the full Hamiltonian projected onto the lowest band is:

$$H = \underbrace{N\varepsilon_0}_{\text{on-site energy}} + \underbrace{\left[-t\sum_{\langle ij\rangle}(a^\dagger_i a_j + \text{h.c.}) + \frac{U}{2}\sum_i n_i(n_i-1)\right]}_{H_{BH}}$$

**Convention:** textbooks always drop the constant $N\varepsilon_0$ and call only the
bracketed part "the Bose-Hubbard Hamiltonian" $H_{BH}$.

This is fine as long as you only compare energies at fixed $N$ — the constant cancels.
But VMC computes the **full** energy, so you must add $N\varepsilon_0$ back when
comparing to $H_{BH}$:

$$\boxed{E_\text{VMC} \approx N\varepsilon_0 + E_{BH}}$$

The "$\approx$" hides: (1) multi-band corrections, (2) longer-range hoppings,
(3) off-site interaction terms, all small when $V_0/E_R \gtrsim 3$.

---

## 6. The on-site energy $\varepsilon_0$ ("zero-point energy")

$\varepsilon_0$ is the energy of a single boson sitting in its Wannier state at a lattice
site. It has two contributions:

$$\varepsilon_0 = \underbrace{\langle w_0 | -\tfrac{1}{2}\partial_x^2 | w_0 \rangle}_{\text{kinetic (zero-point motion)}} + \underbrace{\langle w_0 | V_0\sin^2 | w_0 \rangle}_{\text{lattice potential}}$$

The **kinetic part dominates**: a particle localised at a site has large momentum
uncertainty (Heisenberg), costing significant kinetic energy. This is the lattice
analogue of the harmonic-oscillator zero-point energy $\tfrac{1}{2}\hbar\omega$.

In our parameters ($V_0/E_R = 5$, $a=1$):

$$\varepsilon_0 = \frac{1}{N_\text{sites}}\sum_k \varepsilon_k \approx 9.59, \qquad N\varepsilon_0 \approx 38.4$$

This is the big positive number that makes $E_\text{VMC} \approx 38$ while $E_{BH} \approx -2$
to $0$. Increasing $V_0$ makes $\varepsilon_0$ larger (deeper potential well → more
zero-point kinetic energy).

---

## 7. Parameter relationships

| BH parameter | Depends on | Formula (deep lattice) | Scaling |
|---|---|---|---|
| $\varepsilon_0$ | $V_0, a$ | $\frac{1}{N}\sum_k\varepsilon_k$ | grows $\sim V_0$ |
| $t$ | $V_0, a$ | $\frac{4}{\sqrt{\pi}}E_R(V_0/E_R)^{3/4}e^{-2\sqrt{V_0/E_R}}$ | **exponential** in $V_0$ |
| $U$ | $g_{1D}, V_0, a$ | $g_{1D}/(\sigma_w\sqrt{2\pi})$ | linear in $g_{1D}$, $\sim V_0^{1/4}$ |
| $U/t$ | all | $\propto g_{1D} \cdot e^{+2\sqrt{V_0/E_R}}$ | linear in $g_{1D}$, **exponential** in $V_0$ |

**Key insight:** $U/t$ can be tuned over many orders of magnitude by changing $V_0$ alone,
because $t$ dies exponentially while $U$ only grows as $V_0^{1/4}$.
Optical lattice experiments exploit this: laser intensity controls $V_0$.

In the notebook, we fix $V_0$ and sweep $g_{1D}$, so $U/t$ changes linearly.

---

## 8. Ground-state energy in the two limits

### Superfluid limit ($U/t \to 0$)

All $N$ bosons condense into the $k=0$ Bloch state (lowest single-particle level).
The BH relative energy is:

$$E_{BH}^{SF} = N\,\varepsilon_{k=0} - N\varepsilon_0 = -2Nt$$

(using $\varepsilon_{k=0} = \varepsilon_0 - 2t$ for a cosine band).

This is **exact** for $U=0$.

Full energy: $E_{SF} = N\varepsilon_0 - 2Nt$

### Mott insulator limit ($U/t \to \infty$, unit filling $\bar{n}=1$)

The ground state is the Fock state with exactly one boson per site:
$|\Psi_0\rangle = |1,1,\ldots,1\rangle$.

Zeroth order: $E^{(0)} = 0$ (no double occupancies, no interaction energy).

Second-order perturbation theory in $t/U$: each bond can virtually create a
doublon-hole pair (cost $U$, hopping matrix element $t\sqrt{2}$ from Bose factor):

$$E_{BH}^{MI} \approx -\frac{4Nt^2}{U}$$

The factor 4 = $(\sqrt{2})^2 \times 2$ (Bose factor squared, times both hop directions).

Full energy: $E_{MI} = N\varepsilon_0 - 4Nt^2/U$

### Phase boundary (thermodynamic limit, 1D)

The quantum phase transition from SF to MI occurs at:
$$(U/t)_c \approx 3.37 \quad \text{(DMRG, 1D)}$$

For small systems (like $N=N_\text{sites}=4$) there is no sharp transition, only a
smooth crossover. Exact diagonalization (ED) of the $N=4$ BH model gives the exact
finite-size ground state for any $U/t$.

---

## 9. What VMC computes

The VMC ansatz is a neural network $\psi_\theta(x_1,\ldots,x_N)$ that outputs
$\log|\psi_\theta(\mathbf{x})|$ (the log-wavefunction).

The VMC energy estimator is:
$$E_{VMC} = \frac{\langle\psi_\theta|H|\psi_\theta\rangle}{\langle\psi_\theta|\psi_\theta\rangle}
= \mathbb{E}_{|\psi_\theta|^2}\!\left[E_L(\mathbf{x})\right]$$

where the **local energy** is:
$$E_L(\mathbf{x}) = \underbrace{-\frac{1}{2}\sum_i \left[\Delta_i\log|\psi| + |\nabla_i\log|\psi||^2\right]}_{\text{kinetic}} + \underbrace{V_0\sum_i\sin^2\!\left(\frac{\pi x_i}{a}\right) + g_{1D}\sum_{i<j}\delta(x_i-x_j)}_{\text{potential}}$$

The kinetic part uses the log-derivative identity:
$\frac{\Delta\psi}{\psi} = \Delta\log|\psi| + |\nabla\log|\psi||^2$

This is the full energy — **no approximation**, no projection onto a single band.
VMC is exact in principle; the only error is the variational bias (finite network).

### Comparison to BH

| | $\varepsilon_0$ included | multi-band | result |
|---|---|---|---|
| $H_{BH}$ (textbook) | ✗ | ✗ | $E_{BH}$ |
| $H_{BH}$ + offset | ✓ | ✗ | $N\varepsilon_0 + E_{BH}$ |
| VMC | ✓ | ✓ | $E_{VMC}$ |

The agreement $E_{VMC} \approx N\varepsilon_0 + E_{BH}$ validates the single-band
tight-binding approximation. Deviations tell you how much the higher bands matter.

---

## 10. Summary of all energy scales

For $V_0/E_R = 5$, $a=1$, $g_{1D}=4$, $N=4$:

| Quantity | Value | Meaning |
|---|---|---|
| $E_R = \pi^2/2a^2$ | $4.93$ | recoil energy (scale for single-particle dynamics) |
| $\varepsilon_0$ | $9.59$ | on-site energy per particle |
| $N\varepsilon_0$ | $38.4$ | total lattice zero-point energy |
| $t$ | $0.326$ | hopping (bandwidth/4) |
| $2Nt$ | $2.6$ | kinetic energy gain in SF phase |
| $U = g_{1D}\|w\|^4$ | $\sim g_{1D}\times 0.05$ | on-site repulsion |
| $U/t$ | $\sim 6$ | regime (>3.37 → Mott-like) |
| $E_{BH}$ | $-3$ to $0$ | BH energy relative to $N\varepsilon_0$ |
| $E_{VMC}$ | $\sim 35$–$38$ | full energy measured by VMC |
