# B-spline Jastrow

Implementation: `BSplineJastrow` in `src/qvarnet/ansatz/jastrow.py`. Test: `tests/test_bspline_jastrow.py`.

## Ansatz

1D, raw (lab) coordinates, pair Jastrow in log space:

$$
\log J(x) = \sum_{i<j} u(r_{ij}), \qquad r_{ij} = |x_i - x_j|
$$

$$
u(r) = \sum_{k=-1}^{M+1} c_k \, B\!\left(\frac{r}{\Delta} - k\right), \qquad \Delta = \frac{r_\text{cut}}{M}
$$

`M = n_knots` uniform intervals on $[0, r_\text{cut}]$. The basis index runs $k=-1,\dots,M+1$ ($M+3$ functions) so that every point of $[0, r_\text{cut}]$ is covered by 4 splines.

## Cardinal cubic B-spline

Support $[-2, 2]$, piecewise cubic, $C^2$:

$$
B(t) = \begin{cases}
\frac{1}{6}\left(4 - 6t^2 + 3|t|^3\right) & |t| < 1 \\
\frac{1}{6}\left(2 - |t|\right)^3 & 1 \le |t| < 2 \\
0 & |t| \ge 2
\end{cases}
$$

Values at the knots (needed for the constraints):

| | $t=0$ | $t=\pm1$ |
|---|---|---|
| $B$ | $2/3$ | $1/6$ |
| $B'$ | $0$ | $\mp 1/2$ |
| $B''$ | $-2$ | $1$ |

So at a knot $r = j\Delta$ only $c_{j-1}, c_j, c_{j+1}$ contribute:

$$
u(j\Delta) = \frac{c_{j-1} + 4c_j + c_{j+1}}{6}, \quad
u'(j\Delta) = \frac{c_{j+1} - c_{j-1}}{2\Delta}, \quad
u''(j\Delta) = \frac{c_{j-1} - 2c_j + c_{j+1}}{\Delta^2}
$$

## Constraints (hard-wired, not learned)

**Cutoff.** $u(r_\text{cut}) = u'(r_\text{cut}) = u''(r_\text{cut}) = 0$. Set $j=M$ in the knot formulas: the three equations force

$$
c_{M-1} = c_M = c_{M+1} = 0 .
$$

Since $B(t-k)$ for $k \le M-2$ has support ending at $t = M$, this also gives $u \equiv 0$ for $r \ge r_\text{cut}$ exactly, and $u$ is $C^2$ across the cutoff.

**Cusp.** $u'(0) = \gamma$ (`cusp`). Set $j=0$:

$$
c_{-1} = c_1 - 2\Delta\gamma .
$$

For $M = 2$, $c_1 = c_{M-1} = 0$, so $c_{-1} = -2\Delta\gamma$. Needs $M \ge 2$ (not checked in code).

**Free parameters:** $c_0, \dots, c_{M-2}$ → `params["coeffs"]`, shape `(M-1,)`, initialised to zero ($J = 1$).

## Why the cusp matters

$u(|x|)$ as a function of $x = x_i - x_j$ has a kink at $x=0$ unless $u'(0)=0$. Then

$$
\partial_x^2\, u(|x|) = u''(|x|) + 2u'(0)\,\delta(x) .
$$

Sampled $E_\text{loc}$ never sees the $\delta$ term (measure zero), so a nonzero $u'(0)$ with no matching $\delta$ potential biases the energy and the variance-zero property is lost. Default `cusp=0.0` removes the kink.

**Contact interaction.** For $H = -\tfrac12\sum_i \partial_i^2 + g\sum_{i<j}\delta(x_i - x_j)$ ($\hbar = m = 1$), the relative-coordinate kinetic term is $-\partial_x^2$, and integrating the Schrödinger equation across $x=0$ gives

$$
\psi'(0^+) - \psi'(0^-) = g\,\psi(0) \;\Rightarrow\; 2u'(0) = g \;\Rightarrow\; \gamma = g/2 .
$$

Assumption: these units. Re-derive for others (e.g. Lieb–Liniger $H = -\sum\partial^2 + 2c\sum\delta$ gives $\gamma = c$).

## Periodic box

With `L` set: minimum-image $dx \to dx - L\,\mathrm{round}(dx/L)$, then $r = |dx|$.

Requires $r_\text{cut} \le L/2$ (raises `ValueError` otherwise). Reason: the minimum-image map has a kink at $|dx| = L/2$; since $u \equiv 0$ beyond $r_\text{cut} \le L/2$, the kink is invisible and $\log J$ is exactly $L$-periodic and $C^2$. (Contrast with `LogJastrow`, which needs the $\sin$ form; see `periodic-systems.md`.)

## Derivatives (what `kinetic_log` gets via autodiff)

$$
\partial_{x_i} \log J = \sum_{j \ne i} u'(r_{ij})\,\mathrm{sign}(x_i - x_j), \qquad
\partial_{x_i}^2 \log J = \sum_{j \ne i} u''(r_{ij}) \quad (r_{ij} > 0)
$$

Nothing hand-coded; JAX differentiates the spline. $B$ is $C^2$, so the Laplacian is continuous (piecewise linear in $r$).

**Parameter derivatives.** $\log J$ is linear in the free coefficients:

$$
O_k = \frac{\partial \log J}{\partial c_k} = \sum_{i<j} B\!\left(\frac{r_{ij}}{\Delta} - k\right) + \delta_{k,1}\sum_{i<j} B\!\left(\frac{r_{ij}}{\Delta} + 1\right)
$$

(the second term from $c_{-1}$ depending on $c_1$).

## Code

```python
from qvarnet.ansatz import BSplineJastrow, LogWavefunction

jastrow = BSplineJastrow(
    n_particles=N,
    r_cut=2.0,      # open: a few mean spacings; periodic: <= L/2
    n_knots=8,      # M; M-1 free coefficients
    cusp=0.0,       # u'(0)
    L=None,         # box length, None = open
)
model = LogWavefunction(network=net, jastrow=jastrow, n_particles=N, n_dim=1)
```

Input `(..., N)` (1D raw coords), output `(..., 1)`.

Evaluation: dense, all $M+3$ basis functions per pair → cost $O(N^2 M)$. Fine for $M \sim 10$; switch to gathering the 4 active splines per pair if $M$ gets large.

## Limitations / open choices

- 1D only (`x` is `(..., N)`). Higher $d$ needs $r = \lVert \mathbf{r}_i - \mathbf{r}_j \rVert$ and reshaping.
- Cannot represent a singular core $\lambda \log r$ (Calogero–Sutherland). Combine with `LogJastrow` via a wrapper that sums both (not written; `LogWavefunction` takes one `jastrow`).
- Same $u$ for all pairs: no spin-resolved $u_{\uparrow\uparrow}, u_{\uparrow\downarrow}$ for fermions yet.
- Choice of $r_\text{cut}$ and $M$ is untested on real runs: no evidence yet on what values work.

## Test

`tests/test_bspline_jastrow.py` checks, with random coefficients:
- $u'(0) = \gamma$ for $\gamma \in \{0, 0.7\}$,
- $u, u', u''$ vanish at $r_\text{cut}$, $u = 0$ past it,
- $L$-periodicity under $x_0 \to x_0 + L$, output shape `(batch, 1)`.

Run: `uv run pytest -q tests/test_bspline_jastrow.py`.
