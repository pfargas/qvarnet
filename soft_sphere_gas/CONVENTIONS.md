# Units, conventions, and the sweep — dilute soft-sphere Bose gas

Reference implementation of Mazzanti, Polls & Fabrocini, *"Energy and structure of
dilute hard- and soft-sphere Bose gases"* (PRA 2003, arXiv:cond-mat/0305502). Paper PDF:
`../../comments/hard-soft-spheres-bosons.pdf`.

This document is the single source of truth for **what units we work in, why the numbers
look the way they do, how the problem layer talks to the qvarnet engine, and how the
experiment sweeps are nested.** Read it before touching `point.py` / `sweep.py`.

---

## 1. The physics, and what actually controls it

`N` spinless bosons in a cubic periodic box of volume `Ω = L³`, interacting through a
pairwise **soft (penetrable) sphere** (paper Eq. 7 + Eq. 9):

```
H = -(ℏ²/2m) Σᵢ ∇²ᵢ  +  Σ_{i<j} V(r_ij),     V(r) = V₀ for r < R, else 0.
```

The two-body physics is fixed by the **s-wave scattering length** (Eq. 10):

```
a = R [ 1 − tanh(K₀R)/(K₀R) ],     K₀² = V₀ m/ℏ².
```

A dilute gas does **not** depend on `(V₀, R, ρ)` separately. It depends on exactly two
**dimensionless** combinations:

| knob | meaning | paper values |
|---|---|---|
| `x = ρ a³` | **gas parameter** (diluteness) | swept 10⁻⁵ … 10⁻² |
| `R/a` | **shape / softness** of the potential | SS10 = 10, SS5 = 5, hard sphere = 1 |

Everything else is an overall scale we are free to set to 1. That is the whole reason the
paper (and we) work *in units of `a`*.

---

## 2. The unit system: `a = 1`, `ℏ²/2m = 1`

We adopt the **paper's convention** so every number we produce drops directly onto its
tables and figures:

- **length unit = the scattering length `a`**, i.e. `a = 1`;
- **`ℏ²/2m = 1`** ⇒ the kinetic operator is `−∇²`, and energies are in units of `ℏ²/2ma²`.

Consequences (this is why the numbers look the way they do):

- `R = 10` means the barrier is **10 scattering lengths wide**, not 10 of anything physical.
- `x = ρ a³ = ρ` (since `a = 1`): the gas parameter *is* the number density in these units.
- The box side follows from `x` and `N`:  **`L = (N/x)^{1/3}`**  (because `ρ = N/L³ = x`).
- `E/N` comes out directly in `ℏ²/2ma²`, ready to overlay on Lee–Yang / DMC.
- In this convention Eq. 10 reads **`K₀² = V₀/2`** (because `m/ℏ² = 1/2` when `ℏ²/2m = 1`).

### Why V₀ looks so small

For SS10, `V₀ = 0.00681670`. It looks tiny but is not weak — it is a *wide* barrier
(`R = 10 a`) tuned to scatter like a *small* `a = 1`. In the Born regime (valid here,
`K₀R ≈ 0.58`):

```
a ≈ (m/ℏ²) V₀ (4π/3 R³)/(4π) = V₀ R³ / 6      ⇒      V₀ ≈ 6 a / R³.
```

The `R³` is what shrinks V₀: SS10 (R=10) → V₀ ≈ 6/1000; SS5 (R=5) → V₀ ≈ 6/125, i.e. ~10×
larger. The honest "interaction strength" is the dimensionless `β = V₀R²/2 = (K₀R)²`:
`β(SS10) = 0.34`, `β(SS5) = 0.79` — both O(1).

---

## 3. R, V₀ and a are not independent — the shape axis

**At fixed `a = 1`, picking `R` fixes `V₀`** (invert Eq. 10):

```python
V0_paper = soft_sphere_V0_for_scattering_length(a=1.0, R=R)
```

So a "soft sphere" is **one free number**, not two. The map `R ↔ V₀` (at `a = 1`) is a
monotone bijection:

```
R → ∞   :  V₀ → 0     (very wide, very weak — the softest gas)
R → 1⁺  :  V₀ → ∞     (the HARD SPHERE, a → R)
```

So "sweep V₀ toward the hard sphere" and "sweep R toward 1" are the **same one-dimensional
path**, just labelled by the other coordinate. We label the **shape axis by `R`** (with V₀
inferred), because in units of `a` that is the independent knob and `R = 1` is exactly the
hard-sphere endpoint.

> **Aside — "fix R = 1, vary V₀" is a different *unit* choice, same physics.** That picture
> uses the length unit `= R` instead of `= a`; then `a = a(V₀) < 1` is a *derived* output and
> raising the wall (`V₀ → ∞`) hardens the sphere (`a → R = 1`). It is correct, but every energy
> would need rescaling by `(a/R)²` to compare to the paper's `a`-based tables — bookkeeping we
> avoid by fixing `a`.

---

## 4. The engine bridge — the one factor of two

The qvarnet engine (`PenetrableSphereHamiltonian`, `train`, the kinetic estimator) works in
**`ℏ = m = 1`**, so its kinetic operator is **`−½∇²`** — a factor of two from the paper's
`−∇²`. The translation lives in **two helpers in `dilute_gas.py` and nowhere else**:

| direction | helper | factor |
|---|---|---|
| build the Hamiltonian | `V0_engine = engine_V0(V0_paper)` | `× ½` |
| read the energy back | `E_paper = to_paper_energy(E_engine)` | `× 2` |

With both applied,
`E_paper = ⟨−∇²⟩ + ⟨V₀_paper · θ(R−r)⟩` is exactly the paper's Hamiltonian expectation.
**Lengths (`a`, `R`, box `L`) are identical in both conventions; only `V₀` and the energy
carry the 2.** Get this wrong and you are off by 2× in V₀ (→ 5.7× in `x`) or 2× in energy.

This is regression-tested: `test_paper_published_potentials` (published V₀ ⇒ a = 1),
`test_engine_bridge_is_a_factor_of_two`.

---

## 5. Validation ladder (all in paper units)

| target | function / source | role |
|---|---|---|
| lower bound `4πx` | `lee_yang_energy_per_particle` leading term | rigorous lower bound (Lieb–Yngvason) |
| Lee–Yang `E/N` (Eq. 1) | `lee_yang_energy_per_particle(x)` | low-`x` universal limit; our `x→0` target |
| 1st-order upper bound (Eq. 31) | `first_order_energy_upper_bound(x, V0, R)` | `= ½ρṼ(0)` = uncorrelated ⟨V⟩/N; **free zero-training check** |
| DMC points | paper Tables I/III (Giorgini et al.) | the "exact" benchmark; hierarchy `DMC < VMC < upper bound` |

**Reference numbers** (paper units, `a = 1`):

| system | R | V₀ (paper) | V₀ (engine) | Eq.31 UB @ x=10⁻⁴ | Eq.31 UB @ x=10⁻² |
|---|---|---|---|---|---|
| SS10 | 10 | 0.00681670 | 0.00340835 | 0.0014277 | 0.14277 |
| SS5  |  5 | 0.06308561 | 0.03154280 | 0.0016516 | 0.16516 |

Lee–Yang `E/N`: `x=10⁻⁴ → 0.001317`, `x=10⁻³ → 0.014479`, `x=10⁻² → 0.18616`.

---

## 6. The sweep hierarchy

From outer (physics) to inner (numerics). Each level wraps `run_point`:

```
shape  R          which soft sphere; V₀ inferred at a=1. HS = R→1.
  └ x             the main curve E/N(x); sets L=(N/x)^{1/3} at fixed N.
      └ N         finite-size: run an N-ladder, fit E/N vs 1/N, take the intercept (TL).
          └ seed  independent chains; accept/combine via the three-referee verdict
                  (result.diagnose(): Geweke/HW stationarity, split-R̂≤1.1, MC-error floor).
              └ HP optimizer / sampler / DeepSet widths / epochs. Unknown optimum:
                  ASHA per point by default, switchable to tune-then-freeze.
```

**What each axis answers**
- **x**: the energy curve and its approach to Lee–Yang; structure (`g(r)`, `S(k)`, `n(k)`).
- **N**: removes finite-box bias — the paper is the thermodynamic limit, we are not.
- **seed**: an honest error bar and a convergence gate (discard non-stationary / unmixed runs).
- **shape (R)**: the nonuniversal / shape-dependence story; extrapolation toward the hard sphere.

**Milestone 1 (current): the SS10 `E/N(x)` curve.** Fixed `N`, single (then few) seeds, a
small ASHA tune, `x ∈ {10⁻⁵ … 10⁻²}`, overlaid on Lee–Yang + the Eq. 31 bound. Finite-`N`,
multi-seed and the shape sweep come after the curve is qualitatively right.

> **Gate toward the hard sphere.** As `R → 1` the core becomes nearly impenetrable; a bare
> DeepSet cannot place a clean node at `r ≈ a` and the variance explodes. The HS end needs a
> **3D pairwise Jastrow** (shipped `LogJastrow` is 1D only — see README roadmap). SS10/SS5 at
> low–moderate `x` are fine with DeepSet alone.

---

## 7. The atomic unit — `run_point` contract

```python
run_point(potential: Potential, x: float, N: int, seed: int, hp: HP) -> PointResult
```

- **Inputs split by meaning:** `potential` (= R, V₀ derived) and `x`, `N` are the *physics*
  (they define the true answer); `seed` is RNG; `hp` is the *solver* (optimizer, sampler,
  model, epochs) — the only thing the HP tuner varies.
- **Inside** `run_point`: `L = (N/x)^{1/3}`; build `LogWavefunction(DeepSet(...))`, the
  optimizer, the `SamplingConfig`, and `PenetrableSphereHamiltonian(R, V0=engine_V0(V0_paper),
  PeriodicBoundary(L))` from `hp`; call `train(...)`; then convert with `to_paper_energy`.
- **Returns** `E/N` (paper units), error of the mean, `Var(E_loc)`, acceptance, the
  `result.diagnose()` verdict, and (later) `g(r)/S(k)/n(k)` — plus enough metadata for the
  SQLite queue to be resumable.

`hp` is a frozen, serializable dataclass so a run is fully described by
`(potential, x, N, seed, hp)` — that tuple is the DB primary key.
