# Soft-sphere dilute Bose gas (NQS reproduction)

Reproduces the energy-per-particle results of

> F. Mazzanti, A. Polls, A. Fabrocini,
> *Energy and structure of dilute hard- and soft-sphere Bose gases*,
> arXiv:cond-mat/0305502 (2003).

using neural quantum states (VMC) from the `qvarnet` engine. **Problem-specific
code only** — it depends on `qvarnet` but lives outside `src/qvarnet` so the
engine stays general.

## Physics

3D homogeneous gas of `N` bosons, Hamiltonian (paper Eq. 7)

    H = -∇²ᵢ Σ + Σ_{i<j} V(r_ij),   V(r) = V0 for r<R, else 0   (soft sphere, Eq. 9)

The control parameter is the **gas parameter** `x = ρ a³`, where the s-wave
scattering length is (Eq. 10)

    a = R[1 - tanh(K₀R)/(K₀R)],   K₀ = √(V0/2).

## Units & conventions

We work in the **paper's convention** so every number lands on Mazzanti's tables:
`ℏ²/2m = 1` (kinetic operator `-∇²`), lengths in units of `a` (so `a = 1`),
energies in units of `ℏ²/2ma²`. In this convention `K₀² = V0·m/ℏ² = V0/2`, and the
paper's published barriers `V0(SS10)=0.00681670` (R=10) and `V0(SS5)=0.06308561`
(R=5) both give `a = 1` (regression-tested).

**Bridge to the engine.** `qvarnet.PenetrableSphereHamiltonian` runs in `ℏ = m = 1`
(kinetic `-½∇²`) — a factor of two from the paper. The translation lives in two
helpers only: build with `V0_engine = engine_V0(V0_paper)` (= V0/2) and report
`E_paper = to_paper_energy(E_engine)` (= 2·E). Lengths (`R`, box `L`) are identical
in both conventions; only `V0` and the energy carry the factor of two. Use
`n_dim=3` and `PeriodicBoundary(L)` when constructing the Hamiltonian.

Validated at low `x` against the Lee–Yang expansion (Eq. 1) and against the
first-order upper bound (Eq. 31, `first_order_energy_upper_bound`).

## Files

- `dilute_gas.py` — parametrization layer (3D only): scattering length `a(V0,R)`
  and its inverse, `x → (ρ, L)` box geometry, Lee–Yang `E/N` benchmark.
- `test_dilute_gas.py` — unit tests (physics limits, round-trips, units).
- `soft_sphere.ipynb` — working notebook for this system (DeepSet PBC ansatz,
  training, `E/N`, `g(r)`). Run the kernel from this directory so `import
  dilute_gas` resolves locally.

## Run the tests

    cd soft_sphere_gas
    uv run pytest test_dilute_gas.py -q

## Status / roadmap

1. ✅ Parametrization layer (`dilute_gas.py`).
2. ⬜ Single anchored VMC run at one `x`; check it approaches Lee–Yang.
3. ⬜ 3D pairwise Jastrow ansatz (shipped `LogJastrow` is 1D only).
4. ⬜ Sweep `x` → reproduce the `E/N(x)` curve.
5. ⬜ Hard- vs soft-sphere shape dependence at equal `a` (onset near `x ≳ 10⁻³`).
6. ⬜ Structure: `g(r)`, `S(k)`, `n(k)`.
