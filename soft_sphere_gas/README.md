# Soft-sphere dilute Bose gas (NQS reproduction)

Reproduces the energy-per-particle results of

> F. Mazzanti, A. Polls, A. Fabrocini,
> *Energy and structure of dilute hard- and soft-sphere Bose gases*,
> arXiv:cond-mat/0305502 (2003).

using neural quantum states (VMC) from the `qvarnet` engine. **Problem-specific
code only** — it depends on `qvarnet` but lives outside `src/qvarnet` so the
engine stays general.

## Physics

3D homogeneous gas of `N` bosons, `ℏ = m = 1`, Hamiltonian (paper Eq. 7)

    H = -½ Σ ∇²ᵢ + Σ_{i<j} V(r_ij),   V(r) = V0 for r<R, else 0   (soft sphere, Eq. 9)

implemented by `qvarnet.PenetrableSphereHamiltonian` with `PeriodicBoundary(L)`.

The control parameter is the **gas parameter** `x = ρ a³`, where the s-wave
scattering length is (Eq. 10)

    a = R[1 - tanh(K₀R)/(K₀R)],   K₀ = √V0.

All energies are reported in units of `ℏ²/2ma² = 1/a²`, and validated at low `x`
against the Lee–Yang expansion (Eq. 1).

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
