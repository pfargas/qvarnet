# Calogero-Sutherland model

CS counterpart of `../soft_sphere_gas/`. `N` particles on a line in a harmonic trap with
inverse-square interactions (convention `ħ²/m = 1`):

    H = -Σ ∂²/∂xᵢ² + Σ xᵢ² + 2L(L-1) Σᵢ<ⱼ 1/(xᵢ-xⱼ)²

Exact ground state `log|ψ₀| = L Σᵢ<ⱼ log|xᵢ-xⱼ| - ½ Σ xᵢ²`, energy `E₀ = N(1 + L(N-1))`.

## Files
- `calogero_sutherland.ipynb` — parametrised setup. Builds the Hamiltonian, trains an ansatz,
  and wraps it in `run_cs(L=...)` so you can scan the coupling `L` (or `N`) and compare against
  the exact `E₀`. Pick the ansatz via `kind=`: `'analytic'` (exact single-λ baseline),
  `'mlp_jastrow'` (MLP + Gaussian envelope + `LogJastrow`, default), or `'mlp'`.

## Machinery (all in `qvarnet`)
- `qvarnet.hamiltonian.continuous.CalogeroSutherlandHamiltonian` (registered as `"CS-model"`)
- `qvarnet.models.analytic.CalogeroSutherlandAnalyticModel`
- `qvarnet.models.jastrow.LogJastrow`

See `../notebooks/scripts/cs_model_new.ipynb` for the cusp-condition convergence study.
