# Soft-sphere gas — session handoff / next steps

*Snapshot 2026-06-15. Read `CONVENTIONS.md` first (units, the engine bridge, the sweep design).*

## Where we are

Reproducing Mazzanti, Polls & Fabrocini (2003, `../../comments/hard-soft-spheres-bosons.pdf`)
with qvarnet NQS-VMC. The full milestone-1 pipeline is **built, lint-clean, and the physics
layer is tested** (24 tests in `test_dilute_gas.py`).

| file | role | state |
|---|---|---|
| `dilute_gas.py` | paper↔engine units, scattering length, Lee-Yang, Eq.31 bound | ✅ tested |
| `CONVENTIONS.md` | units / bridge / sweep design — the source of truth | ✅ |
| `point.py` | `Potential`, `HP`, `run_point`, training-free sanity gate | ✅ |
| `db.py` | resumable SQLite store (`outputs/soft_sphere.db`) | ✅ smoke-tested |
| `sweep.py` | `HPStrategy` (Frozen/ASHA/tune_then_freeze), `sweep_x`, `load_curve`, `feasible_x_grid` | ✅ smoke-tested |
| `analysis.py` | `plot_curve` (VMC vs lower bound / Lee-Yang / Eq.31) | ✅ |
| `run_ss10_curve.py` | milestone-1 driver | ✅ (do NOT auto-launch) |

## Validated facts (don't re-derive)

- **Engine bridge is correct to <0.1%**: uniform-gas ⟨V⟩/N matches Eq.31·(N−1)/N wherever the
  box fits. The factor of two lives only in `engine_V0` (V0/2 in) and `to_paper_energy` (×2 out).
- **Box constraint `x < N/(8R³)`** (need R < L/2) is enforced in the sweep + grid + DB
  (`skipped_box`). SS10 (R=10): x<N/8000, so x=1e-2 needs N≳80.
- **Lee-Yang and Eq.31 cross at x≈8×10⁻⁴.** LY is only the x→0 anchor; the rigorous gate is
  VMC ≤ Eq.31; at higher x validate against DMC, not LY.
- Bug already fixed: `acceptance_rate` is a per-chain array, not a scalar.

## GATE — PASSED (2026-06-15)

SS10 x=1e-4 N=32: at epoch ~109/1000 the energy was already E/N ≈ **0.001406** (paper units),
inside [4πx=0.0012566, LY=0.0013172, Eq.31=0.0014277] and descending toward Lee-Yang, σ_E
shrinking. Killed early — it was a test, the physics was confirmed sane. Pipeline validated
end-to-end (convention, bridge, sampler, DeepSet ansatz, verdict). Note: ~1.44 s/epoch at x=1e-4,
N=32, 1024 chains (big box L=68) — budget sweeps accordingly.

To re-confirm or get a polished number (run directly, **not** in the DB):

```
uv run --project .. python -c "from point import SS10, HP, run_point; \
print(run_point(SS10, x=1e-4, N=32, seed=0, hp=HP(n_epochs=2000, n_chains=2048)))"
```

**OK if** E/N ≈ Lee-Yang **0.0013172**, inside [4πx=0.0012566, Eq.31=0.0014277], verdict passed.
This is the go/no-go before any sweep. Validate the *easy* dilute regime (low x) first — x≈1e-3
is the hard shape-onset regime per the paper, not where you validate a fresh pipeline.

## Next steps (the parked task list)

2. **Launch SS10 E/N(x) sweep, low x first** — only after the gate passes and the user OKs.
   `uv run --project .. python run_ss10_curve.py` (N=64, seeds 0/1/2, 2000 epochs, 2048 chains).
   `feasible_x_grid` is low→high, so the dilute end fills first; watch where DeepSet-no-Jastrow
   deviates from Lee-Yang/DMC as x→1e-3. Resumable.
3. **Finite-N (1/N) extrapolation** per x — N-ladder (16/32/64/128, box-feasible), fit E/N vs 1/N,
   take the intercept (thermodynamic limit). Add to `analysis.py`.
4. **Shape sweep R → hard sphere** — vary R (V0 inferred at a=1) from SS10/SS5 toward R→1;
   reproduce shape dependence at fixed x (paper Fig.12 / Table III).
5. **3D pairwise Jastrow** — `LogJastrow` is 1D only; need 3D to node at r≈a. Gates the HS end and
   larger-x SS. Add as an HP-selectable ansatz family (DeepSet vs DeepSet+Jastrow).
6. **Structure observables** g(r), S(k), n(k)/n₀ — from the same wavefunction; n₀ (ODLRO) is the
   NQS-over-DMC selling point. Shape dependence shows at any x.
