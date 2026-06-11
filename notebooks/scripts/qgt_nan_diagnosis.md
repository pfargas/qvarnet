# QGT / Stochastic Reconfiguration NaN diagnosis

**Context:** `bose-hubbard_qgt_and_lrschedule.ipynb`, section 8 — `train(..., use_qgt=True)`
with the DeepSet model (`phi_hidden=[128], F_hidden=[128]`, ~17k parameters) and a batch of
4000 samples (2000 chains × 2 kept samples from `(chain_length 100 − thermalization 90) / thinning 5`).

**Verdict:** the SR implementation in `src/qvarnet/qgt.py` is mathematically correct.
Both failures (Cholesky → NaN immediately; direct/LU → divergence and NaN at step 4) have the
same root cause: **S is numerically indefinite / near-singular, and the `1e-6` diagonal shift is
far too small to fix it.**

---

## Why Cholesky gives NaN

1. **S is massively rank-deficient.** S is built from outer products of per-sample
   log-derivatives, so `rank(S) ≤ batch_size = 4000`, while S is 17k × 17k. At least ~13k
   eigenvalues are exactly zero in theory. In float32, round-off turns those zeros into random
   values of order `eps · λ_max` — some of them **negative**.

2. **The regularization can't compensate.** `compute_qgt` (`qgt.py:100`) adds an absolute shift
   of `1e-6`. Float32 eps is ~1.2e-7 relative, so if `λ_max(S) ~ 100` the round-off noise on the
   null space is ~1e-5 — an order of magnitude bigger than the shift. The matrix stays
   indefinite. (NetKet's default SR `diag_shift` is **0.01**, four orders of magnitude larger.)

3. **The formula is cancellation-prone.** `qgt.py:99` computes
   `S = ⟨OOᵀ⟩ − ⟨O⟩⟨O⟩ᵀ` — the `E[x²] − E[x]²` form, which loses precision catastrophically when
   `⟨O⟩` has large components (typical for the overall-scale parameter of an unnormalized ψ).
   The numerically safe form centers first: `dO = O − ⟨O⟩; S = dOᵀ dO / B`, which is PSD by
   construction up to round-off.

4. **JAX fails silently.** Under jit, `jax.scipy.linalg.cho_factor` on a non-PD matrix does not
   raise — it returns NaNs, which propagate through `cho_solve` into the parameters. Hence the
   `nan_checkpoint.msgpack` saved by `NaNCallback`.

## Why `direct` (LU) also NaNs — just later

From `checkpoints/bose-hubbard/history.csv`:

| step | energy | std | acceptance |
|---|---|---|---|
| 0 | 100.7 | 35.6 | 0.90 |
| 1 | 116.9 | 37.0 | 0.54 |
| 2 | 140.8 | 45.9 | 0.30 |
| 3 | −2.9×10⁷ | 4.4×10⁸ | 0.05 |
| 4 | NaN | NaN | 0.00 |

The NaN is not in the linear solver — the **training diverges**:

1. LU doesn't require positive-definiteness, so it returns finite numbers — but `S⁻¹∇E`
   amplifies the gradient along every near-null eigendirection by up to `1/ε = 10⁶`. The
   "natural gradient" is mostly huge-norm noise.
2. The parameter update (`lr 1e-3 × ~10⁶ amplification`) is enormous. Energy goes *up* each step
   and the Metropolis acceptance collapses 0.90 → 0.05 → 0.00 — the wavefunction becomes so
   pathological that walkers can't move.
3. By step 3 local energies are O(10⁸); at step 4 the stuck, degenerate samples produce
   overflow / 0÷0 in the local energy → NaN in the *energy*.

Cholesky and direct fail from the same disease at different points: Cholesky detects the
indefiniteness immediately inside the factorization; LU lets the divergence play out over a few
steps.

Ruled out: stale-checkpoint poisoning. `load_checkpoint` only reads `checkpoint.msgpack`, which
does not exist in `checkpoints/bose-hubbard/checkpoints/` — runs start fresh.

## Why SR is so much slower than Adam

Each step materializes the full (4000 × 17k) Jacobian via per-sample `vmap(grad)`, forms a dense
17k × 17k matrix (~1.2 GB float32, ~2 TFLOP matmul), then does an O(P³) ≈ 5 TFLOP solve. Adam
does one backprop. Inherent to the dense-S formulation, not a bug.

## Fixes

**Immediate (no source changes — this is the whole ballgame, not optional):**

```python
qgt_config = {"solver": "cholesky", "regularization": 1e-2, "learning_rate": 1e-3}
```

With `ε = 1e-2` the worst-case noise amplification is 100× instead of 10⁶× — the standard SR
operating regime. Anneal ε down (1e-2 → 1e-4) once training is stable.

Also worth doing in the notebook:

- `jax.config.update("jax_enable_x64", True)` at the top — shrinks the round-off floor by ~9
  orders of magnitude at ~2× cost.

**Diagnostic — the smoking gun (run in a cell before training):**

```python
from qvarnet.qgt import compute_natural_gradient, QGTConfig
from jax.flatten_util import ravel_pytree

for reg in [1e-6, 1e-2]:
    ng, _ = compute_natural_gradient(
        params, batch, apply_fn, grads, QGTConfig(solver="direct", regularization=reg)
    )
    print(reg, "‖natural grad‖ =", jnp.linalg.norm(ng),
          " vs ‖plain grad‖ =", jnp.linalg.norm(ravel_pytree(grads)[0]))
```

Expected: the 1e-6 natural-gradient norm is orders of magnitude larger than the plain gradient —
that's the update destroying the wavefunction. Likewise
`jnp.linalg.eigvalsh(S)` on one batch should show a smallest eigenvalue more negative than
−1e-6.

**Later (source changes to `src/qvarnet/qgt.py`):**

- Center before forming S: `dO = O − ⟨O⟩; S = dOᵀ dO / B` (PSD by construction).
- **minSR trick** for speed: with batch B ≪ params P, solve the equivalent B×B system in sample
  space instead of P×P — identical update, ~80× fewer FLOPs here. The existing `"gmres"` solver
  doesn't help as written since it still materializes the full S.
- Consider a relative shift (`ε · mean(diag(S))`) and/or natural-gradient norm clipping.

**Unrelated note:** in the QGT path, `_apply_natural_gradient_step` (`training_step.py:106`)
replaces params without incrementing `state.step`, so a run resumed from a QGT checkpoint
restarts its epoch count at 0.
