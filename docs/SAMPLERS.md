# The qvarnet sampler — Metropolis-Hastings, proposal families, and how jit handles it

*Written 2026-07-11, alongside the Proposal refactor. Code references are to
`src/qvarnet/samplers/`; empirical numbers from
`calogero-sutherland/sr_debug/proposal_mixing_probe.log` and the micro-benchmarks
reproduced in §5.*

---

## 1. What one MH step does (`kernel.py::mh_kernel_log`)

The target is $\pi(x) = |\psi_\theta(x)|^2$, known only through
$\log P(x) = 2\log|\psi(x)|$ (unnormalised — MH never needs the normalisation).
One step:

1. Split the PRNG key: proposal noise + accept/reject draw.
2. `proposal.propose(key, x, step_size)` → candidate $x'$ and `log_q_correction`.
3. If `box_L > 0`, fold $x'$ into $[0, L)$ (PBC; symmetric proposals stay symmetric
   on the torus).
4. Evaluate $\log P(x')$ — **the one model call per step**; everything else is noise
   generation and arithmetic.
5. Accept with probability
   $A = \min\!\big(1,\, e^{\beta\,(\log P(x') - \log P(x)) + \log q_{\mathrm{corr}}}\big)$,
   implemented as accept iff $\log u < \log A$, $u \sim U(0,1)$.
6. On accept, move to $x'$; on reject, **stay at $x$ and count it again** — the
   repeat is what makes the histogram converge to $\pi$; discarding rejections
   would bias it.

$\beta$ is an inverse temperature (traced, default 1). The parallel-tempering
replicas call this same kernel with their own $\beta$ and store the *untempered*
$\log P$ (`parallel_tempering.py::local_step`) — PT is not a second sampler
implementation, just this kernel at $\beta \ne 1$ plus swap moves.

## 2. The Hastings correction — why the code carries a term that is currently always zero

Detailed balance, $\pi(x)\,T(x\to x') = \pi(x')\,T(x'\to x)$, with the transition
split into proposal × acceptance $T = q(x'|x)A(x'|x)$, is satisfied by the
**Metropolis–Hastings** acceptance

$$
A = \min\!\left(1,\;
\frac{\pi(x')\,q(x\,|\,x')}{\pi(x)\,q(x'\,|\,x)}\right),
\qquad
\log q_{\mathrm{corr}} \;=\; \log q(x\,|\,x') - \log q(x'\,|\,x).
$$

The correction answers: *how much easier is the reverse move than the forward one?*
If the proposal has any directional preference, the acceptance must charge for it —
otherwise the chain converges to $\pi \times (\text{wherever } q \text{ drifts})$, a
**silently biased** sampler that produces plausible wrong energies rather than an
error.

**Every current family is symmetric**, so the correction is exactly 0:

- Gaussian / uniform displacement: $q(x'|x)$ depends only on $x'-x$ and is even.
- Subset moves: the subset is chosen uniformly, *independent of the current
  position*, and the displacement on the chosen coordinates is symmetric — so the
  composite kernel is symmetric too.

Symmetric $q$ reduces MH to plain Metropolis (the 1953 special case) — that's the
regime we're in today.

**Where it stops being zero — the reason the term exists**: MALA (Metropolis-adjusted
Langevin),

$$
x' = x + \tfrac{\sigma^2}{2}\nabla \log \pi(x) + \sigma\eta ,
$$

drifts proposals uphill. Then $q(x'|x)$ is a Gaussian centred at
$x + \mathrm{drift}(x)$ while $q(x|x')$ is centred at $x' + \mathrm{drift}(x')$ —
different centres, correction ≠ 0, and omitting it over-samples the modes with no
warning. MALA is a natural future move for qvarnet because $\nabla \log|\psi|$ is the
same quantity the QGT machinery already computes. Other nonzero cases: any
position-dependent step size, energy-guided particle selection, worm/cluster moves.

Cost of carrying the term now: `+ 0.0` in the log-acceptance, constant-folded away by
XLA. Benefit: adding an asymmetric proposal later is *one* `propose()` method — no
kernel change, no chance of shipping the biased-sampler bug. (For calibration on how
easy that genre of bug is: the pre-refactor API let a caller feed normal draws where
the kernel expected a uniform accept draw — $\log(\text{negative}) = \mathrm{NaN}$ →
silent auto-reject bias. `test_tdvp.py` did exactly that for an unknown time.)

## 3. Proposal families and when to use which

| family | moves | use |
|---|---|---|
| `GaussianMove()` (default) | all N·d coordinates | small N; the classic choice |
| `UniformMove()` | all N·d coordinates | as above, box-shaped noise |
| `ParticleSubsetMove(n_move, n_dim)` | all d coords of `n_move` random particles | **N ≳ 30**; `n_move=1` is the validated setting |
| `DoFSubsetMove(k)` | `k` random scalar coordinates | systems where "particle" isn't the natural unit; ≡ ParticleSubsetMove in 1D |

Why subset moves: a full-configuration move changes N·d coordinates at once, so its
log-prob change grows with N and the 50%-acceptance step shrinks — the chain
diffuses ever more slowly. Moving one particle keeps a large step affordable.
Physically, moving one particle at a time is also the natural unit (it is the
standard in fermionic VMC, and would enable Sherman–Morrison determinant updates
later); moving a single *axis* of a particle (`DoFSubsetMove` in d > 1) is
statistically exact but explores anisotropically — prefer whole-particle moves
unless the problem itself is anisotropic.

**Measured on the trained CS N=30 state** (each proposal at its own adapted
50%-acceptance step; IAT of Σx²; `proposal_mixing_probe.log`):

| proposal | adapted step | IAT median | eff. samples / 1000 steps |
|---|---|---|---|
| Gaussian (all 30) | 0.038 | 412 | 1.2 |
| **subset n_move=1** | **0.459** | **192** | **2.6** |
| subset n_move=3 | 0.164 | 382 | 1.3 |
| subset n_move=10 | 0.070 | 460 | 1.1 |

The full move is collapsed at N=30 (step 0.038!). Single-particle moves give ~2.1×
more effective samples per model evaluation; intermediate subset sizes wash out.
Expect the gap to grow with N.

Usage:

```python
train(..., sampler_params={
    "step_size": 0.5, "chain_length": 21, "thermalization_steps": 20,
    "thinning_factor": 1,
    "proposal": ("particle-subset", {"n_move": 1}),   # or "gaussian", or an instance
})
```

## 4. Randomness: per-step keys, not pre-generated buffers

`mh_chain` splits one key into per-step keys inside the `lax.scan`; each step splits
again into proposal/accept keys. The old design pre-generated a
`(n_chains, n_steps, dof+1)` random array whose *layout encoded the proposal family*
(dof noise columns + 1 accept column) — that coupling is what made new proposal
families invasive, dominated peak memory (hence the now-deleted `block_size`
machinery), and allowed the malformed-buffer bug above. Consequence of the switch:
the RNG stream changed, so pre-refactor runs are not bit-reproducible.

## 5. "The proposal is static, not jitted" — why that is the fast path

This worry has it backwards: **static doesn't mean un-jitted, it means baked into
the jitted program.**

When `sample_and_process` is traced, JAX calls `proposal.propose(...)` *as ordinary
Python, once, at trace time*. The jnp operations it emits (noise, mask, scatter) are
recorded into the same XLA graph as the model evaluation and fused/optimised with
it. At run time there is **zero Python involvement, zero dispatch, zero indirection**
— the proposal's arithmetic is compiled instructions inside the scan body, exactly
like the hardcoded Gaussian was before the refactor. This is the same mechanism the
codebase already relies on for `prob_fn` (a Python function!), the Hamiltonian, and
`TrainingConfig`/`SamplingConfig` static args.

What "static" actually costs — and it's compile-time, not run-time:

- **One compilation per distinct proposal value.** The frozen dataclasses have
  value-based `__eq__`/`__hash__`, so `GaussianMove() == GaussianMove()` hits the same
  jit cache entry; `ParticleSubsetMove(1)` vs `ParticleSubsetMove(2)` are two entries.
  A training run uses one proposal → one compile, amortised over all epochs.
- The trap to avoid: *changing* proposal parameters mid-run (e.g. annealing `n_move`
  per epoch) retraces on every change. `step_size` is traced, so step adaptation is
  free — that split (family static, scale traced) is deliberate.

The alternative — making the proposal a traced value — would require encoding the
family as an integer and `lax.switch`-ing over *all* families inside the kernel:
every branch compiled into every program, a runtime branch per step, and no
shape/constant specialisation. Strictly worse.

Measured (RTX, 512 chains × 30 dof × 1000 steps, steady-state after compile):

| target | GaussianMove | ParticleSubsetMove(1) |
|---|---|---|
| trivial logP (proposal-bound) | 20.2 M chain-steps/s | 19.1 M |
| MLP-128 logP (realistic) | 19.1 M | 15.1 M |

i.e. ≤ ~20 % per-step overhead in the worst case (mask + scatter work), against the
2.1× IAT gain — a net ~1.75× in effective samples per wall-second. (`n_move=1`
selection uses `randint` rather than a full permutation — a static branch in
`propose`; with the permutation it was 2.3× slower on the trivial target.)

## 6. File map

| file | contents |
|---|---|
| `samplers/kernel.py` | `Proposal` families, `resolve_proposal`, `mh_kernel_log` (β-aware), `mh_chain` |
| `samplers/step.py` | `sample_and_process` — vmapped chains, burn-in/thinning |
| `samplers/parallel_tempering.py` | replica ladder over the shared kernel + swap moves |
| `samplers/diagnostics.py` | IAT / ESS / split-chain stats (used by the §3 probe) |
| `config/training_setup.py` | `SamplingConfig.proposal` (resolved to an instance at construction, stays hashable/jit-static) |
