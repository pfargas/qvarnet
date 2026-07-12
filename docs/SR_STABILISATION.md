# Stochastic Reconfiguration in qvarnet — equations, guards, and why the gradient clip had to go

*Written 2026-07-11, after the guard-binding experiments. All code references are to the
uncommitted state of branch `clean-slate`; probe scripts and logs live in
`calogero-sutherland/sr_debug/`.*

This note documents (1) the SR/natural-gradient math as implemented, (2) the float32
numerical stack that makes it reliable on a consumer GPU, (3) the guard system — and the
measured proof that a Euclidean gradient clip *silently disables* SR, which is the
"really weird clipping" story, and (4) the experimental record, including the
stale-kernel incident that made the fixed code look broken.

---

## 1. Setup and notation

VMC minimises the variational energy of $\psi_\theta$ with samples $x \sim |\psi_\theta|^2$:

$$
E(\theta) = \frac{\langle\psi|\hat H|\psi\rangle}{\langle\psi|\psi\rangle}
          = \mathbb{E}_{x\sim|\psi|^2}\big[E_{\mathrm{loc}}(x)\big],
\qquad
E_{\mathrm{loc}}(x) = \frac{(\hat H\psi)(x)}{\psi(x)} .
$$

With the log-derivatives (models output $\log|\psi|$, so these are plain parameter
gradients of the model output)

$$
O_k(x) = \frac{\partial \log|\psi_\theta(x)|}{\partial \theta_k},
\qquad
\bar O_k = O_k - \langle O_k\rangle ,
$$

the energy gradient (**force**) and the **quantum geometric tensor** (Fisher/Fubini–Study
metric of the state manifold) are

$$
F_k = \partial_{\theta_k} E = 2\,\big\langle (E_{\mathrm{loc}} - \langle E\rangle)\,\bar O_k \big\rangle,
\qquad
S_{kl} = \langle O_k O_l\rangle - \langle O_k\rangle\langle O_l\rangle
       = \langle \bar O_k \bar O_l \rangle .
$$

* Force: the loss surrogate in `training_step.py::energy_and_grads`
  (`src/qvarnet/vmc/training_step.py:49`).
* QGT: `src/qvarnet/geometry/qgt.py:168` (`compute_qgt`).

**Stochastic reconfiguration** (= natural gradient = imaginary-time evolution projected
onto the variational manifold) replaces the Euclidean step with the metric-corrected one:

$$
\theta \leftarrow \theta - \eta\, \delta, \qquad S\,\delta = F .
$$

Geometrically: $\delta$ is steepest descent measured in *state* distance
$\mathrm{d}s^2 = \delta\theta^\top S\, \delta\theta$, not parameter distance. This is why
everything below must be phrased in the $S$-metric — and why phrasing anything in the
Euclidean metric goes wrong. The same linear system $S\dot\theta = -F$ is t-VMC, which is
the thesis path; everything here is directly reusable there.

---

## 2. The float32 numerical stack

Each item exists because a specific failure was observed (dates in
`sr_debug/` logs and the memory changelog).

### 2.1 Centred QGT build — `qgt.py:168`

$$
S = \frac{1}{B}\,\bar O^\top \bar O
\quad\text{instead of}\quad
S = \langle OO^\top\rangle - \langle O\rangle\langle O\rangle^\top .
$$

Mathematically identical; numerically not. The uncentred form subtracts two large,
nearly equal matrices. For a Jastrow, $O_\lambda = \sum_{i<j}\log|x_i-x_j|$ has a huge
*mean*, so the cancellation happens at the scale of $\langle O\rangle^2$ and float32
loses the difference: $S$ acquired **spurious negative eigenvalues**, and an LU solve
returned a *non-descent* direction (measured $\delta^\top F = -3.7\times10^5$ — the
optimizer confidently walked uphill). The centred product is PSD by construction up to
rounding (`qgt.py:183`).

### 2.2 Jacobi preconditioning ≡ Levenberg–Marquardt regularisation — `qgt.py:229`

The regularised solve is performed in the unit-diagonal basis, with $D=\mathrm{diag}(S)$:

$$
\big(D^{-1/2} S D^{-1/2} + \varepsilon I\big)\, y = D^{-1/2} F,
\qquad
\delta = D^{-1/2} y
\;\;\Longleftrightarrow\;\;
\big(S + \varepsilon\,\mathrm{diag}(S)\big)\,\delta = F .
$$

Why not a plain shift $S+\varepsilon I$:

* **Absolute $\varepsilon$** fails for large-$O$ models (Jastrow: $\mathrm{diag}(S)\sim10^4\!-\!10^6$):
  the shift is below the float32 *rounding of the eigenvalues themselves*
  ($\lambda_{\max}\cdot 2^{-23}$), so Cholesky still meets numerically-negative pivots →
  NaN. This killed the first proof run.
* **Relative $\varepsilon\cdot\mathrm{mean}(\mathrm{diag})$** fails for small-$O$ models
  (TFIM test: mean diag $1.26\times10^{-2}$ → shift $1.26\times10^{-5}$, below the
  rounding noise $8.5\times10^{-6}$).

The LM form $\varepsilon\,\mathrm{diag}(S)$ is *per-direction scale-invariant*: the same
$\varepsilon$ means the same relative damping whether $O_k \sim 1$ or $\sim 10^5$, and
the Jacobi scaling collapses the condition number so float32 factorisations are enough —
**no float64 needed** on the RTX card (1/64 fp64 rate).

$\varepsilon = 10^{-2}$ is the validated default. Empirical floor: at
$\varepsilon=10^{-3}$ in float32 **all 600/600 solves failed** on the CS N=30 model
(guard returned zero steps; `sr_debug/guard_probe.log`). Do not anneal below $10^{-2}$
in float32.

### 2.3 Dead directions get exactly zero step — `qgt.py:266`

Any direction with $\mathrm{Var}(O_k)=0$ (e.g. the additive constant of $\log|\psi|$
that every `LogWavefunction` bias produces) does not change the physical state and has
exact force $F_k = 0$. Its sampled $F_k$ is pure float32 rounding residue. A
floor-based scheme divides that residue by $(\varepsilon\cdot\text{floor})$ and
amplifies it (observed spurious $\delta_k = -8.3$ on the bias). Instead,
$d^{-1/2}_k = 0$ where $\mathrm{diag}_k \le 10^{-9}\max(\mathrm{diag})$: exactly zero
step, which also matches minSR (whose step lives in $\mathrm{row}(\bar O)$ and is
identically zero there).

### 2.4 minSR: the sample-space dual — `qgt.py:341`, auto-selected at `qgt.py:121`

With $\bar O \in \mathbb{R}^{M\times P}$ ($M$ samples, $P$ parameters) and the *same*
Jacobi scaling $\tilde O = \bar O D^{-1/2}$, the push-through identity

$$
\Big(\tfrac{1}{M}\tilde O^\top \tilde O + \varepsilon I_P\Big)^{-1}\tilde O^\top
= \tilde O^\top \Big(\tfrac{1}{M}\tilde O \tilde O^\top + \varepsilon I_M\Big)^{-1}
$$

turns the $P\times P$ solve into an $M\times M$ one. Using
$F = \tfrac{2}{M}\bar O^\top e$ with $e_i = E_{\mathrm{loc}}(x_i)-\bar E$:

$$
\delta = \frac{2}{M}\, D^{-1/2}\,\tilde O^\top\, (T+\varepsilon I_M)^{-1} e,
\qquad
T = \tfrac{1}{M}\,\tilde O\tilde O^\top .
$$

This is the **same regularised step** as full SR (exact identity; the float32 residual
in the equivalence test is $\mathrm{cond}\sim(P/\varepsilon)\cdot2^{-23}$ round-off,
`tests/test_minsr.py`), but costs $O(M^2P + M^3)$ instead of $O(P^2M + P^3)$ and is
full-rank in sample space. It matters because the sampled $S$ has
$\mathrm{rank}\le M$: the CS runs sit at $P=4099$, $M=4096$ — full $S$ is *guaranteed*
rank-deficient there. `solver="auto"` picks minSR when $P>M$, else Cholesky
(Cholesky over LU because it fails loudly on a broken matrix instead of silently
returning garbage).

---

## 3. The guard system — and why the Euclidean clip is poison

Hard rule ([memory: never-clip-local-energy]): $E_{\mathrm{loc}}$ **clipping is
forbidden** — it biases the estimator. All spike protection must act on the *update*,
never on the energy.

### 3.1 Fisher trust region — `qgt.py:303`

Rescale the natural gradient so the **state change per optimizer step** is bounded in
the metric that was actually solved (the LM one):

$$
\|\Delta\theta\|_S = \eta\,\sqrt{\delta^\top\big(S+\varepsilon\,\mathrm{diag}(S)\big)\,\delta}
\;\le\; \texttt{max\_state\_change} .
$$

* The physical knob is `QGTConfig.max_state_change` (default 0.1); the direction-space
  cap $\Delta = \texttt{max\_state\_change}/\eta$ is derived in
  `QGTConfig.resolve_trust_region` (`qgt.py:91`). See §3.3 for why the derivation —
  not a raw $\Delta$ default — is load-bearing.
* In minSR the quadratic form is evaluated without building $S$:
  $\delta^\top S\delta = \|\bar O\delta\|^2/M + \varepsilon\sum_k \mathrm{diag}_k\delta_k^2$
  (`qgt.py:390`).
* **Failed-solve guard**: non-finite $\delta$ or $\delta^\top S\delta < 0$ → exactly
  zero step. A wasted epoch instead of a poisoned parameter vector; the next batch
  retries. This is what converted the $\varepsilon=10^{-3}$ disaster into 600 harmless
  no-op epochs.
* Measured behaviour (probe 2 / scratch probe): the region **binds hard early and
  releases itself** — 99–100 % of epochs in the first quarter of a run, 0 % near
  convergence. It is a spike guard, not a permanent throttle.

### 3.2 Why `clip_by_global_norm` strangles SR (the core weirdness)

The whole point of $\delta = S^{-1}F$ is to *boost flat directions*: an eigendirection
$v_i$ of $S$ with eigenvalue $s_i$ gets $\delta_i = F_i/s_i$. Flat directions
($s_i \to \varepsilon\,\mathrm{diag}$-scale) are directions where large parameter moves
barely change the state — so SR legitimately takes **huge Euclidean steps** along them.
The two norms are related by

$$
\|\delta\|_S^2 = \sum_i s_i\,\delta_i^2
\qquad\text{vs}\qquad
\|\delta\|_2^2 = \sum_i \delta_i^2 ,
$$

so $\|\delta\|_2 \le \|\delta\|_S / \sqrt{s_{\min}}$ with equality-ish when the step is
concentrated in flat directions — which for an over-parametrised MLP it always is.
Measured on CS N=30: $\|\delta\|_S \sim 10^2\!-\!10^3$ (Fisher, pre-cap) while
$\|\delta\|_2 \sim 3\times10^3\!-\!9\times10^3$ (Euclidean).

`optax.clip_by_global_norm(c)` rescales the update by $c/\|\delta\|_2$ whenever
$\|\delta\|_2 > c$. With $c=10$ that is a **~300× shrink of a step the trust region had
already approved** — applied *after* the metric-correct guard, in the *wrong metric*,
every single epoch:

| run (600-epoch SR finetune, `guard_probe.log`) | clip binds | trust binds | E tail |
|---|---|---|---|
| `sr-base` (clip 10 + trust 0.1) | **100 %** | 100 % | 769 ± 32 — **flat, λ frozen** |
| `sr-noclip` (trust 0.1 only) | — | 91.5 % | 733.4 ± 4.1 — descends, matches Adam with 3× cleaner tail |
| `sr-lr1e-2` (clip 10, η=10⁻²) | **100 %** | 100 % | 773 ± 42 — still flat: η is irrelevant while the clip binds |

The clip was originally a hand-hardcoded safety net for **SGD**, where it is the right
tool: an SGD update *is* Euclidean, so a Euclidean cap is metric-consistent (the
SGD+clip probe converged CS N=30 from scratch; that is precisely why the natural
mistake of carrying it over to SR was made). For SR it is a category error: it measures
the step in a metric the optimizer is explicitly designed to ignore.

**Current state**: `grad_clip_norm=None` everywhere by default. The optax chain that
applies it when explicitly set survives at `src/qvarnet/vmc/train.py:123` for
opt-in use.

### 3.3 The units trap (bit three times)

A trust region can be stated as a cap on the *direction* $\|\delta\|_S \le \Delta$ or on
the *update* $\eta\|\delta\|_S \le \texttt{msc}$. These differ by $\eta$, and every time
a number validated in one convention was used in the other, the run silently got a
$1/\eta$-scaled budget:

1. First proof run: 0.1 (validated as *update* cap) passed as the *direction* cap →
   1000× over-throttled.
2. `sr_train` fixed it with an explicit conversion — but the raw
   `QGTConfig(trust_region=...)` stayed in direction units.
3. After making the trust region a default, `QGTConfig(learning_rate=0.02)` in a test
   inherited $\Delta=100$ calibrated for $\eta=10^{-3}$ → a 2.0 state-change budget,
   20× the validated one → spike blow-through.

Resolution: the config stores the **physical** quantity
(`QGTConfig.max_state_change`, Fisher state change per step) and derives
$\Delta = \texttt{msc}/\eta$ at solve time (`qgt.py:91`); `trust_region` remains as a
direction-units override (needed for optax LR schedules, where dividing by a callable
is impossible — attempting it raises). Pinned by
`tests/test_sr_stack.py::test_trust_region_units_resolve`.

### 3.4 Per-epoch guard diagnostics

`compute_natural_gradient(_minsr)` return an `info` dict, threaded through
`compute_step` (`training_step.py:132`) into `result.history`
(`train.py:573`):

| key | meaning |
|---|---|
| `fisher_norm` | $\sqrt{\delta^\top S_{LM}\delta}$ *before* the trust rescale |
| `trust_scale` | factor applied; $<1$ ⇒ trust region bound this epoch |
| `nat_grad_norm` | $\|\delta\|_2$ after trust, before any clip; $>$ `grad_clip_norm` ⇒ clip would bind |
| `solve_ok` | 0.0 ⇒ failed solve, zero step taken |

`result.history.get("trust_scale")` etc. — this instrumentation is how the table in
§3.2 was measured. If SR ever "mysteriously stalls" again, look here first.

---

## 4. Experimental record (CS N=30, L=0.8, MLP[128]+Gaussian envelope+Jastrow)

Exact ground state: $E_0 = N(1+L(N-1)) = 726$, $\lambda^\ast = L = 0.8$,
$\alpha^\ast = 1/\sqrt2 \approx 0.7071$ (envelope is $-\alpha^2\sum x_i^2$,
`models/envelopes.py`). Scripts + logs: `sr_debug/guard_probe.py`,
`guard_probe2.py`, `scratch_probe.py` and their `.log`s.

**Finetune** (2000 epochs from the same unconverged Adam endpoint, λ=0.591):

| run | E tail (exact 726) | λ | notes |
|---|---|---|---|
| Adam continue | 726.90 ± 3.96 | 0.769 | |
| SR, msc=0.1 | 727.19 ± 0.30 | **0.8002** | trust binds 99 % → 0 %, self-releasing |
| SR, msc=0.1, η=10⁻² | 728.7 ± 3.4 | 0.8007 | binds 100 % forever → noise floor; keep η=10⁻³ |
| **SR, msc=0.3** | **726.55 ± 0.24** | **0.8002** | binds 31 % early then free; best overall |

**From scratch** (1500 epochs, 4096 chains, cusp-exact λ_init=L):

| run | E tail | λ | α |
|---|---|---|---|
| Adam, lr=10⁻² | 1306 ± 1751 | 0.48 (dove to 0.38 first) | 0.66 |
| SR, msc=0.1 | 726.15 ± 0.08 | 0.800 | 0.705 |
| **SR, msc=0.3** | **726.07 ± 0.06** | **0.800** | **0.706** |

Two physics points worth keeping:

* **Adam must break the cusp to move** (λ: 0.8 → 0.38 → slow recovery); SR keeps
  λ pinned near the exact value throughout and converges an order of magnitude faster
  and ~10³× more precisely. This is the natural gradient correctly recognising that λ
  is a stiff direction (huge $\mathrm{diag}(S)_\lambda$) and stepping it carefully.
* **SR from scratch on singular interactions still needs λ_init = L**: the cusp
  residual in $E_{\mathrm{loc}}$ is $2[\lambda(\lambda-1) - L(L-1)]/r^2$ — nonzero
  whenever $\lambda \ne L$, producing heavy-tailed spikes from close pairs that no
  update-side guard can remove (and estimator-side clipping is forbidden).

Cost: SR ≈ 3× Adam per epoch (minSR; was 6× with the full $P\times P$ build).

---

## 5. The stale-kernel incident (why the "fixed" cell 22 looked horribly wrong)

The re-run of the recipe-based cell 22 (2026-07-11 13:18–13:20) produced E climbing
2477 → 7050 over 5000 epochs — *with the fixed code on disk*. Diagnosis:

* Kernel pid 9172 started **12:08**, before the fixes landed (`qgt.py` 12:38,
  `recipes.py` 12:39).
* `src/qvarnet/__init__.py:11` does `from .recipes import adam_train, sr_train` — so
  the **pre-fix `sr_train` (with `grad_clip_norm=10` default) was cached in
  `sys.modules`** from the morning's first qvarnet import. Re-running the imports cell
  is a cache hit; nothing reloads.
* The run therefore executed the §3.2 `sr-base` configuration: clip binds every epoch,
  optimizer effectively frozen. Its own checkpoint proves it:
  `checkpoints/cs_sr/checkpoints/final_state.msgpack` has **α = 0.134, λ = 0.543 after
  5000 epochs** (should be 0.707 / 0.800).
* The *rising* energy is not the state getting worse: the walkers start as a σ=0.5
  speck inside a very wide α≈0.1 state whose true energy is ≈ 7100 (the Adam control's
  first-quarter mean at the same settings is 7098). The measured E climbs as the
  walkers equilibrate outward into a state the frozen optimizer never narrows —
  hence also the "best epoch: 0" artifact.

**Rule**: after editing qvarnet source, notebook kernels must be restarted (no
autoreload is configured, and package-level imports in `__init__.py` mean even
"fresh" `from qvarnet.recipes import ...` lines hit the cache).

---

## 6. Practical summary

```python
from qvarnet.recipes import adam_train, sr_train

r1 = train(shape=shape, model=model, hamiltonian=ham,
           **adam_train(n_epochs=..., learning_rate=1e-2, checkpoint_path=...))
r2 = train(shape=shape, model=model, hamiltonian=ham,
           **sr_train(n_epochs=..., prev_result=r1, checkpoint_path=...))
```

**Optimizer/SR separation** (design since 2026-07-11): SR is a gradient
*preconditioner* — `compute_step` hands $\delta = S^{-1}F$ to whatever optax optimizer
was passed to `train()`, which is honoured as the update rule (`train.py` no longer
overrides it). `optax.sgd(η)` = classic SR $\theta \leftarrow \theta - \eta\,\delta$
(what `sr_train` passes); `optax.adam` = SR-preconditioned Adam. Caveat: the trust
region's `max_state_change` semantics are exact only for
`sgd(qgt_config.learning_rate)` — an adaptive optimizer rescales per-parameter after
the Fisher cap, so there it acts as a spike guard rather than exact state-change
control (set `trust_region` directly in that case).

Measured (`sr_debug/sr_adam_probe.log`, 1000-epoch finetune from a λ=0.659 Adam
endpoint): SR-preconditioned **Adam(10⁻³)** reached 726.84 ± 0.20 with λ = 0.8000,
*ahead of* classic SR-SGD (728.55 ± 0.46, λ = 0.7994) on the same budget; Adam(10⁻⁴)
is under-stepped (734.6 ± 7.2). No instability observed from the approximate trust
semantics.

| knob | default | guidance |
|---|---|---|
| `learning_rate` (η) | 10⁻³ | the classic SR step: `sr_train` builds `sgd(η)` from it, and the trust cap is derived as msc/η. Raising it does **not** speed up the guarded phase (the trust region caps the state change, not η) and prevents the region from releasing near convergence |
| `max_state_change` | 0.1 | the one true speed/robustness knob; 0.3 validated ~3× faster on CS N=30, scratch and finetune |
| `regularization` (ε) | 10⁻² | float32 floor; 10⁻³ fails every solve |
| `grad_clip_norm` | **None** | do not enable for SR (§3.2); it is an SGD-only tool |
| `solver` | `"auto"` | minSR when P > M, else Cholesky; never `"direct"`/LU |
| λ_init (Jastrow) | — | must equal L for SR *from scratch* on singular interactions |

Everything in §2–§3 transfers unchanged to t-VMC ($S\dot\theta = -F$ per step), which
is why the stack was built this carefully in the first place.
