# Laplacian Methods for VMC: AD vs Hutchinson

## A complete guide from first principles

---

## 1. Why we need the Laplacian at all

In Variational Monte Carlo (VMC), we minimize the expected energy of the system:

$$\langle E \rangle = \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi | \psi \rangle}$$

For a non-relativistic quantum system the Hamiltonian is:

$$\hat{H} = -\frac{1}{2}\nabla^2 + V(\mathbf{x})$$

where $\nabla^2 \equiv \Delta$ is the **Laplacian** — the sum of all second-order partial derivatives:

$$\Delta f(\mathbf{x}) = \sum_{i=1}^{d} \frac{\partial^2 f}{\partial x_i^2}$$

Our neural network outputs $\log|\psi(\mathbf{x})|$, so we never work with $\psi$ directly. The kinetic energy in log-space is:

$$T(\mathbf{x}) = -\frac{1}{2}\left(\Delta \log|\psi| + \left|\nabla \log|\psi|\right|^2\right)$$

The gradient term $|\nabla \log|\psi||^2$ is cheap — one backward pass gives all partial derivatives simultaneously. The **Laplacian** $\Delta \log|\psi|$ is the hard part, because it requires second-order derivatives.

> **Concrete example.** For $N=10$ bosons in 1D, the input is $\mathbf{x} \in \mathbb{R}^{10}$, so $d=10$. The Laplacian sums 10 second-order derivatives: $\partial^2/\partial x_1^2 + \ldots + \partial^2/\partial x_{10}^2$. For $N=100$ in 3D, $d=300$ and we need 300 such terms.

---

## 2. What automatic differentiation actually builds

Before comparing methods, you need to understand what AD does under the hood.

### 2.1 The computation graph

When you run a function like `model.apply(params, x)`, the computer executes a sequence of elementary operations (multiplications, additions, activations). AD frameworks like JAX record this sequence as a **computation graph** — a directed acyclic graph where:

- **Nodes** are intermediate values (activations, pre-activations, etc.)
- **Edges** encode how each value depends on the previous ones

**Example** for a 2-layer MLP with input $x$, hidden $h = \sigma(Wx + b)$, output $y = Vh$:

```
x → [linear W,b] → pre_h → [σ] → h → [linear V] → y
```

The graph has $O(L \times W)$ nodes for a network of depth $L$ and width $W$.

### 2.2 First-order AD: the backward pass (reverse-mode)

To compute $\nabla_x f(x)$ (all partial derivatives at once), AD does:

1. **Forward pass**: run the function, store all intermediate values.
2. **Backward pass**: propagate gradient signals from output to input by reversing the graph.

The gradient of the output with respect to every input is computed in a **single backward pass** — this is why `jax.grad` is so efficient. Cost: $O(L \times W)$ to store the graph, one forward + one backward pass.

### 2.3 Second-order AD: the problem

The Laplacian requires second-order derivatives: $\partial^2 f / \partial x_i^2$. Naively, you could differentiate the gradient $\nabla f$ with respect to $x$ again — this gives the full **Hessian matrix** $H \in \mathbb{R}^{d \times d}$.

But we don't need the full Hessian. We only need its **trace**: $\Delta f = \text{Tr}(H) = \sum_i H_{ii}$.

The full Hessian costs $O(d^2)$ memory to store (for $d=300$, that's 90,000 floats per sample — times the batch size). Computing it requires $d$ backward passes. This is `laplacian_full_hessian` and is only used for debugging.

We need smarter approaches.

---

## 3. The Jacobian-Vector Product (JVP): the key primitive

Both efficient methods are built on a single primitive: the **Jacobian-Vector Product (JVP)**, called `jax.jvp`.

### 3.1 What a JVP computes

Given $f: \mathbb{R}^n \to \mathbb{R}^m$, a point $x$, and a tangent vector $v \in \mathbb{R}^n$, the JVP computes:

$$\text{JVP}(f, x, v) = J_f(x) \cdot v$$

where $J_f(x)$ is the Jacobian matrix. This is the **directional derivative of $f$ at $x$ in the direction $v$**.

Crucially, this is done without ever materialising $J_f$. Cost: one forward pass through $f$, plus a forward pass through the linearised model.

### 3.2 JVP of the gradient = Hessian-vector product

Now consider $f = \nabla_x \log|\psi|$ (the gradient of our log-wavefunction). The JVP of this gradient in direction $v$ is:

$$\text{JVP}(\nabla f, x, v) = H_f(x) \cdot v$$

where $H_f$ is the Hessian of $f$. This is a **Hessian-vector product (HVP)** — achieved without computing $H$ explicitly.

In code:

```python
_, hess_v = jax.jvp(jax.grad(fn), (x,), (v,))
# hess_v = H_f(x) @ v
```

This is **forward-over-reverse AD**: the outer `jvp` is a forward pass through the inner `grad` (which is a backward pass). Two passes through the network, but memory is $O(L \times W)$ — not $O(d^2)$.

---

## 4. Method 1: Exact AD — `laplacian_forward_ad`

### 4.1 The idea

The diagonal of the Hessian $H_{ii} = \partial^2 f / \partial x_i^2$ can be extracted by probing with canonical basis vectors $e_i = [0, \ldots, 1, \ldots, 0]$:

$$H \cdot e_i = \text{column } i \text{ of } H \quad \Rightarrow \quad e_i^\top (H \cdot e_i) = H_{ii}$$

The Laplacian is then:

$$\Delta f(x) = \sum_{i=1}^{d} e_i^\top \cdot (H \cdot e_i) = \sum_{i=1}^{d} e_i^\top \cdot \text{JVP}(\nabla f, x, e_i)$$

### 4.2 Implementation

```python
def laplacian_single(x):
    n = x.shape[0]

    def body(i, acc):
        e_i = jnp.zeros(n).at[i].set(1.0)         # canonical basis vector
        _, hess_col = jax.jvp(jax.grad(fn), (x,), (e_i,))  # H @ e_i
        return acc + hess_col[i]                   # pick diagonal element

    return jax.lax.fori_loop(0, n, body, 0.0)     # sum over all d dimensions
```

### 4.3 Cost analysis

- **JVPs needed**: exactly `d` — one per dimension, run **sequentially** via `fori_loop`.
- **Memory per JVP**: $O(L \times W)$ — the AD graph of the network. No extra memory for the probes because we process one at a time.
- **Peak memory per sample**: $O(L \times W)$ — constant in $d$.
- **Time**: $d \times$ (cost of one JVP) = $d \times$ (forward + reverse pass through network).

**For $N=10$, $D=1$ ($d=10$):** 10 JVPs. For **$N=100$, $D=3$ ($d=300$):** 300 JVPs.

### 4.4 GPU behaviour

`fori_loop` compiles to a sequential loop in XLA — the GPU executes one JVP at a time. This **does not exploit GPU parallelism** across the $d$ probes. For large $d$, the GPU sits idle between probes.

---

## 5. Method 2: Stochastic Hutchinson — `laplacian_hutchinson`

### 5.1 The Hutchinson trace estimator

The core mathematical insight: for any random vector $z$ with $\mathbb{E}[z z^\top] = I$ (e.g., Rademacher $\pm 1$ or Gaussian),

$$\text{Tr}(H) = \mathbb{E}_{z}\left[z^\top H z\right]$$

**Proof**: $\mathbb{E}[z^\top H z] = \mathbb{E}[\text{Tr}(z^\top H z)] = \mathbb{E}[\text{Tr}(H z z^\top)] = \text{Tr}(H \mathbb{E}[z z^\top]) = \text{Tr}(H \cdot I) = \text{Tr}(H)$.

So we approximate:

$$\Delta f(x) = \text{Tr}(H_f(x)) \approx \frac{1}{k}\sum_{i=1}^{k} z_i^\top \cdot H_f(x) \cdot z_i$$

Each term $z_i^\top (H z_i)$ requires one HVP: `jvp(grad(fn), x, z_i)`.

### 5.2 Why Rademacher over Gaussian?

Both are unbiased. Rademacher vectors ($z_i \in \{-1, +1\}^d$ uniformly) have **lower variance** than Gaussian for trace estimation. The variance of the estimator is:

$$\text{Var}\left[\frac{1}{k}\sum z_i^\top H z_i\right] = \frac{1}{k}\left(\text{Var}[z^\top H z]\right) \sim O\left(\frac{\|H\|_F^2 - \sum_i H_{ii}^2}{k}\right)$$

Rademacher minimises this variance among isotropic distributions.

### 5.3 Implementation

```python
def laplacian_hutchinson(fn, xs, key, n_terms=10, distribution="rademacher"):
    dof = xs.shape[-1]
    z = 2 * jax.random.bernoulli(key, shape=(n_terms, dof)).astype(jnp.float32) - 1
    # z has shape (k, d) — k probe vectors

    def estimate_single(x):
        def single_probe(zi):
            _, hess_zi = jax.jvp(jax.grad(fn), (x,), (zi,))  # H @ zi, shape (d,)
            return jnp.dot(zi, hess_zi)                        # zi^T H zi, scalar

        return jnp.mean(jax.vmap(single_probe)(z))            # average over k probes

    return jax.vmap(estimate_single)(xs)                       # over batch
```

The key line is `jax.vmap(single_probe)(z)`: it runs all $k$ probes **in parallel** on the GPU.

### 5.4 Cost analysis

- **JVPs needed**: $k$ — run in **parallel** via `vmap`.
- **Memory per sample**: $O(k \times L \times W)$ — the AD graph is replicated $k$ times in parallel (vmap batches them).
- **Peak memory per sample**: scales with both $k$ and model size.
- **Time**: effectively 1 parallel batch of $k$ JVPs — on GPU, much less than $k$ sequential JVPs if $k$ fits in parallel.

**Speedup over forward_ad**: $d/k$ in JVP count, but the parallelism on GPU means wall-clock speedup can be much higher when $d$ is large.

---

## 6. Head-to-head comparison

### 6.1 Complexity table

| | `forward_ad` | `hutchinson` (k probes) | `full_hessian` |
|---|---|---|---|
| JVPs per sample | $d$ sequential | $k$ parallel | $d$ sequential |
| Memory per sample | $O(L \cdot W)$ | $O(k \cdot L \cdot W)$ | $O(d^2)$ |
| Memory scales with $d$? | No | No (scales with $k$) | Yes — **avoid** |
| Memory scales with model? | Yes ($L, W$) | Yes, $k$ times more | Yes |
| Exact? | Yes | No (variance $\sim 1/k$) | Yes |
| GPU-parallel? | No | Yes | No |

### 6.2 When $k = d$: both are equivalent

If you set `n_terms = dof`, Hutchinson uses $d$ parallel JVPs vs forward_ad's $d$ sequential JVPs. On GPU, Hutchinson can still be faster due to better hardware utilisation — but it uses $d$ times more memory. For large $d$ this will OOM.

### 6.3 The memory crossover point

forward_ad peak memory: $M_\text{AD} = C \cdot L \cdot W$ (one AD graph at a time)

hutchinson peak memory: $M_\text{H} = k \cdot C \cdot L \cdot W$ (k AD graphs in parallel)

Hutchinson uses more memory by a factor of $k$, independent of $d$. So:

- If your model is **small** ($L, W$ modest): Hutchinson's memory overhead is tolerable, large $k$ is fine.
- If your model is **large** (deep networks, wide DeepSets): even $k=5$ can double memory usage. Prefer forward_ad or small $k$.

---

## 7. How Hutchinson variance propagates into training

This is the critical subtlety.

### 7.1 The energy is noisy

The local energy is:

$$E_\text{loc}(\mathbf{x}) = T(\mathbf{x}) + V(\mathbf{x}) = -\frac{1}{2}\left(\widetilde{\Delta}\log|\psi| + |\nabla\log|\psi||^2\right) + V(\mathbf{x})$$

where $\widetilde{\Delta}$ is the Hutchinson estimate. Since $\widetilde{\Delta}$ is stochastic, $E_\text{loc}$ carries extra noise:

$$\text{Var}[E_\text{loc}] = \underbrace{\text{Var}_\text{physics}[E_\text{loc}]}_{\text{finite sampling}} + \underbrace{\text{Var}_\text{Hutchinson}}_{\sim d^2/k}$$

The Hutchinson variance scales as $d^2/k$ because the Hessian entries themselves grow with $d$.

### 7.2 The gradient is noisier

The VMC gradient uses $E_\text{loc}$ as a weight:

$$\nabla_\theta \langle E \rangle = 2\,\text{Re}\left\langle (E_\text{loc} - \langle E \rangle)\,\nabla_\theta \log\psi \right\rangle$$

Noisy $E_\text{loc}$ → noisy gradient → noisier parameter updates. This is equivalent to working with a **higher effective learning rate noise**, which:

- Is harmless far from the minimum (the signal-to-noise ratio is high)
- Becomes damaging near the minimum (gradient noise dominates the signal)

### 7.3 The estimator is unbiased — but variance matters

Hutchinson does **not** shift the minimum. The true ground state energy is still the minimum of $\langle E \rangle$ regardless of $k$. The problem is **reaching** it:

- $k=1$: very noisy Laplacian → energy fluctuates wildly, convergence is slow and may plateau above the true minimum
- $k=10$: moderately noisy → converges to within a few percent of AD result
- $k=d$: equivalent to AD but with parallel JVPs

### 7.4 The `std` field as a diagnostic

In your training output, `VMCState.std` = $\sqrt{\text{Var}[E_\text{loc}]}$ over the batch. When using Hutchinson, watch this field:
- If `std` is significantly larger than the AD baseline, $k$ is too small for the current system
- `std` should converge to a stable value as training progresses — if it keeps growing, the estimator variance is destabilizing training

---

## 8. Practical decision guide

### 8.1 By system size

| System | $d = N \times D$ | Recommendation |
|---|---|---|
| Tiny: $N \leq 5$, $D=1$ | $d \leq 5$ | forward_ad always — negligible cost |
| Small: $N \leq 20$, $D=1$ | $d \leq 20$ | forward_ad preferred; Hutchinson only for GPU speed |
| Medium: $N \leq 50$, $D \leq 3$ | $d \leq 150$ | Hutchinson $k=10$–$20$ during training |
| Large: $N > 50$, $D=3$ | $d > 150$ | Hutchinson $k=5$–$10$; watch memory |

### 8.2 By training phase

| Phase | Method | Reason |
|---|---|---|
| **Early training** (far from min) | Hutchinson, small $k$ | Gradient signal dominates noise; speed matters |
| **Mid training** | Hutchinson, $k=5$–$10$ | Balance between speed and stability |
| **Fine-tuning** (near convergence) | forward_ad | Eliminate estimator variance to reach tight energies |
| **Final energy evaluation** | forward_ad always | $\sigma_E$ must reflect physical fluctuations only, not estimator noise |
| **Debugging / reference** | full_hessian (small batch) | Exact check, but OOMs at production batch sizes |

### 8.3 The train-cheap, evaluate-exact strategy

The notebook's hypothesis: **train with Hutchinson $k=1$, evaluate final model with forward_ad**.

- **Works if**: the noisy gradients still guide the optimiser to the correct basin. The final AD evaluation recovers the true energy.
- **Fails if**: Laplacian noise is large enough to push the optimiser into a worse local minimum, or prevents convergence entirely.
- **When it's safe**: small systems ($d \lesssim 20$), moderate training budget where the AD energy is checked at the end.
- **Risky for**: large $d$ systems where $\text{Var}_\text{Hutchinson} \sim d^2/k$ can dominate the gradient signal.

---

## 9. Worked example: scaling to larger systems

Suppose you scale from $N=10$ ($d=10$) to $N=100$ ($D=3$, $d=300$).

**forward_ad**:
- JVPs: $300$ sequential
- Memory: $O(L \cdot W)$ — unchanged
- Wall-clock: 30× slower than at $d=10$ (sequential loop)

**Hutchinson $k=10$**:
- JVPs: $10$ parallel
- Memory: $10 \times O(L \cdot W)$
- Wall-clock: roughly unchanged (same number of parallel JVPs, just larger vectors)
- Speedup vs forward_ad: ~$300/10 = 30\times$ in JVP count, ~$10\times$–$20\times$ in wall-clock on GPU

At $d=300$, **Hutchinson with $k=10$ is the only practical option for training**. You would switch to forward_ad for the final few hundred epochs and for all reported results.

---

## 10. Summary

The choice reduces to a single tradeoff:

> **forward_ad gives you the exact Laplacian at cost $O(d)$ sequential JVPs. Hutchinson gives you a noisy estimate at cost $O(k)$ parallel JVPs. Use Hutchinson when $d \gg k$ and you're on GPU. Use forward_ad when you need exactness or $d$ is small.**

The JAX-specific angle: the difference between sequential (`fori_loop`) and parallel (`vmap`) matters enormously on GPU. Hutchinson's vmap is GPU-native; forward_ad's fori_loop is not. Even at the same FLOP count, Hutchinson can be 2–5× faster on GPU purely from better hardware utilisation — as seen in the notebook ($d=10$: 1:41 vs 2:54).

Memory is the constraint that limits $k$: each parallel probe holds a full copy of the AD graph. Large models ($L, W$ high) + large $k$ = OOM. forward_ad avoids this entirely by processing one probe at a time.
