# Is the network doing anything? (Jastrow vs. NN — where the NQS approach actually pays off)

**Short answer (current setup):** the analytic Jastrow is doing essentially everything; the
one-body DeepSet is doing essentially nothing. We reproduce the paper's HNC/EL result almost
exactly — which is a good *validation* but **not a result**.

---

## Why we get HNC

The paper's HNC/EL is "the best possible **two-body** Jastrow" (the optimized pair correlation,
summed via hypernetted-chain). Our ansatz is `SR-Jastrow × one-body-DeepSet`:

- The **analytic Jastrow** is the zero-energy two-body scattering solution. At dilute x it is already
  near-optimal as a two-body factor, so **SR ≈ EL** — that alone lands us at the HNC level.
- The **DeepSet is one-body** (`F(Σ φ(rᵢ))`), so it *cannot* improve the two-body correlation even
  in principle. It only adds one-body / mean-field structure, which a homogeneous gas doesn't need.
- At **dilute x**, beyond-two-body correlations are higher order in x and tiny — so even a perfect
  network would have almost nothing to capture.

Net: we reproduce HNC because the Jastrow supplies the HNC-level physics and there is no residual
left for the net. **Quantitative check:** run `--no-network --jastrow` next to the net-on
`--jastrow` run at the same point — they should agree within error bars. That is the precise
version of "the net does nothing."

---

## Is it worth applying the network? As set up — no.

The variational hierarchy (from the paper) is

```
DMC  ≤  EL  ≤  SR        (all upper bounds; DMC ≈ exact)
```

We sit at the **EL/SR level**. A neural ansatz is only worth it if it gets **below EL, toward DMC** —
i.e. it captures the **beyond-two-body** correlation energy (the `EL − DMC` gap). With a one-body
DeepSet on a dilute gas that gap is (a) tiny and (b) unreachable by the architecture. So as set up
we are doing (slower) VMC to obtain what HNC already gives.

---

## What would make it worth it

Both must hold:

1. **A correlation-capable architecture** — a message-passing GNN (or backflow). A one-body net can
   never beat a two-body Jastrow; a message-passing net *can* add the 3-body / backflow structure
   that EL/HNC misses. (Keep the analytic Jastrow for the short-range cusp; let the GNN learn the
   residual.)
2. **A regime where `EL ≠ DMC`** — higher x and stiffer potentials toward the hard sphere (smaller
   R). At dilute x everything collapses onto Lee-Yang; the interesting gap opens at the dense / hard
   end — exactly where the paper's Fig. 12 shows EL, IPC and DMC separating.

---

## The decisive test

At a **higher-x, stiffer point** (e.g. SS5 or SS2 near the top of its box-feasible range), compare:

- our `--no-network --jastrow`  (bare Jastrow ≈ SR)
- our `--jastrow` net-on
- the paper's **EL / HNC**
- the paper's **DMC**

Outcomes:
- net-on ≈ Jastrow-only ≈ EL, **DMC below** → that gap is the target; the DeepSet provably can't
  close it → clean motivation for the GNN.
- net-on dips below EL toward DMC → the net is earning its keep; push that regime.

---

## Framing for a talk

> "The analytic two-body Jastrow already saturates the two-body physics and recovers HNC; the neural
> part only matters where beyond-two-body correlations do, which is the dense / hard-sphere end. We
> target that next with a message-passing ansatz, benchmarked against DMC."

Reproducing HNC = **validation** (the VMC + Jastrow + pipeline are correct). Beating HNC toward DMC
with a network that captures what the two-body Jastrow can't = **the contribution**. That means
GNN + the dense/stiff regime, not DeepSet + dilute.
