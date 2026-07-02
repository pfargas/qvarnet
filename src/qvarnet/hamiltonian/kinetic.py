"""Kinetic energy for VMC — log-space models only.

  T = -1/2 Σ_k w_k ( ∂²_k log|ψ| + (∂_k log|ψ|)² ),      w_k = 1/m_k per dof

(w_k ≡ 1 for equal masses, the default). :func:`kinetic_log` dispatches between two
implementations of the same operator:

* **AD branch** — the gradient via ``vmap(grad)`` plus a pluggable Laplacian estimator
  from ``laplacian.py`` (``forward_ad`` / ``hutchinson`` / ``central_difference`` /
  ``full_hessian``). Per-dof mass weights are applied natively by each estimator.
* **folx branch** (``use_folx=True``) — the forward-Laplacian algorithm (LapNet): one
  forward pass yields value, gradient and Laplacian together, so the separate ``grad``
  pass and the O(dof) JVP loop both disappear. folx computes only *unweighted*
  Laplacians, so mass weights are realised by the exact √m coordinate rescaling
  y = √m ⊙ x (Δ_y of the rescaled function is the mass-weighted operator) — an internal
  detail; both branches expose identical semantics and are equivalence-tested.
"""

import jax
import jax.numpy as jnp

from .laplacian import laplacian_forward_ad


def kinetic_log(params, samples, model_apply, *, laplacian_fn=None, use_folx=False,
                sparsity_threshold=0, dof_weights=None, key=None):
    """T = -1/2 Σ_k w_k (∂²_k log|ψ| + (∂_k log|ψ|)²), dispatched on the configured method.

    samples:     (batch, dof)
    laplacian_fn: AD branch only — a function from ``laplacian.py`` (default forward_ad)
    use_folx:    select the forward-Laplacian branch (requires the ``folx`` package)
    sparsity_threshold: folx branch only — 0 = dense (see folx docs)
    dof_weights: (dof,) per-dof 1/m weights, or None for equal masses
    key:         PRNGKey — only used by laplacian_hutchinson; ignored otherwise
    returns:     (batch,)
    """
    def log_psi(x):
        return model_apply(params, x[None]).squeeze()  # x: (dof,) -> scalar

    if use_folx:
        return _kinetic_folx(log_psi, samples, sparsity_threshold, dof_weights)
    if laplacian_fn is None:
        laplacian_fn = laplacian_forward_ad
    return _kinetic_ad(log_psi, samples, laplacian_fn, dof_weights, key)


def _kinetic_ad(log_psi, samples, laplacian_fn, dof_weights, key):
    """Gradient via vmap(grad) + pluggable (optionally mass-weighted) Laplacian."""
    grad_log_psi = jax.vmap(jax.grad(log_psi))(samples)          # (batch, dof)
    lap = laplacian_fn(log_psi, samples, key, weights=dof_weights)  # (batch,)
    grad_sq = grad_log_psi**2 if dof_weights is None else dof_weights * grad_log_psi**2
    return -0.5 * (lap + jnp.sum(grad_sq, axis=-1))


def _kinetic_folx(log_psi, samples, sparsity_threshold, dof_weights):
    """One forward-Laplacian pass per sample; masses via exact √m rescaling.

    With g(y) = log_psi(√w ⊙ y) evaluated at y = x / √w (w = 1/m), the chain rule gives
        ∂²_{y,k} g = w_k ∂²_{x,k} log|ψ|,   |∇_y g|² = Σ_k w_k (∂_{x,k} log|ψ|)²
    i.e. the plain kinetic formula applied to g *is* the mass-weighted operator at x.
    """
    try:
        from folx import forward_laplacian
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "laplacian_method='folx' requires the folx package (pip install folx); "
            "other laplacian_method values work without it"
        ) from exc

    if dof_weights is None:
        fn, xs = log_psi, samples
    else:
        scale = jnp.sqrt(dof_weights)                 # a_k = √w_k
        fn = lambda y: log_psi(y * scale)             # noqa: E731  g(y) = f(√w ⊙ y)
        xs = samples / scale                          # y = x / √w

    fwd = forward_laplacian(fn, sparsity_threshold=sparsity_threshold)

    def kinetic_single(x):
        r = fwd(x)
        return -0.5 * (r.laplacian + jnp.sum(r.jacobian.dense_array**2))

    return jax.vmap(kinetic_single)(xs)
