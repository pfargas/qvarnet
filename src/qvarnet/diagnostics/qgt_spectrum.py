"""QGT spectrum diagnostics (PARAMETER_EFFICIENCY.md §4) — host/numpy.

The eigenvalues {λ_k} of the quantum geometric tensor S say how many parameter directions are
actually doing work above the MC noise floor:

    D_eff(ε) = #{k : λ_k > ε·λ_max}        effective number of working parameters
    D_part   = (Σ λ_k)² / Σ λ_k²           participation ratio.

Read like a gauge (see PARAMETER_EFFICIENCY.md §4): D_eff ≪ P with a collapsing spectrum ⇒
oversized net (shrink/prune); D_eff ≈ P with energy stuck above the MC floor ⇒ capacity-limited
(grow); D_eff ≈ P at the MC floor ⇒ right-sized.
"""

import numpy as np
from jax.flatten_util import ravel_pytree

from ..geometry.qgt import compute_qgt


def qgt_eigenvalues(params, batch, model_apply, regularization: float = 0.0) -> np.ndarray:
    """Eigenvalues of the (regularised) QGT S = ⟨O O⟩_c, ascending."""
    flat, unravel = ravel_pytree(params)
    S, _ = compute_qgt(flat, batch, lambda p, x: model_apply(unravel(p), x), regularization)
    return np.linalg.eigvalsh(np.asarray(S))


def d_eff(eigenvalues, eps: float = 1e-3) -> int:
    """Effective dimension: number of eigenvalues above eps·λ_max."""
    e = np.asarray(eigenvalues)
    return int(np.sum(e > eps * e.max()))


def d_part(eigenvalues) -> float:
    """Participation ratio (Σλ)² / Σλ²."""
    e = np.asarray(eigenvalues)
    return float(e.sum() ** 2 / (np.sum(e**2) + 1e-12))
