"""Exact diagonalisation reference for small discrete systems (roadmap step 3).

Dense Lanczos/eigh on the full 2^N Hilbert space — only for N ≲ 16. Gives the known
ground-state energy the discrete-VMC testbed validates against.

Spin/index convention (shared with the VMC path, ``vmc/full_sum.py``):
    basis state index ``a`` ∈ [0, 2^N);  spin at site i is  s_i = 1 - 2·bit_i(a) ∈ {+1, -1};
    flipping site i maps  a ↦ a ^ (1 << i).
"""

import numpy as np


def all_spin_configs(n: int) -> np.ndarray:
    """All 2^n spin configurations as a (2^n, n) array of ±1, row a = basis state a."""
    a = np.arange(2**n)
    bits = (a[:, None] >> np.arange(n)[None, :]) & 1
    return (1 - 2 * bits).astype(np.float64)


def tfim_dense(n: int, J: float = 1.0, h: float = 1.0, pbc: bool = True) -> np.ndarray:
    """Dense transverse-field Ising Hamiltonian H = -J Σ σ^z_i σ^z_{i+1} - h Σ σ^x_i.

    Returns the (2^n, 2^n) matrix in the convention above.
    """
    dim = 2**n
    configs = all_spin_configs(n)  # (dim, n)
    H = np.zeros((dim, dim))

    # Diagonal: classical Ising bonds.
    if pbc:
        diag = -J * np.sum(configs * np.roll(configs, -1, axis=1), axis=1)
    else:
        diag = -J * np.sum(configs[:, :-1] * configs[:, 1:], axis=1)
    H[np.arange(dim), np.arange(dim)] = diag

    # Off-diagonal: transverse field flips one site, matrix element -h.
    a = np.arange(dim)
    for i in range(n):
        b = a ^ (1 << i)
        H[a, b] += -h
    return H


def ground_state_energy(H: np.ndarray) -> float:
    """Lowest eigenvalue of a dense Hermitian H."""
    return float(np.linalg.eigvalsh(H)[0])


def tfim_ground_state_energy(n: int, J: float = 1.0, h: float = 1.0, pbc: bool = True) -> float:
    """Convenience: exact TFIM ground-state energy by dense diagonalisation."""
    return ground_state_energy(tfim_dense(n, J=J, h=h, pbc=pbc))
