"""Plot an E/N(x) curve against the paper's benchmarks (all in paper units, a = 1)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from dilute_gas import first_order_energy_upper_bound, lee_yang_energy_per_particle


def plot_curve(curve: dict, potential, *, scaled: bool = True, ax=None, save: str | None = None):
    """Overlay VMC E/N(x) on lower bound (4πx), Lee-Yang (Eq.1), and Eq.31 upper bound.

    ``scaled=True`` plots E/N / 4πx (the paper's Fig. 1 / Fig. 12 axis), where the rigorous
    lower bound is the horizontal line 1 and shape/correlation effects are the deviation above it.
    """
    x = curve["x"]
    if len(x) == 0:
        raise ValueError("empty curve — no done/passed runs in the DB yet")

    xs = np.geomspace(x.min(), x.max(), 200)
    lb = 4 * math.pi * xs
    ly = np.array([lee_yang_energy_per_particle(xx) for xx in xs])
    ub = np.array([first_order_energy_upper_bound(xx, potential.V0_paper, potential.R) for xx in xs])

    norm = (4 * math.pi * x) if scaled else 1.0
    norm_s = (4 * math.pi * xs) if scaled else 1.0

    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.plot(xs, lb / norm_s, "k:", lw=1, label="lower bound  4πx")
    ax.plot(xs, ly / norm_s, "k--", lw=1, label="Lee-Yang (Eq.1)")
    ax.plot(xs, ub / norm_s, color="tab:red", lw=1, label="upper bound (Eq.31)")
    ax.errorbar(x, curve["e_per_n"] / norm, yerr=curve["err"] / norm, fmt="o",
                color="tab:blue", capsize=3, label=f"VMC ({potential.label})")
    ax.set_xscale("log")
    ax.set_xlabel("x = ρa³")
    ax.set_ylabel("E/N / 4πx" if scaled else "E/N  [ℏ²/2ma²]")
    ax.set_title(f"{potential.label}: energy per particle vs gas parameter")
    ax.legend(fontsize=8)
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=140)
        print(f"saved {save}")
    return ax
