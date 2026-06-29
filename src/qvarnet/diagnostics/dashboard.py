"""Training dashboard (roadmap §8.3, step 7) — the standard end-of-run PNG.

One figure from a ``TrainResult``: energy vs exact/MC-floor, the optimisation diagnostics
(grad norm, θ-step), sampler health (acceptance, per-chain energy spread), and the
three-referee verdict as text. Built via the object-oriented ``Figure`` API so it is
headless-safe without touching the process-wide matplotlib backend.
"""

import numpy as np
from matplotlib.figure import Figure

from .verdict import format_verdict, three_referee_verdict


def plot_dashboard(result, exact_energy=None, save_path=None, title="qvarnet run"):
    """Build the diagnostics dashboard for a ``TrainResult``. Returns the matplotlib Figure."""
    h = result.history
    ep = np.arange(len(h))
    energy = h.get("energy")
    err = h.get("error_of_mean")

    fig = Figure(figsize=(16, 8))
    axes = fig.subplots(2, 3)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(ep, energy, lw=2, label="⟨E⟩")
    ax.fill_between(ep, energy - err, energy + err, alpha=0.25, label="± SEM")
    ax.axhline(float(np.min(energy)), ls=":", color="gray", label="best")
    if exact_energy is not None:
        ax.axhline(exact_energy, ls="--", color="k", label="exact")
    ax.set_title("energy")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.semilogy(ep, np.abs(h.get("grad_norm")) + 1e-30)
    ax.set_title("gradient norm")
    ax.set_xlabel("epoch")

    ax = axes[0, 2]
    ax.semilogy(ep, np.abs(h.get("theta_ratio")) + 1e-30)
    ax.axhspan(1e-3, 1e-2, color="green", alpha=0.1)  # healthy band
    ax.set_title("θ-step / θ (healthy ~1e-3–1e-2)")
    ax.set_xlabel("epoch")

    ax = axes[1, 0]
    acc = h.get("acceptance_rate")
    ax.plot(ep, acc.mean(axis=1) if acc.ndim == 2 else acc)
    ax.set_title("acceptance rate")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    E_chain = h.get("E_chain")
    if E_chain.ndim == 2:
        ax.plot(ep, E_chain.std(axis=1))
    ax.set_title("per-chain energy spread")
    ax.set_xlabel("epoch")

    ax = axes[1, 2]
    ax.axis("off")
    try:
        verdict = three_referee_verdict(h)
        ax.text(
            0.0,
            0.5,
            format_verdict(verdict),
            family="monospace",
            fontsize=9,
            va="center",
            transform=ax.transAxes,
        )
    except Exception as exc:  # pragma: no cover - dashboard must never crash a run
        ax.text(0.0, 0.5, f"verdict unavailable:\n{exc}", fontsize=9, va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
