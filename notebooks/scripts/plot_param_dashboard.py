#!/usr/bin/env python
# To run: conda activate jax && python plot_param_dashboard.py <stats_file>
# If conda is not initialised: eval "$(conda shell.bash hook)" && conda activate jax

"""
Dashboard for visualising parameter/gradient evolution during VMC training.

Quick usage
-----------
# 1. After training, compute and save stats:
from plot_param_dashboard import compute_stats, save_stats
stats = compute_stats(state_history, target='params')   # or target='grads'
save_stats(stats, 'param_stats.json')

# 2. Plot from saved file (CLI):
python plot_param_dashboard.py param_stats.json --mode mean_std

# 3. Plot inside a notebook/script:
from plot_param_dashboard import load_stats, plot_dashboard
stats = load_stats('param_stats.json')
plot_dashboard(stats, mode='norm', title='Gradient norms')
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_pytree(pytree):
    """Return {dot-separated-path: numpy_array} for every leaf in *pytree*."""
    result = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(pytree):
        key = ".".join(p.key if hasattr(p, "key") else str(p) for p in path)
        result[key] = np.asarray(leaf)
    return result


def _get_target(state, target):
    if target == "params":
        return state.params
    if target == "grads":
        return state.grads
    raise ValueError(f"target must be 'params' or 'grads', got {target!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_stats(state_history, target="params", store_all=False):
    """Compute per-leaf statistics from a list of VMCState objects.

    Parameters
    ----------
    state_history : list[VMCState]
        The list returned by ``train()``.
    target : {'params', 'grads'}
    store_all : bool
        If True, also store the full per-parameter trajectories under the key
        ``'values'`` (shape ``(n_epochs, n_params_in_leaf)``).  Required for
        ``plot_dashboard(..., mode='all')``.  Off by default because it can be
        large for wide layers.

    Returns
    -------
    stats : dict
        ``{leaf_path: {'mean': ndarray, 'std': ndarray, 'norm': ndarray}}``
        plus a ``'_meta'`` key with bookkeeping info.
        Each 1-D array has shape ``(n_epochs,)``.
        When *store_all* is True each leaf also has ``'values'`` of shape
        ``(n_epochs, n_params)``.
    """
    if not state_history:
        raise ValueError("state_history is empty")

    first_flat = _flatten_pytree(_get_target(state_history[0], target))
    paths = list(first_flat.keys())

    accum = {p: {"mean": [], "std": [], "norm": [], "values": []} for p in paths}

    for state in state_history:
        flat = _flatten_pytree(_get_target(state, target))
        for p in paths:
            arr = flat[p].ravel().astype(np.float64)
            accum[p]["mean"].append(float(arr.mean()))
            accum[p]["std"].append(float(arr.std()))
            accum[p]["norm"].append(float(np.sqrt((arr ** 2).sum())))
            if store_all:
                accum[p]["values"].append(arr.copy())

    stats = {}
    for p, d in accum.items():
        stats[p] = {
            "mean": np.array(d["mean"]),
            "std":  np.array(d["std"]),
            "norm": np.array(d["norm"]),
        }
        if store_all:
            stats[p]["values"] = np.stack(d["values"])  # (n_epochs, n_params)

    stats["_meta"] = {
        "target": target,
        "n_epochs": len(state_history),
        "paths": paths,
        "store_all": store_all,
    }
    return stats


def save_stats(stats, path):
    """Serialise *stats* to a plain ASCII JSON file (gzip/bzip2-friendly).

    Parameters
    ----------
    stats : dict
        Output of :func:`compute_stats`.
    path : str or Path
    """
    serialisable = {}
    for key, val in stats.items():
        if key == "_meta":
            serialisable[key] = val
        else:
            serialisable[key] = {k: v.tolist() for k, v in val.items()}

    path = Path(path)
    with path.open("w") as f:
        json.dump(serialisable, f, separators=(",", ":"))

    size_kb = path.stat().st_size / 1024
    print(f"Saved {len(stats) - 1} leaf entries → {path}  ({size_kb:.1f} KB)")
    print("Tip: compress with  gzip -k <file>  or  xz -k <file>")


def load_stats(path):
    """Load stats previously saved by :func:`save_stats`.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    stats : dict  (same structure as :func:`compute_stats` output)
    """
    with Path(path).open("r") as f:
        raw = json.load(f)

    stats = {}
    for key, val in raw.items():
        if key == "_meta":
            stats[key] = val
        else:
            stats[key] = {k: np.array(v) for k, v in val.items()}
    return stats


def plot_dashboard(
    stats,
    mode="mean_std",
    title=None,
    save_path=None,
    ncols=4,
    panel_size=(4.0, 2.8),
    log_scale=False,
    xrange=None,
    yrange=None,
):
    """Plot a grid dashboard of per-leaf statistics over training.

    Parameters
    ----------
    stats : dict
        Output of :func:`compute_stats` or :func:`load_stats`.
    mode : {'mean_std', 'norm'}
        * ``'mean_std'`` — line for the mean ± shaded std band.
        * ``'norm'``     — line for sqrt(∑ xᵢ²) (L2 norm of the leaf).
    title : str, optional
        Figure suptitle. Defaults to "<target> <mode> dashboard".
    save_path : str or Path, optional
        If given, save the figure here (PNG/PDF/…).
    ncols : int
        Maximum number of columns in the grid.
    panel_size : (float, float)
        (width, height) in inches per subplot panel.
    log_scale : bool
        Use a log scale on the y-axis (useful for norms).
    """
    if mode not in ("mean_std", "norm", "all"):
        raise ValueError(f"mode must be 'mean_std' or 'norm', got {mode!r}")

    meta = stats.get("_meta", {})
    paths = meta.get("paths", [k for k in stats if k != "_meta"])
    target = meta.get("target", "unknown")
    n_epochs = meta.get("n_epochs", len(stats[paths[0]]["mean"]) if paths else 0)
    epochs = np.arange(n_epochs)

    n = len(paths)
    if n == 0:
        raise ValueError("No leaf paths found in stats")

    ncols = min(n, ncols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )

    for i, p in enumerate(paths):
        ax = axes[i // ncols][i % ncols]
        d = stats[p]

        if mode == "mean_std":
            mu, sigma = d["mean"], d["std"]
            ax.plot(epochs, mu, lw=0.9, label="mean")
            ax.fill_between(epochs, mu - sigma, mu + sigma, alpha=0.25, label="±std")
        elif mode == "norm":  # norm
            ax.plot(epochs, d["norm"], lw=0.9, color="C1", label="‖·‖₂")
        elif mode == "all":
            if "values" not in d:
                raise ValueError(
                    f"mode='all' requires store_all=True in compute_stats() "
                    f"(leaf '{p}' has no 'values' array)."
                )
            vals = d["values"]           # (n_epochs, n_params)
            n_params = vals.shape[1]
            # transparency scales down so overlapping lines stay readable
            alpha = float(np.clip(1.0 / n_params ** 0.5, 0.08, 0.9))
            for j in range(n_params):
                label = f"{n_params} params" if j == 0 else None
                ax.plot(epochs, vals[:, j], lw=0.6, alpha=alpha, color="C0", label=label)

        if log_scale:
            ax.set_yscale("log")

        short = p.removeprefix("params.")
        n_params_suffix = f" ({d['values'].shape[1]})" if mode == "all" and "values" in d else ""
        ax.set_title(short + n_params_suffix, fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        if xrange is not None:
            ax.set_xlim(xrange)
        if yrange is not None:
            ax.set_ylim(yrange)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6, loc="best")
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    suptitle = title or f"{target} · {mode} dashboard"
    fig.suptitle(suptitle, fontsize=11, y=1.005)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot param/grad dashboard from a stats JSON file."
    )
    parser.add_argument("stats_file", help="Path to JSON stats file (from save_stats())")
    parser.add_argument(
        "--mode",
        choices=["mean_std", "norm", "all"],
        default="mean_std",
        help="Plotting mode (default: mean_std)",
    )
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument("--save", default=None, help="Path to save the figure")
    parser.add_argument(
        "--log", action="store_true", help="Use log scale on y-axis"
    )
    parser.add_argument(
        "--ncols", type=int, default=4, help="Max columns in the grid (default: 4)"
    )
    args = parser.parse_args()

    stats = load_stats(args.stats_file)
    meta = stats.get("_meta", {})
    print(
        f"Loaded: target={meta.get('target')}, "
        f"epochs={meta.get('n_epochs')}, "
        f"leaves={len(meta.get('paths', []))}"
    )
    plot_dashboard(
        stats,
        mode=args.mode,
        title=args.title,
        save_path=args.save,
        ncols=args.ncols,
        log_scale=args.log,
    )
