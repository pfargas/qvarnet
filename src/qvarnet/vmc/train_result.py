"""TrainResult — container returned by train()."""


def e_plus_sigma_metric(alpha: float = 1.0):
    """Model-selection key Ē + α·σ_E (roadmap §2.2) — usable in ``TrainResult.best``."""
    return lambda s: float(s.energy) + alpha * float(s.std)


def v_score_metric(n_particles: int, e_inf: float = 0.0):
    """Model-selection key = V-score N·Var(E_loc)/(Ē−E_∞)² (arXiv:2302.04919); lower is better."""
    return (
        lambda s: n_particles
        * float(s.std) ** 2
        / ((float(s.energy) - e_inf) ** 2 + 1e-12)
    )


class TrainResult:
    """Result of a VMC training run.

    Attributes:
        history: a ``MetricsHistory`` — iterate it for per-epoch records
                 (``[s.energy for s in result.history]`` still works), or use
                 ``history.get(field)`` for stacked arrays. It holds scalars and
                 small per-chain vectors only — **no params** (that was the old
                 per-epoch memory bomb).
        cm_mean / cm_std: per-epoch centre-of-mass diagnostics (lists, derived
                 from history for backward compatibility).
        final_positions / final_step_size: last walker positions ``(n_chains, dof)``
                 and last (possibly adapted) MH step size. Feed them into the next run
                 (``ChainInitAndWarmupConfig(init_positions=result.final_positions)``,
                 ``sampler_params={"step_size": result.final_step_size, ...}``) so a
                 warm-started rerun resumes sampling where this one left off instead of
                 re-thermalising at the default step size.
    """

    def __init__(
        self,
        history,
        final_params=None,
        snapshots=None,
        final_positions=None,
        final_step_size=None,
    ):
        self.history = history
        self.final_params = (
            final_params  # host pytree of the last-epoch params, or None
        )
        # each snapshot: {"step", "metric", "params"} — from the SnapshotCallback policy
        self.snapshots = list(snapshots) if snapshots else []
        self.final_positions = final_positions  # host (n_chains, dof) array, or None
        self.final_step_size = final_step_size  # float, or None

    def best_params(self):
        """Parameters of the single best retained snapshot (lowest selection metric).

        Falls back to ``final_params`` when no snapshots were kept (e.g. policy "none").
        """
        if not self.snapshots:
            return self.final_params
        return min(self.snapshots, key=lambda s: s["metric"])["params"]

    def best_k(self, n: int = None):
        """Retained snapshots, lowest selection metric first (best first).

        Each item is a dict ``{"step", "metric", "params"}`` — so the **epoch** that produced a
        given parameter set is ``snapshot["step"]``. ``n`` limits the count; ``None`` returns all.
        """
        ranked = sorted(self.snapshots, key=lambda s: s["metric"])
        return ranked if n is None else ranked[:n]

    def best_k_params(self, n: int = None):
        """List of retained snapshot params, best first. For the epoch of each, use ``best_k``."""
        return [s["params"] for s in self.best_k(n)]

    def best_steps(self, n: int = None):
        """The epochs (``step``) the best-k params were taken from, best first."""
        return [s["step"] for s in self.best_k(n)]

    @property
    def cm_mean(self):
        return [s.cm_mean for s in self.history]

    @property
    def cm_std(self):
        return [s.cm_std for s in self.history]

    def best(self, n: int = 1, metric: list = ["energy"]):
        """Return the N best per-epoch records sorted ascending by each metric.

        Note: records are lightweight (no params). Retrieving the *parameters* of the
        best epoch needs the snapshot policy (none/every_n/all/best_k) landing in
        roadmap step 6; until then ``best()`` ranks metrics only.

        Args:
            n:      number of records to return per metric.
            metric: list of ranking criteria. Each element is either a string
                    shortcut ("energy", "std") or a callable (record) -> float
                    (lower = better).

        Returns:
            If metric has one element: list of records sorted ascending (best first).
            If multiple: dict mapping each metric element to such a list.
        """
        _builtins = {
            "energy": lambda s: float(s.energy),
            "std": lambda s: float(s.std),
            # TODO (step 6): e_plus_alpha_sigma and V-score (arXiv:2302.04919).
        }
        if isinstance(metric, str) or callable(metric):
            metric = [metric]
        records = list(self.history)
        result = {}
        for m in metric:
            if callable(m):
                key_fn = m
            else:
                if m not in _builtins:
                    raise ValueError(
                        f"metric must be one of {list(_builtins)} or a callable, got {m!r}"
                    )
                key_fn = _builtins[m]
            result[m] = sorted(records, key=key_fn)[:n]
        return result if len(metric) > 1 else result[metric[0]]

    def summary(self, print_report: bool = True) -> dict:
        """End-of-run statistics: what you want to see after training in a notebook.

        Reports the best retained model (by the snapshot ``select`` metric), the best and
        final epochs' energy ± error and σ_E, acceptance, total wall time, epochs ran and
        parameter count. Printed automatically at the end of ``train()`` unless
        ``TrainingConfig.print_summary=False``. Returns the same values as a dict.

        Deliberately **no** mean over the training history: those samples come from a
        moving distribution, so their average is not a physical estimator — measure with
        ``qvarnet.evaluate`` / ``evaluate_result`` (frozen params, block-averaged errors).
        """
        import numpy as np

        h = self.history
        n = len(h)
        if n == 0:
            if print_report:
                print("TrainResult.summary: empty history (no epochs ran).")
            return {}

        energy = h.get("energy")
        err = h.get("error_of_mean")
        std = h.get("std")
        tail = slice(n // 2, None)
        acceptance = float(np.mean(np.asarray(h.get("acceptance_rate"))[tail]))
        wall = float(np.sum(h.get("wall_time")))
        i_best = int(np.argmin(std))

        n_params = None
        if self.final_params is not None:
            from jax.flatten_util import ravel_pytree

            n_params = int(ravel_pytree(self.final_params)[0].size)

        out = {
            "epochs_ran": n,
            "wall_time_s": wall,
            "n_parameters": n_params,
            "final": {
                "energy": float(energy[-1]),
                "error_of_mean": float(err[-1]),
                "std": float(std[-1]),
            },
            "best_epoch": {
                "step": int(h.get("step")[i_best]),
                "energy": float(energy[i_best]),
                "error_of_mean": float(err[i_best]),
                "std": float(std[i_best]),
            },
            "acceptance_tail": acceptance,
            "n_snapshots_kept": len(self.snapshots),
        }
        if self.snapshots:
            s = min(self.snapshots, key=lambda s: s["metric"])
            out["best_snapshot"] = {
                "step": int(s["step"]),
                "metric": float(s["metric"]),
            }

        if print_report:
            mins, secs = divmod(wall, 60.0)
            lines = [
                "── training summary " + "─" * 40,
                f"epochs ran       : {n}   ({int(mins)}m {secs:04.1f}s wall"
                + (f", {n_params} parameters)" if n_params is not None else ")"),
                f"final epoch      : E = {out['final']['energy']:.6f} "
                f"± {out['final']['error_of_mean']:.2e}   σ_E = {out['final']['std']:.4f}",
                f"best epoch ({out['best_epoch']['step']:>5}) : E = {out['best_epoch']['energy']:.6f} "
                f"± {out['best_epoch']['error_of_mean']:.2e}   σ_E = {out['best_epoch']['std']:.4f}",
                f"acceptance (tail): {acceptance:.3f}",
            ]
            if "best_snapshot" in out:
                lines.append(
                    f"best snapshot    : epoch {out['best_snapshot']['step']} "
                    f"(select metric = {out['best_snapshot']['metric']:.6g}; "
                    f"{len(self.snapshots)} kept — result.best_params() to load)"
                )
            lines.append("─" * 60)
            print("\n".join(lines))
        return out

    def diagnose(self, print_report: bool = True, **kwargs) -> dict:
        """Run the three-referee convergence verdict on the history (roadmap §3).

        Returns the verdict dict (stationary? / at MC floor? / chains mixed?), and prints the
        formatted report by default — the standard end-of-run artifact. Extra kwargs go to
        ``diagnostics.three_referee_verdict`` (tail_frac, rhat_threshold, z_thr, t_thr, ...).
        """
        from ..diagnostics import format_verdict, three_referee_verdict

        verdict = three_referee_verdict(self.history, **kwargs)
        if print_report:
            print(format_verdict(verdict))
        return verdict

    def __iter__(self):
        # Backward compat: allows  history, cm_mean, cm_std = result
        return iter((self.history, self.cm_mean, self.cm_std))

    def __repr__(self):
        n = len(self.history)
        if n:
            last_e = self.history[-1].energy
            return f"TrainResult(n_steps={n}, last_energy={float(last_e):.6f})"
        return "TrainResult(n_steps=0)"
