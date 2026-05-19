"""TrainResult — container returned by train()."""


class TrainResult:
    """Result of a VMC training run.

    Attributes:
        history: list of VMCState, one per epoch (includes full params).
        cm_mean: list of float — per-epoch mean centre-of-mass.
        cm_std:  list of float — per-epoch std of centre-of-mass.
    """

    def __init__(self, history, cm_mean, cm_std):
        self.history = history
        self.cm_mean = cm_mean
        self.cm_std = cm_std

    def best(self, n: int = 1, metric="energy"):
        """Return the N VMCState objects sorted ascending by metric (lowest = best).

        Args:
            n:      number of states to return.
            metric: a string shortcut OR any callable (VMCState) -> float.

                    Shortcuts:  "energy"  — lowest ⟨E⟩
                                "std"     — lowest σ_E

                    Custom examples:
                        metric=lambda s: float(s.energy) / float(s.std)
                        metric=lambda s: abs(float(s.acceptance_rate.mean()) - 0.5)

        Returns:
            List of VMCState sorted ascending by metric (best first).
        """
        if callable(metric):
            key_fn = metric
        else:
            _builtins = {
                "energy": lambda s: float(s.energy),
                "std":    lambda s: float(s.std),
            }
            if metric not in _builtins:
                raise ValueError(
                    f"metric must be one of {list(_builtins)} or a callable, got {metric!r}"
                )
            key_fn = _builtins[metric]
        return sorted(self.history, key=key_fn)[:n]

    def __iter__(self):
        # Backward compat: allows  history, cm_mean, cm_std = result
        return iter((self.history, self.cm_mean, self.cm_std))

    def __repr__(self):
        n = len(self.history)
        if n:
            last_e = self.history[-1].energy
            return f"TrainResult(n_steps={n}, last_energy={float(last_e):.6f})"
        return "TrainResult(n_steps=0)"
