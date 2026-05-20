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

    def best(self, n: int = 1, metric: list = ["energy"]):
        """Return the N VMCState objects sorted ascending by each metric (lowest = best).

        Args:
            n:      number of states to return per metric.
            metric: list of ranking criteria. Each element is either a string shortcut
                    or a callable (VMCState) -> float (lower = better).

                    Shortcuts:  "energy"  — lowest ⟨E⟩
                                "std"     — lowest σ_E

                    Custom examples:
                        metric=[lambda s: float(s.energy) + float(s.std)]
                        metric=["energy", lambda s: abs(float(s.acceptance_rate.mean()) - 0.5)]

        Returns:
            If metric has one element: list of VMCState sorted ascending (best first).
            If metric has multiple elements: dict mapping each metric element to such a list.
        """
        _builtins = {
            "energy": lambda s: float(s.energy),
            "std": lambda s: float(s.std),
            # TODO: add 2 more built-in metrics: E+alpha*sigma where alpha is tunable and V-Score (https://arxiv.org/abs/2302.04919v2)
        }
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
            result[m] = sorted(self.history, key=key_fn)[:n]
        return result if len(metric) > 1 else result[metric[0]]

    def __iter__(self):
        # Backward compat: allows  history, cm_mean, cm_std = result
        return iter((self.history, self.cm_mean, self.cm_std))

    def __repr__(self):
        n = len(self.history)
        if n:
            last_e = self.history[-1].energy
            return f"TrainResult(n_steps={n}, last_energy={float(last_e):.6f})"
        return "TrainResult(n_steps=0)"
