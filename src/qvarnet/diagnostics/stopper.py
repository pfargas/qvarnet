"""StationarityStopper (roadmap §3.5) — early-stop when the energy trace has converged.

A ``Callback``: after ``warmup`` epochs, every ``check_every`` epochs it runs both the
Heidelberger-Welch and Geweke referees on the last ``window`` energies. When both pass for
``patience`` consecutive checks, it confirms a plateau and stops (or fires a pluggable
``action`` — e.g. drop the LR, switch Adam→SR). Requiring two tests *with patience* is what
makes it robust: a single lucky window doesn't stop, a single noisy window doesn't reset.
"""

import numpy as np

from ..callbacks.base import Callback
from .stationarity import geweke_z, heidelberger_welch_t


class StationarityStopper(Callback):
    def __init__(
        self,
        warmup: int = 200,
        check_every: int = 50,
        window: int = 200,
        patience: int = 2,
        z_thr: float = 3.0,
        t_thr: float = 2.0,
        action=None,
        verbose: bool = False,
    ):
        self.warmup = warmup
        self.check_every = check_every
        self.window = window
        self.patience = patience
        self.z_thr = z_thr
        self.t_thr = t_thr
        self.action = action
        self.verbose = verbose
        self._energies: list[float] = []
        self._consecutive = 0
        self.stopped_at: int | None = None

    def on_step_end(self, step, state, metrics) -> bool:
        self._energies.append(float(metrics["energy"]))
        if step < self.warmup or step % self.check_every != 0:
            return False
        if len(self._energies) < self.window:
            return False

        w = np.asarray(self._energies[-self.window :])
        z = geweke_z(w)
        t = heidelberger_welch_t(w)
        passed = abs(z) < self.z_thr and abs(t) < self.t_thr
        self._consecutive = self._consecutive + 1 if passed else 0
        if self.verbose:
            print(
                f"[StationarityStopper] step={step} |z|={abs(z):.2f} |t|={abs(t):.2f} "
                f"pass={passed} streak={self._consecutive}"
            )

        if self._consecutive >= self.patience:
            self.stopped_at = step
            if self.action is not None:
                self.action(state)
            return True
        return False
