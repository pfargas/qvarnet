"""Early stopping by the three-referee convergence verdict (roadmap §3).

Long VMC runs usually converge well before their epoch budget. This callback runs the standard
``three_referee_verdict`` (stationarity + MC-error floor + split-R̂ chain mixing) on the history
so far, every ``check_every`` epochs after a ``min_epochs`` warm-up, and stops training once the
verdict has *passed* ``patience`` checks in a row — optionally also requiring the relative MC error
``err/|E|`` to drop below ``target_rel_err`` so it can't stop while still noisy.

It keeps its own light copy of the per-epoch metrics (the loop doesn't hand callbacks the engine's
``MetricsHistory``); these are scalars plus the per-chain ``E_chain`` the R̂ referee needs, so the
overhead is negligible next to the model.

    train(..., callbacks=[EarlyStopCallback(min_epochs=200, patience=2)])

``n_epochs`` then acts as a *ceiling*: the run ends at the verdict, or at ``n_epochs`` if it never
converges. ``stopped_at`` records the stopping epoch (``None`` if it ran to the cap).
"""

import numpy as np

from ..diagnostics import three_referee_verdict
from ..vmc.metrics_history import MetricsHistory
from .base import Callback


class EarlyStopCallback(Callback):
    def __init__(
        self,
        check_every: int = 50,
        min_epochs: int = 200,
        patience: int = 2,
        target_rel_err: float | None = None,
        plateau_rel: float | None = None,
        verdict_kwargs: dict | None = None,
    ):
        """Stop when a run looks converged for ``patience`` checks in a row.

        A check counts as "converged" if the three-referee verdict passes (optionally also
        ``err/|E| < target_rel_err``), **or** — when ``plateau_rel`` is set — if the tail-mean
        energy improved by less than ``plateau_rel`` (relative) since the previous check. The
        plateau trigger catches the common NN-VMC case where the energy is still drifting down too
        slowly to matter but the strict stationarity referee would keep waiting. ``plateau_rel``
        defaults off (verdict-only = never stops while genuinely improving).
        """
        if check_every < 1:
            raise ValueError("check_every must be >= 1")
        self.check_every = check_every
        self.min_epochs = min_epochs
        self.patience = patience
        self.target_rel_err = target_rel_err
        self.plateau_rel = plateau_rel
        self.verdict_kwargs = dict(verdict_kwargs or {})
        self._hist = MetricsHistory()
        self._consec = 0
        self.stopped_at: int | None = None
        self.last_verdict: dict | None = None
        self.stop_reason: str | None = None

    def on_step_end(self, step: int, state, metrics: dict) -> bool:
        self._hist.append(metrics)
        if len(self._hist) < self.min_epochs or step % self.check_every != 0:
            return False

        v = three_referee_verdict(self._hist, **self.verdict_kwargs)
        self.last_verdict = v

        verdict_ok = bool(v["passed"])
        if verdict_ok and self.target_rel_err is not None:
            rel = float(v["tail_error_of_mean"]) / (abs(float(v["tail_energy"])) + 1e-30)
            verdict_ok = rel < self.target_rel_err

        # plateau: compare the last `check_every` energies to the block just before it. A small
        # (or negative) relative improvement means the optimiser has effectively stopped lowering
        # the energy — stop even if the strict stationarity referee would keep waiting.
        plateau_ok = False
        if self.plateau_rel is not None:
            e = self._hist.get("energy")
            w = self.check_every
            if len(e) >= 2 * w:
                prev = float(np.mean(e[-2 * w : -w]))
                curr = float(np.mean(e[-w:]))
                improvement = (prev - curr) / (abs(prev) + 1e-30)
                plateau_ok = improvement < self.plateau_rel

        reason = "verdict" if verdict_ok else ("plateau" if plateau_ok else None)
        self._consec = self._consec + 1 if reason else 0
        if self._consec >= self.patience:
            self.stopped_at = step
            self.stop_reason = reason
            return True
        return False
