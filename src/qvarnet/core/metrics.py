"""Per-epoch scalar history of a run.

Columns are appended per epoch and retrieved as arrays with ``.get(name)``. Holds
scalars plus the per-chain energies ``E_chain`` -- deliberately no parameters or
gradients, so nothing here pins device memory or grows with model size.

``E_chain`` is what makes split-R-hat and any post-hoc correlated error estimate
possible; see docs/adr/0001-correlated-error-estimates.md.
"""

import numpy as np


class EpochRecord:
    """Read-only attribute/dict view over one epoch's metrics dict."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as exc:  # pragma: no cover - mirrors AttributeError contract
            raise AttributeError(name) from exc

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def as_dict(self) -> dict:
        return dict(self._data)

    def __repr__(self):
        e = self._data.get("energy")
        s = self._data.get("step")
        return f"EpochRecord(step={s}, energy={e})"


class MetricsHistory:
    """Struct-of-arrays per-epoch metrics; no params/grads/optimizer state."""

    def __init__(self):
        self._records: list[dict] = []

    def append(self, metrics: dict) -> None:
        """Store one epoch's metrics. Caller is responsible for passing host
        (``jax.device_get``-ed) values so nothing keeps device memory alive."""
        self._records.append(dict(metrics))

    def get(self, field: str) -> np.ndarray:
        """Stack ``field`` across all epochs, shape ``(n_epochs, *field_shape)``."""
        return np.array([r[field] for r in self._records])

    def keys(self) -> list[str]:
        """Field names available to ``get()``.

        Which fields exist depends on the run -- the SR guard diagnostics
        (``trust_scale``, ``solve_ok``, ...) only appear when use_qgt is on.
        """
        return list(self._records[0]) if self._records else []

    @property
    def energy(self) -> np.ndarray:
        return self.get("energy")

    @property
    def std(self) -> np.ndarray:
        return self.get("std")

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return (EpochRecord(r) for r in self._records)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [EpochRecord(r) for r in self._records[idx]]
        return EpochRecord(self._records[idx])

    def __repr__(self):
        n = len(self._records)
        if n:
            return f"MetricsHistory(n_epochs={n}, last_energy={self._records[-1].get('energy')})"
        return "MetricsHistory(n_epochs=0)"
