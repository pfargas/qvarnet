"""Per-epoch metrics history (roadmap step 2).

Replaces the old ``state_history`` that stored a full ``VMCState`` (params + grads +
optimizer state) every epoch — a memory bomb on long runs. Here we keep **only**
scalars and small per-chain vectors on the host.

Iterating a ``MetricsHistory`` yields lightweight ``EpochRecord`` objects with
attribute access, so existing code like ``[s.energy for s in result.history]`` and
``result.history[-1].std`` keeps working. ``get(field)`` stacks a field across epochs
into an array for analysis/plots.

Canonical per-epoch fields (an objective may emit more — the store is schema-free):

    step           epoch index
    energy         ⟨E⟩
    std            σ of E_loc over the batch
    error_of_mean  σ_E / sqrt(M)  (naive; upgraded to σ_E·sqrt(τ_int/M) in step 4)
    E_chain        per-chain mean E_loc, shape (n_chains,)  → split-R̂ / Geweke
    acceptance_rate per-chain MH acceptance, shape (n_chains,)
    step_size      MH proposal step size
    cm_mean        centre-of-mass mean (diagnostic)
    cm_std         centre-of-mass std (diagnostic)
    wall_time      seconds for the epoch (equal-time comparisons, roadmap §8.4)
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
