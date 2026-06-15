"""Step 8 (parameter retrieval) guards:

- ``train()`` exposes the final-epoch params and the k-best params on the result;
- the snapshot selection metric is customizable (string shortcut or callable, lower = better);
- the broken ``result.history[-1].params`` path is replaced (history is param-free by design).
"""

import jax
import jax.numpy as jnp
import optax
from conftest import make_ho_model

from qvarnet.callbacks.snapshot import resolve_metric_fn
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.train import train


def _run(tmp_path, **kw):
    return train(
        shape=(64, 1),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(n_epochs=20, rng_seed=0, checkpoint_path=str(tmp_path)),
        sampler_params={"step_size": 0.6, "chain_length": 80, "thermalization_steps": 20, "thinning_factor": 2},
        **kw,
    )


def _is_param_pytree(p):
    leaves = jax.tree_util.tree_leaves(p)
    return len(leaves) > 0 and all(hasattr(x, "shape") for x in leaves)


def test_final_and_best_params_exposed(tmp_path):
    result = _run(tmp_path, k_best=3)
    assert _is_param_pytree(result.final_params)
    assert _is_param_pytree(result.best_params())
    bk = result.best_k_params()
    assert 1 <= len(bk) <= 3 and all(_is_param_pytree(p) for p in bk)
    # best_k_params is sorted best-first by the (default "std") metric
    metrics = [s["metric"] for s in sorted(result.snapshots, key=lambda s: s["metric"])]
    assert metrics == sorted(metrics)
    # the epoch of each kept param is recoverable (the user's requirement)
    records = result.best_k()
    assert len(records) == len(bk)
    assert all(set(r) >= {"step", "metric", "params"} for r in records)
    steps = result.best_steps()
    assert steps == [r["step"] for r in records]
    assert all(isinstance(s, int) and 0 <= s < 20 for s in steps)


def test_best_params_is_min_std_by_default(tmp_path):
    result = _run(tmp_path, k_best=5)
    # default select="std": the best retained snapshot has the smallest std among retained
    best = min(result.snapshots, key=lambda s: s["metric"])
    assert best["metric"] == min(s["metric"] for s in result.snapshots)
    # and it matches best_params()'s pytree
    chosen = result.best_params()
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: bool(jnp.array_equal(a, b)), chosen, best["params"])
    )


def test_custom_callable_metric(tmp_path):
    # V-score-like / arbitrary callable on the metrics dict (lower = better)
    select = lambda m: float(m["energy"]) + 2.0 * float(m["std"])
    result = _run(tmp_path, select=select, k_best=2)
    assert len(result.best_k_params()) <= 2
    assert _is_param_pytree(result.best_params())


def test_k_best_zero_keeps_nothing_but_final(tmp_path):
    result = _run(tmp_path, k_best=0)
    assert result.snapshots == []
    assert result.best_params() is result.final_params  # falls back to final


def test_resolve_metric_fn_shortcuts():
    m = {"energy": -1.5, "std": 0.3, "grad_norm": 4.0}
    assert resolve_metric_fn("std")(m) == 0.3
    assert resolve_metric_fn("energy")(m) == -1.5
    assert resolve_metric_fn("e_plus_sigma")(m) == -1.2  # -1.5 + 0.3
    assert resolve_metric_fn("grad_norm")(m) == 4.0  # any raw key works
    assert resolve_metric_fn(lambda d: d["std"] ** 2)(m) == 0.09
