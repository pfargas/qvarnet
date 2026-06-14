"""Single-host data-parallel sharding primitives (roadmap §10).

VMC is embarrassingly parallel in exactly one place: the chains/samples axis. The model is
small (replicated everywhere); the walker batch is what shards. The modern JAX way (not the
legacy ``pmap``): a 1-D device ``Mesh``, params replicated, walkers sharded over the ``chains``
axis with ``NamedSharding``, and ``jax.jit`` inserting the collective reductions automatically:

    Ē = (1/M) Σ_devices Σ_local E_loc(R_i)     # an automatic psum under jit-with-sharding

These are the building blocks honouring the §10 constraints (leading chains axis, no host
round-trips inside the epoch, per-chain quantities stay per-chain). Wiring them into ``train``
is the documented "~day's bump" — left until real multi-GPU hardware is available to validate.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def make_chain_mesh(n_devices: int | None = None) -> Mesh:
    """1-D mesh named ``chains`` over (a prefix of) the available devices."""
    devices = jax.devices()
    if n_devices is not None:
        devices = devices[:n_devices]
    return Mesh(np.array(devices), axis_names=("chains",))


def shard_over_chains(x, mesh: Mesh):
    """Place an array sharded along its leading (chains) axis across the mesh."""
    return jax.device_put(x, NamedSharding(mesh, P("chains")))


def replicate(tree, mesh: Mesh):
    """Replicate a pytree (e.g. params) on every device."""
    return jax.device_put(tree, NamedSharding(mesh, P()))


def sharded_mean(local_fn, params, batch, mesh: Mesh):
    """⟨local_fn(params, R)⟩ with ``batch`` sharded over chains and ``params`` replicated.

    The mean is a cross-device all-reduce inserted automatically by ``jax.jit`` — the same
    code runs on 1 or N devices. ``local_fn(params, batch) -> (M,)``.
    """
    params = replicate(params, mesh)
    batch = shard_over_chains(batch, mesh)

    @jax.jit
    def _mean(params, batch):
        return jnp.mean(local_fn(params, batch))

    return _mean(params, batch)
