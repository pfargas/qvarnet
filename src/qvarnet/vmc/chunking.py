"""Evaluate the local energy in chunks of samples, so a big batch fits in GPU memory.

The local energy (its Laplacian) is the dominant memory cost of a training step. Wrapping the
Hamiltonian splits only that evaluation; the rest of the step is untouched.
"""

from typing import Any

import jax
from flax import struct

MEMORY_FRACTION = 0.8  # use at most this fraction of the device limit (XLA's estimate is not exact)


@struct.dataclass  # a pytree like the Hamiltonians it wraps (compute_step traces it)
class ChunkedHamiltonian:
    """``inner`` with ``local_energy`` evaluated in ``n_chunks`` sequential chunks (``jax.lax.map``).

    Same E_loc as ``inner`` for a deterministic Laplacian. With a stochastic one, each chunk gets
    its own key (still unbiased, not bitwise equal). Every other attribute is ``inner``'s.
    The batch size must be divisible by ``n_chunks``.
    """

    inner: Any
    n_chunks: int = struct.field(pytree_node=False)

    def __getattr__(self, name):
        if name == "inner":  # not set yet (copy/pickle): avoid infinite recursion
            raise AttributeError(name)
        return getattr(self.inner, name)

    def local_energy(self, params, samples, model_apply, key=None):
        b, dof = samples.shape
        if b % self.n_chunks:
            raise ValueError(f"batch {b} not divisible by n_chunks = {self.n_chunks}")
        xs = samples.reshape(self.n_chunks, b // self.n_chunks, dof)

        def one(x, k):
            e = self.inner.local_energy(params, x, model_apply, key=k)
            assert e.size == x.shape[0], e.shape  # one value per sample
            return e.reshape(x.shape[0])

        if key is None:
            out = jax.lax.map(lambda x: one(x, None), xs)
        else:
            out = jax.lax.map(lambda a: one(*a), (xs, jax.random.split(key, self.n_chunks)))
        return out.reshape(b)


def _device_memory_limit():
    """Bytes the allocator may use on the default device, or None (CPU reports none)."""
    stats = jax.devices()[0].memory_stats()
    return (stats or {}).get("bytes_limit")


def choose_n_chunks(ctx, update_fn, max_chunks):
    """Set ``ctx.hamiltonian`` to the fewest chunks (1, 2, 4, ... ≤ max_chunks) whose compiled step fits.

    Compiles ``update_fn`` ahead of time and reads XLA's memory estimate; nothing runs. No-op when
    ``max_chunks <= 1`` or the device reports no memory limit (CPU). Raises MemoryError if even
    ``max_chunks`` does not fit. Returns the chosen number of chunks.
    """
    limit = _device_memory_limit() if max_chunks > 1 else None
    if limit is None:
        return 1
    inner, tried = ctx.hamiltonian, []
    args = dict(state=ctx.state, key=ctx.key, current_pos=ctx.positions, prob_fn=ctx.prob_fn,
                step_size=ctx.step_size, sampling_config=ctx.sampling_config,
                training_config=ctx.training_config)
    n = 1
    while n <= max_chunks:
        h = inner if n == 1 else ChunkedHamiltonian(inner, n)
        try:
            m = update_fn.lower(hamiltonian=h, **args).compile().memory_analysis()
            need = (m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes
                    - m.alias_size_in_bytes)
        except ValueError:  # batch not divisible by n
            n *= 2
            continue
        except jax.errors.JaxRuntimeError:  # compilation itself ran out of memory
            need = float("inf")
        tried.append(f"{n}: {need / 2**30:.2f} GiB")
        if need <= MEMORY_FRACTION * limit:
            print(f"[qvarnet] local energy in {n} chunk(s): ~{need / 2**30:.2f} GiB of {limit / 2**30:.2f} GiB")
            ctx.hamiltonian = h
            return n
        n *= 2
    raise MemoryError(f"training step does not fit in {MEMORY_FRACTION:.0%} of {limit / 2**30:.2f} GiB "
                      f"with up to {max_chunks} chunks (estimates per n_chunks: {', '.join(tried)}). "
                      "Use fewer chains or a smaller model.")
