Callbacks (``qvarnet.callbacks``)
==================================

Lightweight hooks called during the training loop.

:func:`~qvarnet.callbacks.callback.nan_callback`
    JIT-compiled check that returns ``True`` if ``x`` contains any NaN.
    Used to detect divergence and stop training early.

:func:`~qvarnet.callbacks.callback.update_best_params`
    Functionally updates the best stored parameters when the current energy
    is lower than the running minimum.  Uses :func:`jax.lax.cond` to keep
    the operation JIT-compatible.

Module reference
----------------

.. automodule:: qvarnet.callbacks.callback
   :members:
   :undoc-members:
