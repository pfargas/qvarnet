Sampling Step (``qvarnet.sampling_step``)
==========================================

High-level orchestration of one MCMC sampling step.

:func:`~qvarnet.sampling_step.sample_and_process` is the JIT-compiled
entry point called at every training iteration.  It:

1. Generates random numbers for all chains in one batch.
2. Runs :func:`~qvarnet.samplers.mh_chain` in parallel across chains via
   :func:`jax.vmap`.
3. Discards burn-in samples (thermalization).
4. Applies thinning to reduce sample autocorrelation.
5. Flattens the per-chain samples into a single batch for energy evaluation.

:func:`~qvarnet.sampling_step.create_sampler_fn` is a helper that wraps
a single-chain MH kernel with ``jax.vmap`` to produce a multi-chain sampler.

Module reference
----------------

.. automodule:: qvarnet.sampling_step
   :members:
   :undoc-members:
   :show-inheritance:
