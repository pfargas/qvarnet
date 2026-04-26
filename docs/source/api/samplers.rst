Samplers (``qvarnet.samplers``)
================================

Low-level Metropolis-Hastings kernels.

Two kernel variants are provided:

:func:`~qvarnet.samplers.sampler.mh_kernel`
    Operates in probability space (:math:`P(x) = |\psi(x)|^2`).
    Suitable for models that output :math:`\psi` directly.

:func:`~qvarnet.samplers.sampler.mh_kernel_log`
    Operates in log-probability space (:math:`\log P(x)`).
    More numerically stable for models that output :math:`\log|\psi|`.

Both kernels accept/reject with the standard MH criterion:

.. math::

    A(x \to x') = \min\!\left(1,\, \frac{P(x')}{P(x)}\right)

:func:`~qvarnet.samplers.sampler.mh_chain` assembles a full MCMC chain by
scanning a kernel over pre-generated random numbers with :func:`jax.lax.scan`,
which allows efficient JIT compilation of the entire chain.

Module reference
----------------

.. automodule:: qvarnet.samplers.sampler
   :members:
   :undoc-members:
   :show-inheritance:
