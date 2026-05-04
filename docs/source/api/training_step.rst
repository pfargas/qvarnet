Training Step (``qvarnet.training_step``)
==========================================

This module isolates the energy and gradient computation from the main training
loop.  The primary entry point is :func:`~qvarnet.training_step.compute_step`,
which dispatches to either vanilla gradient descent or natural gradient (QGT)
depending on configuration.

Energy computation
------------------

Local energy :math:`E_\mathrm{loc}(x) = \hat{H}\psi(x) / \psi(x)` is
computed for each sample, then averaged:

.. math::

    \langle E \rangle = \frac{1}{N} \sum_{i=1}^N E_\mathrm{loc}(x_i)

Gradient computation
--------------------

The variance-minimisation gradient is:

.. math::

    \nabla_\theta \mathcal{L} =
    2\,\mathbb{E}_{x}\!\Big[
        \bigl(E_\mathrm{loc}(x) - \langle E \rangle\bigr)
        \nabla_\theta \log|\psi_\theta(x)|
    \Big]

Module reference
----------------

.. automodule:: qvarnet.training_step
   :members:
   :undoc-members:
   :show-inheritance:
