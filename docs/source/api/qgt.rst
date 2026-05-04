Quantum Geometric Tensor (``qvarnet.qgt``)
===========================================

The Quantum Geometric Tensor (QGT) is the metric tensor on the manifold of
quantum states parametrised by :math:`\theta`.  In VMC it acts as a
preconditioner for the energy gradient, giving the natural gradient update
(Stochastic Reconfiguration):

.. math::

    \theta_{t+1} = \theta_t - \eta\, S^{-1}(\theta_t)\,\nabla_\theta E(\theta_t)

where

.. math::

    S_{ij}(\theta) =
    \left\langle O_i^* O_j \right\rangle
    -
    \left\langle O_i^* \right\rangle
    \left\langle O_j \right\rangle,
    \qquad
    O_i = \frac{\partial}{\partial\theta_i} \log|\psi_\theta\rangle

Four solvers are available for the linear system :math:`S\,\delta\theta = \nabla_\theta E`:

* ``"cholesky"`` — Cholesky decomposition (default, good for small systems).
* ``"direct"`` — Direct LU solve via :func:`jnp.linalg.solve`.
* ``"gmres"`` — GMRES iterative solver (large, sparse systems).
* ``"diagonal"`` — Diagonal approximation (very cheap, least accurate).

Pre-built configurations
------------------------

Three ready-to-use :class:`~qvarnet.qgt.QGTConfig` objects are exported:

* :data:`~qvarnet.qgt.DEFAULT_QGT_CONFIG` — Cholesky, lr=1e-3, reg=1e-6.
* :data:`~qvarnet.qgt.MEMORY_EFFICIENT_QGT_CONFIG` — Diagonal, lr=1e-3, reg=1e-4.
* :data:`~qvarnet.qgt.LARGE_SYSTEM_QGT_CONFIG` — GMRES, lr=5e-4, reg=1e-4.

Module reference
----------------

.. automodule:: qvarnet.qgt
   :members:
   :undoc-members:
   :show-inheritance:
