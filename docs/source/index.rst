qvarnet — Neural Quantum States for VMC
========================================

**qvarnet** is a modular Python library for Variational Monte Carlo (VMC)
simulations using artificial neural networks as ansatz wave functions.
It is built on top of `JAX <https://jax.readthedocs.io>`_ and
`Flax <https://flax.readthedocs.io>`_ for high-performance, JIT-compiled
quantum simulations on CPU and GPU.

**Key features**

* Metropolis-Hastings MCMC sampling with multi-chain parallelism via ``jax.vmap``
* Variance-minimisation gradient for energy optimisation
* Natural gradient (stochastic reconfiguration) via the Quantum Geometric Tensor
* Flexible model registry: MLP, Deep Set, Fermionic ansätze, and custom models
* Hamiltonian registry: harmonic oscillator, nearest-neighbour oscillator,
  soft-core potential, and custom Hamiltonians
* CLI with JSON preset configurations

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
