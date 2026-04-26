Training Loop (``qvarnet.train``)
=================================

The ``train`` module contains the main VMC optimisation loop.

It ties together MCMC sampling (:mod:`qvarnet.sampling_step`),
energy/gradient computation (:mod:`qvarnet.training_step`), and
adaptive step-size control into a single call.

Overview
--------

.. math::

    \theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)

where the loss is defined as:

.. math::

    \mathcal{L}(\theta) = 2\,
    \mathbb{E}_{x \sim |\psi_\theta|^2}\!\Big[
        \bigl(E_\mathrm{loc}(x) - \langle E \rangle\bigr)
        \log|\psi_\theta(x)|
    \Big]

This biases optimisation toward low-variance, low-energy wave functions.

Module reference
----------------

.. automodule:: qvarnet.train
   :members:
   :undoc-members:
   :show-inheritance:
