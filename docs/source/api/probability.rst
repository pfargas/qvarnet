Probability Functions (``qvarnet.probability``)
================================================

Factory functions that build the probability density used by the MCMC sampler
from the model's output type.

Two output conventions are supported:

* **Direct models** — output :math:`\psi(x)`, so probability is
  :math:`P(x) = |\psi(x)|^2`.
* **Log models** — output :math:`\log|\psi(x)|`, so the probability is
  :math:`\log P(x) = 2\log|\psi(x)|` (used in log-space MCMC kernels for
  numerical stability).

The factory :func:`~qvarnet.probability.build_prob_fn` selects the correct
variant automatically.

Module reference
----------------

.. automodule:: qvarnet.probability
   :members:
   :undoc-members:
   :show-inheritance:
