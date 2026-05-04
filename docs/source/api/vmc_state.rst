VMC State (``qvarnet.vmc_state``)
==================================

``VMCState`` extends Flax's ``TrainState`` with VMC-specific fields such as
energy, standard error, acceptance rate, and step size.  Every training step
returns a new ``VMCState`` snapshot that is appended to the history list.

Module reference
----------------

.. automodule:: qvarnet.vmc_state
   :members:
   :undoc-members:
   :show-inheritance:
