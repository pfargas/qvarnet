Training State Management (``qvarnet.training_state``)
=======================================================

This module provides helper classes for recording and accessing training
diagnostics over the course of an optimisation run.

:class:`~qvarnet.training_state.TrainingHistory`
    Appends energy, standard error, acceptance rate, and step size at every
    epoch.  Provides conversion to JAX arrays for efficient post-processing.

:class:`~qvarnet.training_state.StateManager`
    Wraps :class:`~qvarnet.training_state.TrainingHistory` and adds
    checkpoint load/save so experiments can be resumed from a previous run.

Module reference
----------------

.. automodule:: qvarnet.training_state
   :members:
   :undoc-members:
   :show-inheritance:
