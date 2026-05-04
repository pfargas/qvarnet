Configuration (``qvarnet.config``)
====================================

Typed, immutable dataclasses that hold parsed training and sampling
configuration.  They replace raw dictionaries inside the training loop,
giving explicit type annotations and default values.

:class:`~qvarnet.config.training_setup.SamplingConfig`
    Holds MCMC parameters: ``step_size``, ``chain_length``,
    ``thermalization_steps``, ``thinning_factor``, ``PBC``, ``is_log_prob``.

:class:`~qvarnet.config.training_setup.TrainingConfig`
    Holds optimisation parameters: ``n_epochs``, ``init_positions``,
    ``is_update_step_size``, ``min_step``, ``max_step``,
    ``target_acceptance``, ``adaptation_rate``.

The two parser functions convert the raw JSON config dict (from the CLI or
Python API) into these dataclasses.

Module reference
----------------

.. automodule:: qvarnet.config.training_setup
   :members:
   :undoc-members:
   :show-inheritance:
