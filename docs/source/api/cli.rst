CLI (``qvarnet.cli``)
=====================

The ``qvarnet`` command-line interface is the primary way to run experiments
without writing Python.  It exposes two sub-commands:

``run``
    Launch a VMC training run.

``plot-energy``
    Plot the energy history from a previous run.

Usage
-----

.. code-block:: text

    qvarnet run [--config FILE] [--preset NAME] [--override KEY=VALUE ...] [options]
    qvarnet plot-energy --plot-path PATH

Options for ``run``
~~~~~~~~~~~~~~~~~~~

.. option:: --config, -c FILE

    Path to a JSON experiment configuration file.

.. option:: --preset NAME

    Use a bundled preset configuration by name.

.. option:: --preset-list

    Print all available presets and exit.

.. option:: --override, -o KEY=VALUE

    Override one configuration value.  Can be repeated.
    Nested keys use dot notation: ``training.num_epochs=5000``.

.. option:: --config-dump

    Print the resolved configuration and continue.

.. option:: --custom-model, -cm FILE

    Path to a Python file that registers a custom model class.

.. option:: --custom-hamiltonian, -ch FILE

    Path to a Python file that registers a custom Hamiltonian class.

Configuration system
--------------------

Configurations are JSON files validated by
:class:`~qvarnet.cli.cli_config.base.ExperimentConfig`.  The required
top-level keys are:

* ``experiment`` — name, description, seed.
* ``model`` — type and architecture fields.
* ``training`` — ``batch_size``, ``num_epochs``.
* ``optimizer`` — ``type`` (``"adam"`` or ``"sgd"``), ``learning_rate``.
* ``sampler`` — step size, chain length, thermalization, thinning, PBC.
* ``hamiltonian`` — name and optional params.
* ``output`` — ``save_dir``, ``save_checkpoints``.

Preset configurations live in
``src/qvarnet/cli/cli_config/presets/``.

Module reference
----------------

.. automodule:: qvarnet.cli.run
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qvarnet.cli.cli_config.base
   :members:
   :undoc-members:
   :show-inheritance:
