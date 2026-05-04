Quickstart
==========

This page shows the minimal code needed to run a VMC optimisation of the
1-D harmonic oscillator with an MLP ansatz.

Prerequisites
-------------

Install the package and its dependencies following the :doc:`installation` guide.

Python API
----------

.. code-block:: python

    import jax.numpy as jnp
    import optax
    from flax import linen as nn

    from qvarnet import train
    from qvarnet.hamiltonian import get_hamiltonian
    from qvarnet.models import MLP

    # 1. Define the wave function ansatz
    model = MLP(architecture=[1, 32, 32, 1])

    # 2. Choose the Hamiltonian
    hamiltonian = get_hamiltonian("harmonic-oscillator", omega=1.0)

    # 3. Configure the optimizer
    optimizer = optax.adam(learning_rate=1e-3)

    # 4. Run the training loop
    history = train(
        n_epochs=2000,
        shape=(500, 1),          # (n_walkers, n_degrees_of_freedom)
        model=model,
        optimizer=optimizer,
        sampler_params={
            "step_size": 0.5,
            "chain_length": 100,
            "thermalization_steps": 50,
            "thinning_factor": 5,
            "PBC": 0.0,
        },
        hamiltonian=hamiltonian,
        rng_seed=42,
    )

    energies = jnp.array([s.energy for s in history])
    print(f"Final energy: {energies[-1]:.4f}  (exact: 0.5000)")

CLI
---

Run the same experiment using a preset configuration:

.. code-block:: bash

    qvarnet run --preset harmonic_oscillator_standard

List all available presets:

.. code-block:: bash

    qvarnet run --preset-list

Use a custom JSON config file:

.. code-block:: bash

    qvarnet run --config my_experiment.json

Override individual fields at the command line:

.. code-block:: bash

    qvarnet run --config my_experiment.json --override training.num_epochs=5000

Custom Hamiltonians and Models
------------------------------

You can inject custom Python modules at runtime without modifying the source:

.. code-block:: bash

    qvarnet run --config my_config.json --custom-model my_model.py
    qvarnet run --config my_config.json --custom-hamiltonian my_hamiltonian.py

Inside ``my_model.py`` register your class with :func:`qvarnet.models.register_model`
so it is available under a new name in the registry.

Plotting results
----------------

.. code-block:: bash

    qvarnet plot-energy --plot-path results/my_run/
