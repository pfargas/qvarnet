Hamiltonians (``qvarnet.hamiltonian``)
========================================

Hamiltonian classes encode the physics of the system.  They are responsible
for computing the local energy

.. math::

    E_\mathrm{loc}(x) = \frac{\hat{H}\psi(x)}{\psi(x)}
    = -\frac{1}{2}\frac{\nabla^2\psi(x)}{\psi(x)} + V(x)

All Hamiltonians inherit from :class:`~qvarnet.hamiltonian.base.BaseHamiltonian`
and are registered in :data:`~qvarnet.hamiltonian.hamiltonian_registry.HAMILTONIAN_REGISTER`.

Available Hamiltonians
----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Registry key
     - Description
   * - ``"harmonic-oscillator"``
     - :math:`V = \frac{1}{2}\omega^2 r^2`.  Parameter: ``omega``.
   * - ``"nn-oscillator"``
     - Harmonic trap + nearest-neighbour interactions.
       Parameters: ``omega_trap``, ``omega_interaction``, ``with_pbc``.
   * - ``"soft-core"``
     - Step potential :math:`V = V_0` for :math:`r < R`.
       Parameters: ``R``, ``V0``.
   * - ``"gross-struct-hamiltonian"``
     - Gross-structure atomic Hamiltonian (electron-nuclear attraction).
       Parameters: ``Z``, ``n_fermions``.

Registering a Custom Hamiltonian
---------------------------------

.. code-block:: python

    from qvarnet.hamiltonian.hamiltonian_registry import register_hamiltonian
    from qvarnet.hamiltonian.continuous import ContinuousHamiltonian
    from flax import struct
    import jax.numpy as jnp

    @register_hamiltonian("double-well")
    @struct.dataclass
    class DoubleWellHamiltonian(ContinuousHamiltonian):
        a: float = 1.0

        def potential_energy(self, samples):
            return (samples**2 - self.a**2)**2

Then inject at runtime:

.. code-block:: bash

    qvarnet run --config my_config.json --custom-hamiltonian my_hamiltonian.py

Module reference
----------------

Base class
~~~~~~~~~~

.. automodule:: qvarnet.hamiltonian.base
   :members:
   :undoc-members:
   :show-inheritance:

Continuous Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qvarnet.hamiltonian.continuous
   :members:
   :undoc-members:
   :show-inheritance:

Registry
~~~~~~~~

.. automodule:: qvarnet.hamiltonian.hamiltonian_registry
   :members:
   :undoc-members:

Kinetic term
~~~~~~~~~~~~

.. automodule:: qvarnet.hamiltonian.kinetic
   :members:
   :undoc-members:

Laplacian
~~~~~~~~~

.. automodule:: qvarnet.hamiltonian.laplacian
   :members:
   :undoc-members:
