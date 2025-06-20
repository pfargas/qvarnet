Hamiltonians
=================

.. automodule:: qvarnet.hamiltonians
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
----------------
Below are examples of how to use the Hamiltonians in `qvarnet`.

.. code-block:: python

    from qvarnet.hamiltonians import HarmonicOscillator

    # Initialize the Hamiltonian
    hamiltonian = HarmonicOscillator()

    # Calculate the energy
    local_energies = hamiltonian(positions)

    # Supposing that `positions` is distributed following the wavefunction
    print(f"Calculated energy: {torch.mean(local_energies)}")