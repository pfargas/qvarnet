from .continuous import HarmonicOscillatorHamiltonian

# NOTE: LatticeBoseHamiltonian lives in .periodic, which subclasses BoundaryHamiltonian
# from qvarnet.boundaries. Importing it here would create a circular import (boundaries
# imports hamiltonian.continuous). It is imported from qvarnet/__init__ instead, after
# boundaries is fully loaded; that import also triggers its registry registration.

from .hamiltonian_registry import HAMILTONIAN_REGISTER, list_hamiltonians

from .custom_definition import define_hamiltonian


def get_hamiltonian(name, **kwargs):
    """Get a hamiltonian class from the HAMILTONIAN_REGISTER.
    Args:
        name: Name of the hamiltonian to retrieve.
    Returns:
        The hamiltonian class associated with the given name."""
    try:
        return HAMILTONIAN_REGISTER[name](**kwargs)
    except TypeError as e:
        print(
            f"Hamiltonian '{name}' could not be instantiated with the provided arguments: {e}. Skipping arguments."
        )
        return HAMILTONIAN_REGISTER[name]()
