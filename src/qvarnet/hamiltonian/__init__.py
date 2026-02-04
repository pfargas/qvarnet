from .continuous import HarmonicOscillatorHamiltonian

from .hamiltonian_registry import HAMILTONIAN_REGISTER


def get_hamiltonian(name, **kwargs):
    """Get a hamiltonian class from the HAMILTONIAN_REGISTER.
    Args:
        name: Name of the hamiltonian to retrieve.
    Returns:
        The hamiltonian class associated with the given name."""
    return HAMILTONIAN_REGISTER[name](**kwargs)
