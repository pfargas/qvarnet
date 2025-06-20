import torch
from torch import nn

class GeneralHamiltonian(nn.Module):
    """
    General Hamiltonian class
    """

    def __init__(self, model: nn.Module = None):
        super().__init__()
        self.model = model

    def kinetic_local_energy(self, x: torch.Tensor) -> torch.Tensor:
        psi = self.model(x)
        # numerical second derivative no pytorch
        # d2psi = torch.zeros_like(psi)
        # for i,x_val in enumerate(x):
        #     if i == 0:
        #         d2psi[i] = (psi[i+1] - 2*psi[i] + psi[i]) / ((x[i+1] - x[i])**2)
        #     elif i == len(x) - 1:
        #         d2psi[i] = (psi[i] - 2*psi[i] + psi[i-1]) / ((x[i] - x[i-1])**2)
        #     else:
        #         d2psi[i] = (psi[i+1] - 2*psi[i] + psi[i-1]) / ((x[i+1] - x[i-1])**2)
                
        dpsi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(x),
                                    create_graph=True, retain_graph=True)[0]
        
        d2psi = torch.autograd.grad(dpsi, x, grad_outputs=torch.ones_like(dpsi),
                                    create_graph=True, retain_graph=True)[0]

        return -0.5 * d2psi/psi
    
    def trapping_potential(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def interaction_local_potential(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the local energy of the system.
        """
        return self.kinetic_local_energy(x) + self.trapping_potential(x) + self.interaction_local_potential(x)
    
    def __repr__(self):
        """
        String representation of the Hamiltonian.
        """
        return f"{self.__class__.__name__}(): "
    

class HarmonicOscillator(GeneralHamiltonian):
    """
    Harmonic oscillator Hamiltonian class
    
    This class implements the Hamiltonian for a 1D, single particle quantum harmonic oscillator.
    
    The Hamiltonian is given by:
    
    .. math:: 
    
        H = -\\frac{\\hbar^2}{2m} \\frac{d^2}{dx^2} + \\frac{1}{2} m \\omega^2 x^2

    where :math:`\\hbar` is the reduced Planck's constant, :math:`m` is the mass of the particle, and :math:`\\omega` is the angular frequency of the oscillator.
    The kinetic energy operator is given by:
    :math:`T = -\\frac{\\hbar^2}{2m} \\frac{d^2}{dx^2}`
    The potential energy operator is given by:
    :math:`V = \\frac{1}{2} m \\omega^2 x^2`
    The interaction potential is set to zero for this case.
    
    Args:
        omega (float): Angular frequency of the oscillator. Default is 1.0.
    """

    def __init__(self, model: nn.Module, omega: float = 1.0):
        super().__init__(model=model)
        self.omega = omega
    
    def trapping_potential(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.omega**2 * x**2
    

class GrossPitaevskii(GeneralHamiltonian):
    """
    Gross-Pitaevskii Hamiltonian class
    
    This class implements the Hamiltonian for a 1D, single particle Gross-Pitaevskii equation.
    
    The Hamiltonian is given by:

    .. math::

        H = -\\frac{\\hbar^2}{2m} \\frac{d^2}{dx^2} + V(x) + g |\\psi(x)|^2
    
    where :math:`\\hbar` is the reduced Planck's constant, :math:`m` is the mass of the particle, :math:`V(x)` is the external potential, and :math:`g` is the interaction strength.
    
    Args:
        omega (float): Angular frequency of the oscillator. Default is 1.0.
        g (float): Interaction strength. Default is 1.0.
    """

    def __init__(self, model: nn.Module, omega: float = 1.0, g: float = 1.0):
        super().__init__(model=model)
        self.omega = omega
        self.g = g
    
    def trapping_potential(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.omega**2 * x**2
    
    def interaction_local_potential(self, x: torch.Tensor) -> torch.Tensor:
        """
        Interaction potential term in the Gross-Pitaevskii equation.

        .. math::

            V_{int}(x) = g |\\psi(x)|^2\\psi(x)
        
        where :math:`g` is the interaction strength and :math:`\\psi(x)` is the wave function.
        """
        psi = self.model(x)
        return self.g * torch.conj(psi) # can be done like this?
        return self.g * psi.pow(2)/psi # This is the normal way of doing it.
    
