"""qvarnet -- variational Monte Carlo with neural-network wavefunctions, in JAX.

Optimise a log-amplitude ansatz log|psi_theta(x)| against a Hamiltonian, sampling
|psi|^2 by Metropolis-Hastings::

    from qvarnet import VMC, TrainingConfig, SamplingConfig

    result = VMC(
        shape=(n_chains, n_particles * n_dim),
        model=psi,
        optimizer=optax.adam(1e-3),
        hamiltonian=ham,
        training_config=TrainingConfig(n_epochs=5_000),
        sampler_params=SamplingConfig(step_size=0.5, chain_length=21,
                                      thermalization_steps=20, thinning_factor=1),
    ).run()

Everything pluggable is an object you construct and pass -- the ansatz, the
Hamiltonian, the optimizer, the sampler, the callbacks. To add one, write a class
and pass an instance; there is nothing to register.

The package is layered, and imports only ever point downward::

    core        primitives with no qvarnet dependencies
    physics     Hamiltonians, boundaries, particle species
    ansatz      wavefunctions and their building blocks
    sampling    proposals and samplers
    optim       QGT / stochastic reconfiguration, TDVP, auxiliary losses
    analysis    convergence diagnostics and property estimators
    callbacks   hooks into the training loop
    vmc         the driver

``scripts/check_layers.py`` enforces that.
"""

from qvarnet.ansatz.layers import AppendPairwiseDiffs, SubtractCM
from qvarnet.core.coords import JacobiCoords, LabCoords
from qvarnet.physics.boundaries import (
    BoundaryHamiltonian,
    BoundaryModel,
    NoBoundary,
    PeriodicBoundary,
)

# After .physics.boundaries: .periodic subclasses BoundaryHamiltonian, and importing
# it earlier would be circular (boundaries imports hamiltonian.continuous).
from qvarnet.physics.hamiltonian.periodic import (
    LatticeBoseHamiltonian,
    PenetrableSphereHamiltonian,
)
from qvarnet.physics.particles import Particles
from qvarnet.recipes import adam_train, sr_train
from qvarnet.sampling import Metropolis, OrderedMetropolis, Sampler
from qvarnet.sampling.config import SamplingConfig
from qvarnet.vmc.config import ChainInitAndWarmupConfig, CuspConfig, TrainingConfig
from qvarnet.vmc.driver import VMC, train
from qvarnet.vmc.evaluate import EvalResult, evaluate, evaluate_result
from qvarnet.vmc.result import TrainResult

__all__ = [
    # driver
    "VMC",
    "train",
    "TrainResult",
    "adam_train",
    "sr_train",
    # configuration
    "TrainingConfig",
    "SamplingConfig",
    "ChainInitAndWarmupConfig",
    "CuspConfig",
    "LabCoords",
    "JacobiCoords",
    # samplers
    "Sampler",
    "Metropolis",
    "OrderedMetropolis",
    # physics
    "LatticeBoseHamiltonian",
    "PenetrableSphereHamiltonian",
    "NoBoundary",
    "PeriodicBoundary",
    "BoundaryModel",
    "BoundaryHamiltonian",
    "Particles",
    # ansatz building blocks
    "SubtractCM",
    "AppendPairwiseDiffs",
    # evaluation
    "evaluate",
    "evaluate_result",
    "EvalResult",
]
