from abc import ABC, abstractmethod


class BaseHamiltonian(ABC):
    @abstractmethod
    def local_energy(self, params, samples, model_apply):
        pass
