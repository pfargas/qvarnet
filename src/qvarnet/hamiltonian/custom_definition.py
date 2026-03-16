import inspect
from typing import Callable
from flax import struct
from .base import BaseHamiltonian
from .hamiltonian_registry import register_hamiltonian
from .continuous import ContinuousHamiltonian


def define_hamiltonian(name: str):
    def wrapper(potential_fn: Callable):
        # 1. Inspect function
        sig = inspect.signature(potential_fn)
        params = sig.parameters
        param_names = list(params.keys())[1:]  # Skip 'samples'

        # 2. Build Class Dictionary
        # We need to manually build the __annotations__ dict for Flax to read
        annotations = {}
        class_dict = {}

        for pname in param_names:
            default_val = params[pname].default
            if default_val == inspect.Parameter.empty:
                default_val = 0.0

            # 1. Type Hint (Required for dataclasses)
            annotations[pname] = float

            # 2. Field Definition (The actual default value/logic)
            class_dict[pname] = struct.field(default=default_val)

        # 3. Create the Potential Method
        def potential_energy_method(self, samples):
            kwargs = {p: getattr(self, p) for p in param_names}
            return potential_fn(samples, **kwargs)

        # 4. Add method and annotations to class dict
        class_dict["__annotations__"] = annotations
        class_dict["potential_energy"] = potential_energy_method

        # 5. Create a RAW Python class (NOT a dataclass yet)
        # type(name, bases, dict)
        RawClass = type(f"{name}_Hamiltonian", (ContinuousHamiltonian,), class_dict)

        # 6. NOW let Flax turn it into a Frozen Dataclass & PyTree
        DynamicClass = struct.dataclass(RawClass)

        # 7. Register
        register_hamiltonian(name)(DynamicClass)

        return DynamicClass

    return wrapper
