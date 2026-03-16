from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Dict, Any, Optional


class BaseConfig(ABC):
    """Base class for all configuration types."""

    def __init__(self, config_path: Path):
        self.config_path: Path = config_path
        self.data: Dict[str, Any] = self._load_config()
        self._validate_config()

    @abstractmethod
    def _load_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _validate_config(self):
        pass

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def merge_with(self, other_config: Dict[str, Any]):
        """Merge current configuration with another configuration dictionary."""
        self.data = self._merge_dicts(self.data, other_config)
        return self.data

    def _merge_dicts(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        result = d1.copy()
        for key, value in d2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_dicts(result[key], value)
            elif "." in key:
                # Handle nested keys in the form "a.b.c"
                keys = key.split(".")
                sub_dict = result
                for sub_key in keys[:-1]:
                    if sub_key not in sub_dict or not isinstance(
                        sub_dict[sub_key], dict
                    ):
                        sub_dict[sub_key] = {}
                    sub_dict = sub_dict[sub_key]
                sub_dict[keys[-1]] = value
            else:
                result[key] = value
        return result

    def save(self, output_path: Optional[Path] = None):
        path = output_path or self.config_path
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

    def to_dict(self):
        return self.data.copy()


class ExperimentConfig(BaseConfig):
    """Complete experiment configuration with validation."""

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return self._get_default_config()
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _get_default_config(self):
        return {
            "experiment": {
                "name": "qvarnet_experiment",
                "description": "VMC optimization with neural quantum states",
                "tags": ["vmc", "quantum", "optimization"],
                "seed": None,
            },
            "model": {"type": "mlp", "architecture": [2, 10, 1], "activation": "tanh"},
            "training": {
                "batch_size": 1000,
                "num_epochs": 3000,
                "early_stopping": {"patience": 100, "min_delta": 1e-6},
            },
            "optimizer": {"type": "adam", "learning_rate": 1e-3},
            "sampler": {
                "type": "metropolis_hastings",
                "step_size": 0.5,
                "chain_length": 100,
                "thermalization_steps": 50,
            },
            "hamiltonian": {
                "type": "harmonic_oscillator",
                "params": {"omega": 1.0, "dimensions": 2},
            },
            "output": {
                "save_dir": "./results",
                "save_frequency": 100,
                "save_best_only": False,
            },
        }

    def _validate_config(self):
        required_keys = [
            "experiment",
            "model",
            "training",
            "optimizer",
            "sampler",
            "hamiltonian",
            "output",
        ]

        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate model configuration
        # FIXME: Careful with hardcoded validation. Not always true.
        model_config = self.data.get("model", {})
        if not model_config.get("type"):
            raise ValueError("Model type is required")
        # if not model_config.get("architecture"):
        #     raise ValueError("Model architecture is required")

        # Validate training configuration
        training_config = self.data.get("training", {})
        if not isinstance(training_config.get("batch_size"), int):
            raise ValueError("batch_size must be an integer")
        if not isinstance(training_config.get("num_epochs"), int):
            raise ValueError("num_epochs must be an integer")

        # Validate optimizer configuration
        optimizer_config = self.data.get("optimizer", {})
        if not optimizer_config.get("type"):
            raise ValueError("Optimizer type is required")
        if not isinstance(optimizer_config.get("learning_rate"), (int, float)):
            raise ValueError("learning_rate must be a number")

        print(
            f"Configuration validation passed for: {self.data.get('experiment', {}).get('name', 'unnamed')}"
        )


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from file path."""
    return ExperimentConfig(Path(config_path))


def create_config_from_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Create configuration from dictionary."""
    config = ExperimentConfig(Path("dummy.json"))
    config.data = config_dict
    config._validate_config()
    return config
