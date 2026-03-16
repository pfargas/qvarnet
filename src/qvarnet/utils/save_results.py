import os
import json
import csv
from datetime import datetime


def save_energy_history(base_path: str, energy_values, energy_std_values) -> None:
    """Save energy history as CSV with header.

    Args:
        base_path: Directory where the file should be saved.
        energy_values: Array or list of energy values per epoch.
        energy_std_values: Array or list of energy standard deviation values per epoch.
    """
    filepath = os.path.join(base_path, "energy_history.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "energy", "energy_std"])
        for i, energy in enumerate(energy_values):
            writer.writerow([i, float(energy), float(energy_std_values[i])])


def save_metrics(base_path: str, metrics: dict, name: str = "metrics.json") -> None:
    """Save final metrics as JSON.

    Args:
        base_path: Directory where the file should be saved.
        metrics: Dictionary of metric names and values.
    """
    filepath = os.path.join(base_path, name)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def save_config(base_path: str, config: dict) -> None:
    """Save experiment configuration as JSON.

    Args:
        base_path: Directory where the file should be saved.
        config: Dictionary containing full experiment configuration.
    """
    filepath = os.path.join(base_path, "config.json")
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
