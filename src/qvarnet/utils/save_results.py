import os
import json
import csv
from datetime import datetime


def save_energy_history(base_path: str, energy_values) -> None:
    """Save energy history as CSV with header.
    
    Args:
        base_path: Directory where the file should be saved.
        energy_values: Array or list of energy values per epoch.
    """
    filepath = os.path.join(base_path, "energy_history.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'energy'])
        for i, energy in enumerate(energy_values):
            writer.writerow([i, float(energy)])


def save_metrics(base_path: str, metrics: dict) -> None:
    """Save final metrics as JSON.
    
    Args:
        base_path: Directory where the file should be saved.
        metrics: Dictionary of metric names and values.
    """
    filepath = os.path.join(base_path, "metrics.json")
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_config(base_path: str, config: dict) -> None:
    """Save experiment configuration as JSON.
    
    Args:
        base_path: Directory where the file should be saved.
        config: Dictionary containing full experiment configuration.
    """
    filepath = os.path.join(base_path, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


# Keep the old save_results function for backward compatibility, but simplify it
def save_results(base_path, csv_delimiters=None, **kwargs) -> bool:
    """Legacy save_results function for backward compatibility.
    
    DEPRECATED: Use save_energy_history(), save_metrics(), or save_config() instead.
    
    Args:
        base_path: Directory where files should be saved.
        csv_delimiters: (Deprecated) CSV delimiters.
        **kwargs: Key-value pairs where key is filename and value is data.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        os.makedirs(base_path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return False

    for key, value in kwargs.items():
        try:
            # if it's a dictionary, save as a json
            if isinstance(value, dict):
                with open(os.path.join(base_path, f"{key}.json"), "w") as f:
                    json.dump(value, f, indent=4)
                continue
            elif (
                isinstance(value, list)
                and all(isinstance(item, str) for item in value)
                and csv_delimiters
                and all(
                    any(delim in item for delim in csv_delimiters) for item in value
                )
            ):
                with open(os.path.join(base_path, f"{key}.csv"), "w") as f:
                    for line in value:
                        f.write(line)
                        f.write("\n")
                continue
            # otherwise, save as a text file
            with open(os.path.join(base_path, f"{key}.txt"), "w") as f:
                for num in value:
                    f.write(str(num))
                    f.write("\n")
        except Exception as e:
            print(f"Error saving {key}: {e}")
            return False
    return True
