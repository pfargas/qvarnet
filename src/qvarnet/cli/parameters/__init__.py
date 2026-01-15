from .base import ExperimentConfig, load_config, create_config_from_dict
from pathlib import Path
from typing import Dict, List


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from file path."""
    return ExperimentConfig(Path(config_path))


def create_preset(preset_name: str, **overrides) -> ExperimentConfig:
    """Create configuration from preset with overrides."""
    presets_dir = Path(__file__).parent / "presets"
    preset_path = presets_dir / f"{preset_name}.json"
    
    if not preset_path.exists():
        available_presets = _list_available_presets()
        raise ValueError(f"Preset '{preset_name}' not found. Available: {available_presets}")
    
    base_config = ExperimentConfig(preset_path)
    if overrides:
        base_config.data = base_config.merge_with(overrides)
    return base_config


def _list_available_presets() -> List[str]:
    """List all available preset configurations."""
    presets_dir = Path(__file__).parent / "presets"
    return [f.stem for f in presets_dir.glob("*.json")]


def list_presets():
    """List all available preset configurations with descriptions."""
    presets = []
    presets_dir = Path(__file__).parent / "presets"
    
    for preset_file in presets_dir.glob("*.json"):
        config = ExperimentConfig(preset_file)
        experiment_info = config.get("experiment", {})
        presets.append({
            "name": preset_file.stem,
            "description": experiment_info.get("description", "No description"),
            "tags": experiment_info.get("tags", [])
        })
    
    return presets


def validate_config(config_dict: dict) -> bool:
    """Validate configuration dictionary against basic requirements."""
    try:
        config = create_config_from_dict(config_dict)
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def get_default_config() -> ExperimentConfig:
    """Get the default configuration (harmonic oscillator standard)."""
    return create_preset("harmonic_oscillator_standard")