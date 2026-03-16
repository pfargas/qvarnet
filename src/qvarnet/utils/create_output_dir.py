import os
from datetime import datetime


def create_output_directory(
    base_path: str, load_config: bool = False
) -> str:
    """Create output directory for experiment results.
    
    The directory will be created at the exact path specified. If the directory
    already exists, a timestamp will be appended to create a unique directory.
    
    Args:
        base_path: Path where results should be saved.
        load_config: If True, the directory should already exist (for loading
                    existing configurations). No new directory will be created.
    
    Returns:
        The full absolute path to the output directory.
    
    Raises:
        ValueError: If load_config is True but the directory doesn't exist.
    """
    # Convert to absolute path if relative
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.getcwd(), base_path)
    
    # If loading config, directory must exist
    if load_config:
        if not os.path.exists(base_path):
            raise ValueError(
                f"Configuration loading requested, but directory {base_path} does not exist."
            )
        return os.path.abspath(os.path.normpath(base_path))
    
    # If directory doesn't exist, create it
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return os.path.abspath(os.path.normpath(base_path))
    
    # Directory exists - append timestamp to create unique directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = f"{base_path}_{timestamp}"
    
    # Edge case: if timestamped path also exists (very unlikely), add milliseconds
    if os.path.exists(new_path):
        timestamp_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        new_path = f"{base_path}_{timestamp_ms}"
    
    os.makedirs(new_path, exist_ok=True)
    return os.path.abspath(os.path.normpath(new_path))
