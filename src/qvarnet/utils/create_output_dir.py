import os


def create_output_directory(
    base_path: str, experiment_name: str, load_config: bool
) -> str:
    """Create output directory for experiment results. The directory will be created in the
    specified base path. If experiment_name is not provided, a unique run ID will be generated.

    That means the relative directory structure will be:
        ./base_path/experiment_name/
    or
        ./base_path/run_000/
        ./base_path/run_001/
        ...
    depending on whether experiment_name is provided.

    If load_config is True, it indicates that the function is being called to load an existing
    configuration, and thus the directory should already exist. In this case, no new directory
    will be created.

    Args:
        base_path: Base path where results should be saved.
        experiment_name: Name of the experiment (used for subdirectory).
    Returns:
        The full path to the created output directory."""

    if load_config:
        if os.path.exists(base_path):
            return os.path.abspath(os.path.normpath(base_path))
        else:
            raise Warning(
                f"Configuration loading requested, but directory {base_path} does not exist. Creating new directory."
            )

    if not os.path.isabs(base_path):
        cwd = os.getcwd()
        base_path = os.path.join(cwd, base_path)

    if experiment_name is None or experiment_name.strip() == "":
        directories_in_base = os.listdir(base_path)
        run_id = 0
        while f"run_{run_id:03d}" in directories_in_base:
            run_id += 1
        base_path = os.path.join(base_path, f"run_{run_id:03d}")
        os.makedirs(base_path, exist_ok=True)

    else:
        base_path = os.path.join(base_path, experiment_name)
    # if the directory does exists, throw a warning
    if os.path.exists(base_path):
        print(
            f"Warning: Directory {base_path} already exists. Creating different run ID."
        )
        directories_in_base = os.listdir(os.path.dirname(base_path))
        run_id = 0
        while f"{experiment_name}_{run_id:03d}" in directories_in_base:
            run_id += 1
        base_path = os.path.join(
            os.path.dirname(base_path), f"{experiment_name}_{run_id:03d}"
        )
    os.makedirs(base_path, exist_ok=True)
    return os.path.abspath(os.path.normpath(base_path))
