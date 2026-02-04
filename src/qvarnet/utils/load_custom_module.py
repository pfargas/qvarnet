import sys
import importlib.util
from pathlib import Path
import hashlib


def load_custom_module(module_path: str):
    """Dynamically loads a python file as a module."""
    path = Path(module_path).resolve()
    if not path.exists():
        print(f"Error: Custom module file not found at {path}")
        sys.exit(1)

    module_name = "custom_" + hashlib.md5(str(path).encode()).hexdigest()

    print(f"Loading custom definitions from: {path}")
    print(f"Assigned module name: {module_name}")

    # Python magic to load a file as a module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
