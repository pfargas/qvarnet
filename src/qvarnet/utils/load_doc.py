# src/qvarnet/utils/doc_loader.py
from pathlib import Path
from inspect import cleandoc


def load_doc(filename):
    def decorator(obj):
        # 1. Path to this file (doc_loader.py)
        # 2. .parents[1] goes UP two levels (from doc_loader.py -> utils -> qvarnet)
        package_root = Path(__file__).resolve().parents[1]
        file_path = package_root / "_docs" / filename
        print("*-" * 40)
        print(file_path)
        print("*-" * 40)

        if file_path.exists():
            obj.__doc__ = cleandoc(file_path.read_text(encoding="utf-8"))
        else:
            obj.__doc__ = f"Documentation file not found at: {file_path}"

        return obj

    return decorator
