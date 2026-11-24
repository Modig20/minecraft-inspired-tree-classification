import json
from pathlib import Path


class FileMissingError(FileNotFoundError):
    pass


def load_classes(processed_dir):
    path = Path(processed_dir) / "classes.json"
    if not path.exists():
        raise FileMissingError("Class list not found. Run training/prepare_dataset.py first.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
