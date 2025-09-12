import os
from datetime import datetime
from typing import Any
from joblib import dump, load


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_model(model: Any, base_name: str, models_dir: str = "Models", timestamp: bool = True, ext: str = "joblib") -> str:
    """Save a model under Models/ with optional timestamp to avoid overwrites.

    Returns the full filesystem path to the saved artifact.
    """
    ensure_dir(models_dir)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
    filename = f"{base_name}_{suffix}.{ext}" if suffix else f"{base_name}.{ext}"
    path = os.path.join(models_dir, filename)
    dump(model, path)
    return path


def load_model(path: str) -> Any:
    return load(path)

