"""Path helper utilities."""

from pathlib import Path
from typing import Dict


def ensure_directories(paths_config: Dict[str, str]) -> None:
    """Create configured directories if they do not exist."""
    for path_str in paths_config.values():
        Path(path_str).mkdir(parents=True, exist_ok=True)