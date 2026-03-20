"""Configuration loader utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "configs/app_config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)