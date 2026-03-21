"""Main entry point for local setup checks."""

from src.utils.config import load_config
from src.utils.paths import ensure_directories


def main() -> None:
    """Initialize project folders and validate config."""
    config = load_config()
    ensure_directories(config["paths"])
    print("Project structure initialized successfully.")


if __name__ == "__main__":
    main()