"""Configuration loading helpers."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/course_project.yaml") -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create configured output directories if needed."""
    Path(config["outputs"]["results_dir"]).mkdir(parents=True, exist_ok=True)
