"""Configuration helpers for the real-world grasp pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_realworld_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            f"Reading {config_path} requires PyYAML. Install yaml support or keep defaults in code."
        ) from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def config_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def config_path(config: dict[str, Any], dotted_key: str, workspace_root: Path, default: Path | str) -> Path:
    raw_value = config_get(config, dotted_key, default)
    path = Path(os.path.expandvars(str(raw_value))).expanduser()
    if path.is_absolute():
        return path
    return (workspace_root / path).resolve()
