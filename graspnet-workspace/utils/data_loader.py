"""Small data helpers shared by the real-world grasp pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def sample_points(cloud: np.ndarray, num_points: int) -> np.ndarray:
    replace = len(cloud) < num_points
    indices = np.random.choice(len(cloud), num_points, replace=replace)
    return cloud[indices].astype(np.float32, copy=False)
