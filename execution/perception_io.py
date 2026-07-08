"""Input adapters for perception outputs consumed by the execution layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    raw = Path(os.path.expanduser(str(path)))
    if raw.is_absolute():
        return raw
    return (REPO_ROOT / raw).resolve()


def load_array(path: str | os.PathLike[str]) -> np.ndarray:
    resolved = resolve_repo_path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".npy":
        return np.load(resolved)
    if suffix == ".npz":
        data = np.load(resolved)
        for key in ("points", "point_cloud", "xyz", "arr_0", "depth", "mask"):
            if key in data:
                return data[key]
        raise ValueError(f"NPZ has no supported array key: {resolved}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to read image masks/depths; use .npy/.npz or install pillow."
        ) from exc

    return np.asarray(Image.open(resolved))


def load_point_cloud(path: str | os.PathLike[str]) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(load_array(path), dtype=np.float32)
    if points.ndim == 3:
        points = points.reshape(-1, points.shape[-1])
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud must have shape (N,3+) or (H,W,3+): {path}")

    points = points[:, :3]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    return points.astype(np.float32), {
        "source": "point_cloud_path",
        "path": str(resolve_repo_path(path)),
        "num_points": int(len(points)),
    }


def load_intrinsics(path: str | os.PathLike[str]) -> dict[str, float]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "camera_matrix" in data:
        matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        return {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
        }
    if "K" in data:
        matrix = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        return {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
        }

    required = ("fx", "fy", "cx", "cy")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Intrinsics JSON missing keys {missing}: {resolved}")
    return {key: float(data[key]) for key in required}


def load_transform(path: str | os.PathLike[str]) -> np.ndarray:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("T_world_camera", "camera_to_world", "transform", "matrix"):
        if key in data:
            matrix = np.asarray(data[key], dtype=np.float64)
            break
    else:
        matrix = np.asarray(data, dtype=np.float64)
    matrix = matrix.reshape(4, 4)
    return matrix


def backproject_masked_depth(
    *,
    depth_path: str | os.PathLike[str],
    mask_path: str | os.PathLike[str],
    intrinsics_path: str | os.PathLike[str],
    depth_scale: float = 1.0,
    camera_to_world_path: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth = np.asarray(load_array(depth_path), dtype=np.float32) * float(depth_scale)
    mask = np.asarray(load_array(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Depth and mask shape mismatch: depth={depth.shape}, mask={mask.shape}"
        )

    intrinsics = load_intrinsics(intrinsics_path)
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u.astype(np.float32) - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (v.astype(np.float32) - intrinsics["cy"]) * z / intrinsics["fy"]
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    frame = "camera"
    transform_path = None
    if camera_to_world_path:
        transform = load_transform(camera_to_world_path)
        hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
        points = (hom @ transform.T)[:, :3].astype(np.float32)
        frame = "world"
        transform_path = str(resolve_repo_path(camera_to_world_path))

    return points, {
        "source": "mask_depth_intrinsics",
        "depth_path": str(resolve_repo_path(depth_path)),
        "mask_path": str(resolve_repo_path(mask_path)),
        "intrinsics_path": str(resolve_repo_path(intrinsics_path)),
        "camera_to_world_path": transform_path,
        "depth_scale": float(depth_scale),
        "frame": frame,
        "num_points": int(len(points)),
    }


def build_target_points_from_scene(scene: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    point_cloud_path = scene.get("point_cloud_path")
    if point_cloud_path:
        points, info = load_point_cloud(point_cloud_path)
        info["frame"] = scene.get("point_cloud_frame", "world")
        info["unit"] = scene.get("point_cloud_unit", "meter")
        if info["unit"] == "millimeter":
            points = points / 1000.0
            info["unit_converted_to"] = "meter"
        return points, info

    mask_path = scene.get("mask_path")
    depth_path = scene.get("depth_path")
    intrinsics_path = scene.get("camera_intrinsics_path")
    if mask_path and depth_path and intrinsics_path:
        return backproject_masked_depth(
            depth_path=depth_path,
            mask_path=mask_path,
            intrinsics_path=intrinsics_path,
            depth_scale=float(scene.get("depth_scale", 1.0)),
            camera_to_world_path=scene.get("camera_to_world_path"),
        )

    return None, {
        "source": "pybullet_segmentation_fallback",
        "reason": "scene.point_cloud_path not provided and mask/depth/intrinsics incomplete",
    }
