"""Map a Perception object mask to the corresponding PyBullet scene object."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PYBULLET_BODY_ID_MASK = (1 << 24) - 1


def decode_body_ids(segmentation: np.ndarray) -> np.ndarray:
    """Decode PyBullet's body/link segmentation values into body IDs."""
    values = np.asarray(segmentation, dtype=np.int64)
    return np.where(values >= 0, values & PYBULLET_BODY_ID_MASK, -1)


def load_object_mask(
    mask_path: str | Path,
    *,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a binary Perception mask and align it to the camera image size."""
    path = Path(mask_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Perception object mask not found: {path}")

    source = Image.open(path).convert("L")
    source_shape = (source.height, source.width)
    resized = source_shape != target_shape
    if resized:
        source = source.resize(
            (int(target_shape[1]), int(target_shape[0])),
            resample=Image.Resampling.NEAREST,
        )
    mask = np.asarray(source, dtype=np.uint8) > 0
    if not np.any(mask):
        raise ValueError(f"Perception object mask is empty: {path}")

    return mask, {
        "mask_path": str(path),
        "source_shape": [int(source_shape[0]), int(source_shape[1])],
        "target_shape": [int(target_shape[0]), int(target_shape[1])],
        "resized": bool(resized),
        "mask_pixels": int(np.count_nonzero(mask)),
    }


def match_scene_object_by_mask(
    scene,
    segmentation: np.ndarray,
    mask_path: str | Path,
    *,
    minimum_iou: float = 0.01,
):
    """Return the scene object whose visible segmentation has maximum mask IoU."""
    body_ids = decode_body_ids(segmentation)
    reference_mask, diagnostics = load_object_mask(
        mask_path,
        target_shape=body_ids.shape,
    )

    candidates: list[dict[str, Any]] = []
    for body_id in scene.object_ids:
        scene_mask = body_ids == int(body_id)
        scene_pixels = int(np.count_nonzero(scene_mask))
        if scene_pixels == 0:
            continue

        intersection = int(np.count_nonzero(reference_mask & scene_mask))
        union = int(np.count_nonzero(reference_mask | scene_mask))
        iou = float(intersection / union) if union else 0.0
        reference_coverage = float(intersection / diagnostics["mask_pixels"])
        scene_coverage = float(intersection / scene_pixels)
        scene_object = scene.get_object_info(int(body_id))
        candidates.append(
            {
                "body_id": int(body_id),
                "name": scene_object.name,
                "iou": iou,
                "intersection_pixels": intersection,
                "scene_pixels": scene_pixels,
                "reference_coverage": reference_coverage,
                "scene_coverage": scene_coverage,
            }
        )

    if not candidates:
        raise RuntimeError("No visible PyBullet objects are available for mask matching")

    candidates.sort(
        key=lambda item: (
            item["iou"],
            item["intersection_pixels"],
            -item["body_id"],
        ),
        reverse=True,
    )
    best = candidates[0]
    if best["iou"] < float(minimum_iou):
        raise RuntimeError(
            "Reason object mask could not be matched reliably to a PyBullet object: "
            f"best_iou={best['iou']:.6f}, minimum_iou={float(minimum_iou):.6f}, "
            f"best_name={best['name']!r}"
        )

    diagnostics.update(
        {
            "source": "perception_mask_iou",
            "minimum_iou": float(minimum_iou),
            "selected_body_id": int(best["body_id"]),
            "selected_object_name": best["name"],
            "selected_iou": float(best["iou"]),
            "candidates": candidates,
        }
    )
    selected_body_id = int(best["body_id"])
    return (
        selected_body_id,
        scene.get_object_info(selected_body_id),
        diagnostics,
    )
