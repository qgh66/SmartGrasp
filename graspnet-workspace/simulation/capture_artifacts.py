"""Persist synchronized RGB-D and segmentation artifacts from a camera frame."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


def _safe_file_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return sanitized or "object"


def _decode_body_ids(segmentation: np.ndarray) -> np.ndarray:
    segmentation_values = np.asarray(segmentation, dtype=np.int64)
    return np.where(
        segmentation_values >= 0,
        segmentation_values & ((1 << 24) - 1),
        -1,
    ).astype(np.int32)


def _body_color(body_id: int) -> np.ndarray:
    """Return a deterministic, bright RGB color for one PyBullet body id."""
    return np.array(
        [
            55 + (67 * (body_id + 1)) % 200,
            55 + (131 * (body_id + 1)) % 200,
            55 + (193 * (body_id + 1)) % 200,
        ],
        dtype=np.uint8,
    )


def _colorize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    requested_near: float | None = None,
    requested_far: float | None = None,
) -> tuple[np.ndarray, float, float]:
    import matplotlib

    valid_depth = depth[valid_mask]
    percentile_near, percentile_far = np.percentile(valid_depth, [1.0, 99.0])
    display_near = float(
        requested_near if requested_near is not None else percentile_near
    )
    display_far = float(
        requested_far if requested_far is not None else percentile_far
    )
    if display_near < 0.0:
        raise ValueError("Depth display near value must be non-negative")
    if display_far <= display_near:
        raise ValueError("Depth display far value must be greater than near")

    normalized = np.clip(
        (depth - display_near) / (display_far - display_near),
        0.0,
        1.0,
    )
    color_map = matplotlib.colormaps.get_cmap("turbo")
    colorized = np.asarray(color_map(normalized, bytes=True)[..., :3], dtype=np.uint8)
    colorized[~valid_mask] = 0
    return colorized, display_near, display_far


def export_camera_frame(
    *,
    output_dir: str | Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    segmentation: np.ndarray,
    object_names_by_id: Mapping[int, str],
    target_body_id: int | None = None,
    depth_display_near_m: float | None = None,
    depth_display_far_m: float | None = None,
) -> dict[str, Any]:
    """Save one synchronized camera frame and return JSON-safe artifact metadata."""
    frame_dir = Path(output_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    rgb_array = np.asarray(rgb, dtype=np.uint8)
    depth_array = np.asarray(depth, dtype=np.float32)
    segmentation_array = np.asarray(segmentation, dtype=np.int32)
    if rgb_array.ndim != 3 or rgb_array.shape[2] not in (3, 4):
        raise ValueError(f"rgb must have shape (H, W, 3|4), got {rgb_array.shape}")
    if depth_array.shape != rgb_array.shape[:2]:
        raise ValueError(
            f"depth shape {depth_array.shape} does not match rgb {rgb_array.shape[:2]}"
        )
    if segmentation_array.shape != rgb_array.shape[:2]:
        raise ValueError(
            "segmentation shape "
            f"{segmentation_array.shape} does not match rgb {rgb_array.shape[:2]}"
        )

    valid_depth_mask = np.isfinite(depth_array) & (depth_array > 0.0)
    if not np.any(valid_depth_mask):
        raise ValueError("Camera frame contains no finite positive depth pixels")

    body_ids = _decode_body_ids(segmentation_array)
    rgb_path = frame_dir / "rgb.png"
    depth_npy_path = frame_dir / "depth_m.npy"
    depth_mm_path = frame_dir / "depth_mm.png"
    depth_color_path = frame_dir / "depth_color.png"
    segmentation_npy_path = frame_dir / "segmentation.npy"
    body_ids_path = frame_dir / "body_ids_plus_one.png"
    segmentation_color_path = frame_dir / "segmentation_color.png"
    target_mask_path = frame_dir / "target_mask.png"
    masks_dir = frame_dir / "masks"
    metadata_path = frame_dir / "capture.json"

    Image.fromarray(rgb_array[..., :3], mode="RGB").save(rgb_path)
    np.save(depth_npy_path, depth_array)
    depth_mm = np.zeros(depth_array.shape, dtype=np.uint16)
    depth_mm[valid_depth_mask] = np.clip(
        np.rint(depth_array[valid_depth_mask] * 1000.0),
        1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    Image.fromarray(depth_mm).save(depth_mm_path)
    depth_color, display_near, display_far = _colorize_depth(
        depth_array,
        valid_depth_mask,
        depth_display_near_m,
        depth_display_far_m,
    )
    Image.fromarray(depth_color, mode="RGB").save(depth_color_path)

    np.save(segmentation_npy_path, segmentation_array)
    body_ids_plus_one = np.where(body_ids >= 0, body_ids + 1, 0).astype(np.uint16)
    Image.fromarray(body_ids_plus_one).save(body_ids_path)
    segmentation_color = np.zeros((*body_ids.shape, 3), dtype=np.uint8)
    for visible_body_id in np.unique(body_ids):
        if visible_body_id >= 0:
            segmentation_color[body_ids == visible_body_id] = _body_color(
                int(visible_body_id)
            )
    Image.fromarray(segmentation_color, mode="RGB").save(segmentation_color_path)

    masks_dir.mkdir(parents=True, exist_ok=True)
    object_masks: dict[str, dict[str, Any]] = {}
    for body_id, object_name in object_names_by_id.items():
        object_mask = body_ids == int(body_id)
        mask_file_name = (
            f"body_{int(body_id):03d}_{_safe_file_component(object_name)}.png"
        )
        mask_path = masks_dir / mask_file_name
        Image.fromarray((object_mask.astype(np.uint8) * 255), mode="L").save(mask_path)
        object_masks[str(int(body_id))] = {
            "object_name": str(object_name),
            "pixel_count": int(object_mask.sum()),
            "path": str(mask_path),
        }

    target_mask_pixels = None
    if target_body_id is not None:
        target_mask = body_ids == int(target_body_id)
        target_mask_pixels = int(target_mask.sum())
        Image.fromarray((target_mask.astype(np.uint8) * 255), mode="L").save(
            target_mask_path
        )

    valid_depth = depth_array[valid_depth_mask]
    metadata: dict[str, Any] = {
        "same_frame_rgb_depth_segmentation": True,
        "image_shape": [int(value) for value in rgb_array.shape[:2]],
        "depth_unit": "meter",
        "depth_min_m": float(valid_depth.min()),
        "depth_max_m": float(valid_depth.max()),
        "depth_display_near_m": display_near,
        "depth_display_far_m": display_far,
        "segmentation_encoding": (
            "segmentation.npy stores raw PyBullet values; body_ids_plus_one.png "
            "stores background=0 and body_id+1 for visible bodies"
        ),
        "target_body_id": int(target_body_id) if target_body_id is not None else None,
        "target_mask_pixels": target_mask_pixels,
        "paths": {
            "rgb_png": str(rgb_path),
            "depth_m_npy": str(depth_npy_path),
            "depth_mm_png": str(depth_mm_path),
            "depth_color_png": str(depth_color_path),
            "segmentation_npy": str(segmentation_npy_path),
            "body_ids_plus_one_png": str(body_ids_path),
            "segmentation_color_png": str(segmentation_color_path),
            "target_mask_png": (
                str(target_mask_path) if target_body_id is not None else None
            ),
            "metadata_json": str(metadata_path),
        },
        "object_masks": object_masks,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _object_names_from_result(result_data: Mapping[str, Any]) -> dict[int, str]:
    objects = (
        result_data.get("objects")
        or result_data.get("final_scene_objects")
        or {}
    )
    if not isinstance(objects, Mapping):
        return {}

    object_names: dict[int, str] = {}
    for body_id, object_data in objects.items():
        if isinstance(object_data, Mapping) and object_data.get("name"):
            object_names[int(body_id)] = str(object_data["name"])
    return object_names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RGB-D, segmentation and mask files from trusted simulation PKL data"
    )
    parser.add_argument("--viz-data", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--depth-near", type=float, default=None)
    parser.add_argument("--depth-far", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.viz_data.is_file():
        raise FileNotFoundError(f"Visualization data not found: {args.viz_data}")

    # Only load PKL files generated locally by SmartGrasp. Pickle is not safe
    # for data received from untrusted sources.
    with args.viz_data.open("rb") as viz_data_file:
        viz_data = pickle.load(viz_data_file)
    if not isinstance(viz_data, Mapping):
        raise ValueError("Visualization PKL must contain a mapping")

    result_data: Mapping[str, Any] = {}
    if args.results is not None:
        with args.results.open("r", encoding="utf-8") as result_file:
            loaded_result = json.load(result_file)
        if isinstance(loaded_result, Mapping):
            result_data = loaded_result

    object_names = _object_names_from_result(viz_data)
    if not object_names:
        object_names = _object_names_from_result(result_data)

    segmentation = viz_data.get("seg", viz_data.get("segmentation"))
    if segmentation is None:
        raise ValueError("Visualization data does not contain 'seg' or 'segmentation'")
    target_body_id = viz_data.get(
        "target_body_id",
        result_data.get("target_body_id"),
    )
    metadata = export_camera_frame(
        output_dir=args.output_dir,
        rgb=np.asarray(viz_data.get("rgb")),
        depth=np.asarray(viz_data.get("depth")),
        segmentation=np.asarray(segmentation),
        object_names_by_id=object_names,
        target_body_id=(int(target_body_id) if target_body_id is not None else None),
        depth_display_near_m=args.depth_near,
        depth_display_far_m=args.depth_far,
    )
    print(f"Camera artifacts: {args.output_dir.resolve()}")
    print(f"RGB: {Path(metadata['paths']['rgb_png']).resolve()}")
    print(f"Color depth: {Path(metadata['paths']['depth_color_png']).resolve()}")
    print(
        "Measured depth range: "
        f"{metadata['depth_min_m']:.6f}–{metadata['depth_max_m']:.6f} m"
    )
    print(f"Object masks: {len(metadata['object_masks'])}")


if __name__ == "__main__":
    main()
