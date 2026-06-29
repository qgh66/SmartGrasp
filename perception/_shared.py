"""Shared utilities: mask ops, I/O, drawing, logging — used by all perception modules."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMARTGRASP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SMARTGRASP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _safe_label(label: str, max_len: int = 200) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in label.strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    if max_len > 0 and len(normalized) > max_len:
        normalized = normalized[:max_len].rstrip("_")
    return normalized or "object"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _prepare_mask_output_dir(output_mask_dir: Path, save_candidates: bool) -> None:
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    for old_mask in output_mask_dir.glob("*.png"):
        old_mask.unlink()
    if not save_candidates:
        for candidate_dir_name in ("mask_candidates", "langsam_candidates", "sam2_auto_candidates"):
            candidate_dir = output_mask_dir.parent / candidate_dir_name
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)


def _load_depth_map(depth_path: Path) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(depth_path)
    elif suffix == ".npz":
        npz = np.load(depth_path)
        if len(npz.files) != 1:
            raise ValueError(f"Expected exactly one array inside {depth_path}, found {npz.files}.")
        depth = npz[npz.files[0]]
    else:
        depth = np.array(Image.open(depth_path))

    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D after loading, got shape {depth.shape}.")
    return depth.astype(np.float32, copy=False)


def _save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(out_path)


def _clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if kernel_size <= 1:
        return mask_bool

    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        closed = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return closed > 0

    pad = kernel_size // 2
    padded = np.pad(mask_bool, pad_width=pad, mode="constant", constant_values=False)
    closed = np.zeros_like(mask_bool, dtype=bool)
    for row_offset in range(kernel_size):
        for col_offset in range(kernel_size):
            closed |= padded[row_offset : row_offset + mask_bool.shape[0], col_offset : col_offset + mask_bool.shape[1]]

    padded = np.pad(closed, pad_width=pad, mode="constant", constant_values=True)
    eroded = np.ones_like(mask_bool, dtype=bool)
    for row_offset in range(kernel_size):
        for col_offset in range(kernel_size):
            eroded &= padded[row_offset : row_offset + mask_bool.shape[0], col_offset : col_offset + mask_bool.shape[1]]
    return eroded


def _as_numpy_mask(mask: Any) -> np.ndarray:
    import torch

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask_np = np.asarray(mask)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        raise ValueError(f"LangSAM mask must be 2D after squeezing, got {mask_np.shape}.")
    return mask_np > 0


def _normalize_box(raw_box: Any) -> list[int] | None:
    try:
        values = np.asarray(raw_box, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if values.size < 4:
        return None
    x0, y0, x1, y1 = [int(round(float(value))) for value in values[:4]]
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def _mask_centroid_xy(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0
    return int(np.round(xs.mean())), int(np.round(ys.mean()))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.count_nonzero(mask_a & mask_b))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(mask_a | mask_b))
    return float(intersection / max(1, union))


def _mask_overlap_fraction(candidate: np.ndarray, existing: np.ndarray) -> float:
    candidate_area = int(np.count_nonzero(candidate))
    if candidate_area == 0:
        return 0.0
    intersection = int(np.count_nonzero(candidate & existing))
    return float(intersection / candidate_area)


def _draw_labeled_image_matplotlib(
    image: Image.Image,
    points_with_ids: list[tuple[int, int, int]],
    out_png_path: Path | str,
) -> None:
    out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    for obj_id, x, y in points_with_ids:
        plt.text(
            x, y, str(obj_id),
            color="yellow", fontsize=8, fontweight="bold",
            ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )
    plt.axis("off")
    plt.savefig(str(out_png_path), bbox_inches="tight", dpi=300)
    plt.close()


def _draw_mask_records_label(
    image_path: Path,
    mask_records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    points_with_ids: list[tuple[int, int, int]] = []
    for record in mask_records:
        point = record.get("point", {})
        x = int(point.get("x", -1))
        y = int(point.get("y", -1))
        if x < 0 or y < 0:
            continue
        object_id = int(record.get("object_id", record.get("node_id", len(points_with_ids) + 1)))
        points_with_ids.append((object_id, x, y))
    with Image.open(image_path) as image:
        _draw_labeled_image_matplotlib(
            image=image,
            points_with_ids=points_with_ids,
            out_png_path=str(out_path),
        )


def _dilate_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if kernel_size <= 1:
        return mask_bool
    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0

    pad = kernel_size // 2
    padded = np.pad(mask_bool, pad_width=pad, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask_bool, dtype=bool)
    for row_offset in range(kernel_size):
        for col_offset in range(kernel_size):
            dilated |= padded[row_offset : row_offset + mask_bool.shape[0], col_offset : col_offset + mask_bool.shape[1]]
    return dilated


def _valid_depth_values(depth_map: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(depth_map, dtype=np.float32)[np.asarray(sample_mask, dtype=bool)]
    return values[np.isfinite(values) & (values > 0)]


def _box_xywh_to_xyxy(box: list[int]) -> list[int] | None:
    if len(box) < 4:
        return None
    x, y, width, height = [int(value) for value in box[:4]]
    if width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


_SCENE_ID: int | None = None


def set_log_scene_id(scene_id: int) -> None:
    global _SCENE_ID
    _SCENE_ID = scene_id


def _log_step(step: str, start: float | None = None) -> float:
    now = time.time()
    prefix = f"[scene_{_SCENE_ID}]" if _SCENE_ID is not None else "[perception]"
    if start is not None:
        elapsed = now - start
        print(f"{prefix} {step}  ({elapsed:.1f}s)", flush=True)
    else:
        print(f"{prefix} {step} ...", flush=True)
    return now
