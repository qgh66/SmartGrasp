"""Perception pipeline: SAM2 auto masks -> OpenAI review -> LangSAM refine -> occlusion graph JSON."""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import time
from html import unescape
from pathlib import Path
from typing import Any

SMARTGRASP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SMARTGRASP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from SmartGrasp.perception.occul_map.org import build_occlusion_graph, graph_to_jsonable

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only in cv2-less environments
    cv2 = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LANGSAM_CACHE: dict[tuple[str, str], Any] = {}
GROUNDING_DINO_LOCAL_MODEL_PATH = Path(
    "/home/data/models/huggingface/hub/models--IDEA-Research--grounding-dino-base/"
    "snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
)


def _safe_label(label: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in label.strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "object"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def _resolve_path(points_json_path: Path, candidate: str) -> Path:
    raw = Path(candidate)
    if raw.is_absolute() and raw.exists():
        return raw

    search_roots = [
        points_json_path.parent,
        points_json_path.parent.parent,
        Path.cwd(),
    ]
    for root in search_roots:
        resolved = (root / raw).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Could not resolve path {candidate!r} relative to {points_json_path}.")


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


def _load_langsam(device: str) -> Any:
    cache_key = ("default", device)
    if cache_key not in _LANGSAM_CACHE:
        try:
            from lang_sam import LangSAM  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LangSAM is not installed in the active environment. "
                "Install lang-segment-anything/LangSAM before using --segmentation-backend langsam."
            ) from exc

        if GROUNDING_DINO_LOCAL_MODEL_PATH.exists():
            try:
                model = LangSAM(
                    device=device,
                    gdino_model_ckpt_path=str(GROUNDING_DINO_LOCAL_MODEL_PATH),
                    gdino_processor_ckpt_path=str(GROUNDING_DINO_LOCAL_MODEL_PATH),
                )
            except TypeError:
                try:
                    model = LangSAM(
                        gdino_model_ckpt_path=str(GROUNDING_DINO_LOCAL_MODEL_PATH),
                        gdino_processor_ckpt_path=str(GROUNDING_DINO_LOCAL_MODEL_PATH),
                    )
                except TypeError:
                    model = LangSAM(device=device)
        else:
            try:
                model = LangSAM(device=device)
            except TypeError:
                model = LangSAM()
        _LANGSAM_CACHE[cache_key] = model
    return _LANGSAM_CACHE[cache_key]


def _as_numpy_mask(mask: Any) -> np.ndarray:
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


def _langsam_predict(model: Any, image: Image.Image, prompt: str) -> tuple[list[np.ndarray], list[float], list[list[int] | None]]:
    result = model.predict([image], [prompt])
    if isinstance(result, list):
        if not result:
            return [], [], []
        item = result[0]
        if isinstance(item, dict):
            raw_masks = item.get("masks")
            if raw_masks is None:
                raw_masks = []
            raw_scores = item.get("mask_scores")
            if raw_scores is None:
                raw_scores = item.get("scores")
            if raw_scores is None:
                raw_scores = []
            raw_boxes = item.get("boxes")
            if raw_boxes is None:
                raw_boxes = []
        else:
            raw_masks = item
            raw_scores = []
            raw_boxes = []
    elif isinstance(result, dict):
        raw_masks = result.get("masks")
        if raw_masks is None:
            raw_masks = []
        raw_scores = result.get("mask_scores")
        if raw_scores is None:
            raw_scores = result.get("scores")
        if raw_scores is None:
            raw_scores = []
        raw_boxes = result.get("boxes")
        if raw_boxes is None:
            raw_boxes = []
    else:
        raw_masks = getattr(result, "masks", [])
        raw_scores = getattr(result, "mask_scores", getattr(result, "scores", []))
        raw_boxes = getattr(result, "boxes", [])

    if isinstance(raw_masks, torch.Tensor):
        raw_masks = list(raw_masks)
    if isinstance(raw_scores, torch.Tensor):
        raw_scores = raw_scores.detach().cpu().numpy().tolist()
    elif isinstance(raw_scores, np.ndarray):
        raw_scores = np.atleast_1d(raw_scores).tolist()
    elif np.isscalar(raw_scores):
        raw_scores = [raw_scores]

    masks = [_as_numpy_mask(mask) for mask in raw_masks]
    scores = [float(score) for score in raw_scores] if raw_scores is not None else []
    if len(scores) < len(masks):
        scores.extend([0.0] * (len(masks) - len(scores)))
    if isinstance(raw_boxes, torch.Tensor):
        raw_boxes = raw_boxes.detach().cpu().numpy()
    boxes = [_normalize_box(box) for box in list(raw_boxes)] if raw_boxes is not None else []
    if len(boxes) < len(masks):
        boxes.extend([None] * (len(masks) - len(boxes)))
    return masks, scores, boxes


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


def _finalize_independent_scene_masks(
    mask_records: list[dict[str, Any]],
    output_mask_dir: Path,
    background_exclusion_mask: np.ndarray | None,
    image_shape: tuple[int, int],
    containment_threshold: float = 0.92,
    overlap_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Make final masks non-overlapping and report approximate scene coverage."""
    if not mask_records:
        return mask_records, {
            "dedup_report": [],
            "overlap_report": [],
            "coverage": {
                "foreground_coverage_ratio": 0.0,
                "scene_coverage_ratio_with_background": 0.0,
                "missing_non_background_ratio": 1.0,
            },
        }

    kept = list(mask_records)
    dedup_report: list[dict[str, Any]] = []
    masks = [np.asarray(record["mask_array"], dtype=bool).copy() for record in kept]
    areas = [int(np.count_nonzero(mask)) for mask in masks]
    explicit_scores = [
        0 if str(record.get("segmentation_backend", "")).startswith("sam2_auto_review") else 1
        for record in kept
    ]
    order = sorted(range(len(masks)), key=lambda index: (explicit_scores[index], areas[index]))
    owner = np.full(image_shape, -1, dtype=np.int32)
    overlap_report: list[dict[str, Any]] = []

    for index in order:
        mask = masks[index]
        overlap = mask & (owner >= 0)
        overlap_pixels = int(np.count_nonzero(overlap))
        if overlap_pixels > 0:
            owner_ids, counts = np.unique(owner[overlap], return_counts=True)
            overlap_report.append(
                {
                    "object_id": int(kept[index].get("object_id", index + 1)),
                    "removed_overlap_pixels": overlap_pixels,
                    "removed_overlap_fraction": float(overlap_pixels / max(1, areas[index])),
                    "overlap_with": [
                        {
                            "object_id": int(kept[int(owner_id)].get("object_id", int(owner_id) + 1)),
                            "pixels": int(count),
                        }
                        for owner_id, count in zip(owner_ids, counts)
                        if int(owner_id) >= 0
                    ],
                }
            )
            mask = mask & (owner < 0)
        masks[index] = mask
        owner[mask] = index

    finalized: list[dict[str, Any]] = []
    removed_empty: list[dict[str, Any]] = []
    image_area = max(1.0, float(image_shape[0] * image_shape[1]))
    for index, record in enumerate(kept):
        mask = masks[index]
        area = int(np.count_nonzero(mask))
        area_ratio = float(area) / image_area
        if area == 0 or area_ratio < LANGSAM_MIN_AREA_RATIO:
            removed = {key: value for key, value in record.items() if key != "mask_array"}
            removed["duplicate_reason"] = (
                "removed_after_overlap_exclusivity" if area == 0 else "removed_too_small_after_overlap"
            )
            if area_ratio < LANGSAM_MIN_AREA_RATIO and area > 0:
                removed["removed_area_ratio"] = float(area_ratio)
                removed["min_area_ratio_threshold"] = float(LANGSAM_MIN_AREA_RATIO)
            removed_empty.append(removed)
            continue
        record["mask_array"] = mask
        record["mask_area"] = area
        cx, cy = _mask_centroid_xy(mask)
        record["point"] = {"x": int(cx), "y": int(cy)}
        old_path = Path(str(record.get("mask_path", "")))
        if old_path.exists():
            _save_mask_png(mask, old_path)
        finalized.append(record)

    foreground_union = np.zeros(image_shape, dtype=bool)
    if finalized:
        foreground_union = np.any(np.stack([np.asarray(record["mask_array"], dtype=bool) for record in finalized], axis=0), axis=0)
    background = np.asarray(background_exclusion_mask, dtype=bool) if background_exclusion_mask is not None else np.zeros(image_shape, dtype=bool)
    background_only = background & ~foreground_union
    scene_union = foreground_union | background_only
    image_area = max(1, int(image_shape[0] * image_shape[1]))
    non_background = ~background
    non_background_area = max(1, int(np.count_nonzero(non_background)))
    missing_non_background = non_background & ~foreground_union
    coverage = {
        "foreground_coverage_ratio": float(np.count_nonzero(foreground_union) / image_area),
        "background_coverage_ratio": float(np.count_nonzero(background_only) / image_area),
        "scene_coverage_ratio_with_background": float(np.count_nonzero(scene_union) / image_area),
        "missing_non_background_ratio": float(np.count_nonzero(missing_non_background) / non_background_area),
        "overlap_pixels_after_finalize": 0,
        "background_mask_included_in_scene_coverage": bool(background_exclusion_mask is not None),
    }
    return finalized, {
        "dedup_report": dedup_report + removed_empty,
        "overlap_report": overlap_report,
        "coverage": coverage,
        "containment_threshold": float(containment_threshold),
        "overlap_threshold": float(overlap_threshold),
    }


def _renumber_masks(mask_records: list[dict[str, Any]], output_mask_dir: Path) -> list[dict[str, Any]]:
    """Renumber object_id/node_id sequentially (1,2,3...), rename mask files on disk, keep labels in sync."""
    renumbered: list[dict[str, Any]] = []
    for index, record in enumerate(mask_records, start=1):
        old_path = Path(str(record.get("mask_path", "")))
        label = _safe_label(str(record.get("label", f"object_{index}")))
        backend = record.get("segmentation_backend", "")
        source = "sam2" if "sam2" in backend else "object"
        new_name = f"{index:03d}_{source}_{label}.png"
        new_path = output_mask_dir / new_name

        # Remove stale files with same index but different source/label
        for pattern in (f"{index:03d}_*.png", f"mask_{index:03d}_*.png"):
            for leftover in output_mask_dir.glob(pattern):
                if leftover != new_path and leftover != old_path:
                    leftover.unlink()

        if old_path.exists() and old_path != new_path:
            old_path.replace(new_path)

        record["node_id"] = index - 1
        record["object_id"] = index
        record["point"] = {"x": int(record.get("point", {}).get("x", 0)), "y": int(record.get("point", {}).get("y", 0))}
        record["mask_path"] = str(new_path.resolve())
        renumbered.append(record)

    # Clean up orphaned files (proposals rejected by dedup, old-format files)
    kept_names = {Path(str(r["mask_path"])).name for r in renumbered}
    kept_names.add("000_background.png")
    for f in output_mask_dir.glob("*.png"):
        if f.name not in kept_names:
            f.unlink()

    return renumbered


def _border_touch_fraction(mask: np.ndarray, border_pixels: int = 3) -> float:
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    height, width = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    border[:border_pixels, :] = True
    border[max(0, height - border_pixels) :, :] = True
    border[:, :border_pixels] = True
    border[:, max(0, width - border_pixels) :] = True
    return float(np.count_nonzero(mask & border) / area)


def _proposal_color_name(image_np: np.ndarray, mask: np.ndarray) -> str:
    pixels = image_np[mask]
    if pixels.size == 0:
        return "visible"
    rgb = np.median(pixels.astype(np.float32), axis=0)
    r, g, b = [float(value) for value in rgb]
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    if max_c < 45:
        return "black"
    if min_c > 205 and (max_c - min_c) < 35:
        return "white"
    if (max_c - min_c) < 30:
        return "gray"
    if r > 1.25 * g and r > 1.25 * b:
        return "red"
    if b > 1.2 * r and b > 1.15 * g:
        return "blue"
    if g > 1.15 * r and g > 1.15 * b:
        return "green"
    if r > 130 and g > 95 and b < 120:
        return "yellow" if g > 0.65 * r else "orange"
    if r > 110 and b > 95 and g < 120:
        return "purple"
    return "colored"


def _proposal_shape_name(mask: np.ndarray) -> str:
    x, y, width, height = _mask_bbox(mask)
    if width <= 0 or height <= 0:
        return "visible"
    aspect = width / max(1, height)
    fill = int(np.count_nonzero(mask)) / max(1, width * height)
    if aspect >= 1.6:
        return "horizontal"
    if aspect <= 0.62:
        return "vertical"
    if fill >= 0.68:
        return "compact"
    if fill <= 0.38:
        return "thin"
    return "rectangular"


def _proposal_label(image_np: np.ndarray, mask: np.ndarray) -> str:
    color = _proposal_color_name(image_np, mask)
    shape = _proposal_shape_name(mask)
    parts = [part for part in (color, shape, "piece") if part and part != "visible"]
    return " ".join(parts) if parts else "visible piece"


def _proposal_score(proposal: dict[str, Any], area_ratio: float, border_fraction: float) -> float:
    predicted_iou = float(proposal.get("predicted_iou", proposal.get("iou", 0.0)) or 0.0)
    stability = float(proposal.get("stability_score", 0.0) or 0.0)
    area_prior = -abs(area_ratio - 0.035)
    border_penalty = 0.5 * border_fraction
    return predicted_iou + 0.25 * stability + area_prior - border_penalty


def _is_support_like_horizontal_strip(mask: np.ndarray) -> bool:
    height, width = mask.shape
    x, y, bbox_width, bbox_height = _mask_bbox(mask)
    if bbox_width <= 0 or bbox_height <= 0:
        return False
    aspect = bbox_width / max(1, bbox_height)
    vertical_center = (y + 0.5 * bbox_height) / max(1, height)
    horizontal_coverage = bbox_width / max(1, width)
    thin_height = bbox_height / max(1, height)
    return bool(
        aspect >= 3.0
        and vertical_center >= 0.86
        and horizontal_coverage >= 0.22
        and thin_height <= 0.08
    )


def _is_tray_or_background_like_proposal(
    image_np: np.ndarray,
    mask: np.ndarray,
    background_exclusion_mask: np.ndarray | None = None,
) -> bool:
    x, y, bbox_width, bbox_height = _mask_bbox(mask)
    if bbox_width <= 0 or bbox_height <= 0:
        return True

    # Prefer the pre-computed depth-based background exclusion mask when available.
    if background_exclusion_mask is not None and int(np.count_nonzero(background_exclusion_mask)) > 0:
        background_overlap = _background_overlap_fraction(mask, background_exclusion_mask)
        if background_overlap >= LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD:
            return True

    height, width = mask.shape
    area = int(np.count_nonzero(mask))
    bbox_area = bbox_width * bbox_height
    fill = area / max(1, bbox_area)
    aspect = bbox_width / max(1, bbox_height)
    cx = (x + 0.5 * bbox_width) / max(1, width)
    cy = (y + 0.5 * bbox_height) / max(1, height)

    pixels = image_np[mask].astype(np.float32)
    if pixels.size == 0:
        return True
    median_rgb = np.median(pixels, axis=0)
    r, g, b = [float(value) for value in median_rgb]
    greenish = g >= 85 and g >= r * 1.02 and g >= b * 0.92
    low_saturation = (max(r, g, b) - min(r, g, b)) < 35
    near_tray_wall = cx <= 0.18 or cx >= 0.82 or cy <= 0.12 or cy >= 0.88

    if greenish and (near_tray_wall or fill >= 0.78):
        return True
    if greenish and cy >= 0.68 and aspect >= 1.45 and bbox_width / max(1, width) >= 0.25:
        return True
    if low_saturation and near_tray_wall and fill >= 0.72:
        return True
    if aspect >= 3.0 and bbox_width / max(1, width) >= 0.22 and bbox_height / max(1, height) <= 0.08:
        return True
    return False


def _sam2_auto_generate(model: Any, image: Image.Image) -> list[dict[str, Any]]:
    image_np = np.asarray(image.convert("RGB"))
    sam = getattr(model, "sam", None)
    if sam is None or not hasattr(sam, "generate"):
        raise RuntimeError("Loaded LangSAM object does not expose SAM2 automatic mask generation.")
    return list(sam.generate(image_np))


def _configure_sam2_auto_generator(
    model: Any,
    points_per_side: int | None = None,
    crop_n_layers: int | None = None,
    pred_iou_thresh: float | None = None,
    stability_score_thresh: float | None = None,
) -> dict[str, Any]:
    sam = getattr(model, "sam", None)
    generator = getattr(sam, "mask_generator", None)
    sam_model = getattr(sam, "model", None)
    if generator is None or sam_model is None:
        return {"configured": False, "reason": "sam2_mask_generator_unavailable"}

    kwargs: dict[str, Any] = {}
    for name in (
        "points_per_side",
        "points_per_batch",
        "pred_iou_thresh",
        "stability_score_thresh",
        "stability_score_offset",
        "mask_threshold",
        "box_nms_thresh",
        "crop_n_layers",
        "crop_nms_thresh",
        "crop_overlap_ratio",
        "crop_n_points_downscale_factor",
        "min_mask_region_area",
        "output_mode",
    ):
        if hasattr(generator, name):
            kwargs[name] = getattr(generator, name)

    if points_per_side is not None and points_per_side > 0:
        kwargs["points_per_side"] = int(points_per_side)
        kwargs.pop("point_grids", None)
    if crop_n_layers is not None and crop_n_layers >= 0:
        kwargs["crop_n_layers"] = int(crop_n_layers)
    if pred_iou_thresh is not None:
        kwargs["pred_iou_thresh"] = float(pred_iou_thresh)
    if stability_score_thresh is not None:
        kwargs["stability_score_thresh"] = float(stability_score_thresh)

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam.mask_generator = SAM2AutomaticMaskGenerator(sam_model, **kwargs)
    except Exception as exc:
        return {"configured": False, "reason": str(exc), "requested": kwargs}
    return {"configured": True, "settings": kwargs}


DEPTH_BACKGROUND_THRESHOLD = 79.802 - 0.05
DEPTH_BACKGROUND_EXPANSION_MIN = 73.5
DEPTH_BACKGROUND_EXPANSION_ANCHOR = 78.5
BACKGROUND_EXPANSION_HUE_TOLERANCE = 14.0
BACKGROUND_EXPANSION_MIN_ANCHOR_FRACTION = 0.025
BACKGROUND_EXPANSION_MAX_HUE_MODES = 4
BACKGROUND_EXPANSION_MAX_COMPONENT_AREA_RATIO = 0.23
LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD = 0.5


def _hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    diff = np.abs(hue.astype(np.float32) - float(center))
    return np.minimum(diff, 180.0 - diff)


def _seed_hue_profiles(hsv: np.ndarray, seed_mask: np.ndarray) -> list[tuple[float, float]]:
    saturation = hsv[..., 1]
    seed_chromatic = seed_mask & (saturation >= 35)
    if int(np.count_nonzero(seed_chromatic)) < 500:
        return []

    hues = hsv[..., 0][seed_chromatic].astype(np.int32)
    hist = np.bincount(hues, minlength=180).astype(np.float32)
    hist_total = float(hist.sum())
    if hist_total <= 0:
        return []

    padded = np.concatenate([hist[-3:], hist, hist[:3]])
    smoothed = np.convolve(padded, np.ones(7, dtype=np.float32), mode="valid")
    working = smoothed.copy()
    profiles: list[tuple[float, float]] = []
    min_mode_weight = max(500.0, hist_total * 0.025)

    for _ in range(BACKGROUND_EXPANSION_MAX_HUE_MODES):
        mode_idx = int(np.argmax(working)) % 180
        if float(working[mode_idx]) < min_mode_weight:
            break

        hue_center = float(mode_idx)
        mode_mask = seed_chromatic & (_hue_distance(hsv[..., 0], hue_center) <= BACKGROUND_EXPANSION_HUE_TOLERANCE)
        if int(np.count_nonzero(mode_mask)) >= 100:
            saturation_floor = float(max(25.0, np.percentile(saturation[mode_mask], 10) * 0.45))
            profiles.append((hue_center, saturation_floor))

        suppress = _hue_distance(np.arange(180, dtype=np.float32), hue_center) <= BACKGROUND_EXPANSION_HUE_TOLERANCE
        working[suppress] = 0.0

    return profiles


def _expand_background_from_seed(
    seed_mask: np.ndarray,
    image: Image.Image | None,
    depth_map: np.ndarray,
    mask_clean_kernel: int,
) -> np.ndarray:
    if image is None or cv2 is None or int(np.count_nonzero(seed_mask)) == 0:
        return seed_mask

    image_np = np.asarray(image.convert("RGB"))
    if image_np.shape[:2] != seed_mask.shape:
        return seed_mask

    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hue_profiles = _seed_hue_profiles(hsv, seed_mask)
    if not hue_profiles:
        return seed_mask

    depth = np.asarray(depth_map, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0)
    chromatic_candidate = np.zeros(seed_mask.shape, dtype=bool)
    for hue_center, saturation_floor in hue_profiles:
        chromatic_candidate |= (
            (_hue_distance(hsv[..., 0], hue_center) <= BACKGROUND_EXPANSION_HUE_TOLERANCE)
            & (hsv[..., 1].astype(np.float32) >= saturation_floor)
        )
    candidate = (
        valid_depth
        & (depth >= DEPTH_BACKGROUND_EXPANSION_MIN)
        & chromatic_candidate
    )

    component_source = candidate & ~seed_mask
    if int(np.count_nonzero(component_source)) == 0:
        return seed_mask

    seed_kernel = np.ones((9, 9), dtype=np.uint8)
    seed_neighborhood = cv2.dilate(seed_mask.astype(np.uint8), seed_kernel, iterations=1) > 0
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        component_source.astype(np.uint8),
        connectivity=8,
    )

    expanded = seed_mask.copy()
    image_area = float(seed_mask.shape[0] * seed_mask.shape[1])
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < 64 or area / image_area > BACKGROUND_EXPANSION_MAX_COMPONENT_AREA_RATIO:
            continue

        component = labels == label_idx
        if not np.any(component & seed_neighborhood):
            continue

        component_depth = depth[component]
        anchor_fraction = float(np.mean(component_depth >= DEPTH_BACKGROUND_EXPANSION_ANCHOR))
        if anchor_fraction < BACKGROUND_EXPANSION_MIN_ANCHOR_FRACTION:
            continue

        expanded |= component

    return _clean_mask(expanded, mask_clean_kernel)


def _generate_background_exclusion_mask(
    depth_map: np.ndarray,
    image: Image.Image | None = None,
    mask_clean_kernel: int = 3,
) -> np.ndarray:
    """Use the tray-bottom depth plane as seed, then add adjacent tray/background edges."""
    depth = np.asarray(depth_map, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0)
    background = valid_depth & (depth >= DEPTH_BACKGROUND_THRESHOLD)
    background = _clean_mask(background, mask_clean_kernel)
    return _expand_background_from_seed(background, image, depth, mask_clean_kernel)


def _background_overlap_fraction(mask: np.ndarray, background_mask: np.ndarray | None) -> float:
    if background_mask is None or int(np.count_nonzero(background_mask)) == 0:
        return 0.0
    mask_bool = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask_bool))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(mask_bool & np.asarray(background_mask, dtype=bool)) / area)


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


def _mask_boundary_depth_delta(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    depth_map: np.ndarray,
    boundary_kernel_size: int = 3,
) -> tuple[float | None, int]:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    dilated_a = _dilate_mask(a, boundary_kernel_size)
    dilated_b = _dilate_mask(b, boundary_kernel_size)

    # The boundary samples are the pixels from each mask that touch the other
    # mask after dilation. Comparing medians on those two thin strips estimates
    # the physical depth step across the visual seam, while avoiding unrelated
    # depth values from the rest of either object.
    boundary_a = a & dilated_b
    boundary_b = b & dilated_a
    contact_pixels = int(np.count_nonzero(boundary_a) + np.count_nonzero(boundary_b))
    if contact_pixels == 0:
        return None, 0

    depth_a = _valid_depth_values(depth_map, boundary_a)
    depth_b = _valid_depth_values(depth_map, boundary_b)
    if depth_a.size == 0 or depth_b.size == 0:
        return None, contact_pixels

    return float(abs(np.median(depth_a) - np.median(depth_b))), contact_pixels


def _box_xywh_to_xyxy(box: list[int]) -> list[int] | None:
    if len(box) < 4:
        return None
    x, y, width, height = [int(value) for value in box[:4]]
    if width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"mask"}
    }


def _sam2_auto_candidate_pool(
    image_path: Path,
    output_mask_dir: Path,
    min_area_ratio: float,
    max_area_ratio: float,
    border_fraction_threshold: float,
    mask_clean_kernel: int,
    save_candidates: bool,
    device: str | None,
    background_exclusion_mask: np.ndarray | None,
    points_per_side: int | None = None,
    crop_n_layers: int | None = None,
    pred_iou_thresh: float | None = None,
    stability_score_thresh: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, Image.Image]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    image_area = float(image_np.shape[0] * image_np.shape[1])
    model = _load_langsam(device)
    generator_settings = _configure_sam2_auto_generator(
        model,
        points_per_side=points_per_side,
        crop_n_layers=crop_n_layers,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    raw_proposals = _sam2_auto_generate(model, image)
    candidates: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = [{"stage": "sam2_auto_generator", **generator_settings}]

    effective_min_area_ratio = max(0.0002, min_area_ratio * 0.15)
    effective_max_area_ratio = max(max_area_ratio, 0.45)

    for idx, proposal in enumerate(raw_proposals):
        raw_mask = proposal.get("segmentation")
        if raw_mask is None:
            continue
        mask = _clean_mask(_as_numpy_mask(raw_mask), mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        area_ratio = float(area / image_area)
        border_fraction = _border_touch_fraction(mask)
        background_overlap = _background_overlap_fraction(mask, background_exclusion_mask)
        bbox_xywh = [int(value) for value in proposal.get("bbox", _mask_bbox(mask))]
        bbox_xyxy = _box_xywh_to_xyxy(bbox_xywh)
        cx, cy = _mask_centroid_xy(mask)

        reason: str | None = None
        if area_ratio < effective_min_area_ratio:
            reason = "too_small"
        elif area_ratio > effective_max_area_ratio:
            reason = "too_large"
        elif border_fraction > border_fraction_threshold:
            reason = "touches_image_border"
        elif _is_support_like_horizontal_strip(mask) or _is_tray_or_background_like_proposal(image_np, mask, background_exclusion_mask):
            reason = "support_or_tray_like"
        elif background_overlap > 0.5:
            reason = "overlaps_background_exclusion"

        metadata = {
            "proposal_index": int(idx),
            "area": area,
            "area_ratio": area_ratio,
            "bbox": bbox_xywh,
            "bbox_xyxy": bbox_xyxy,
            "predicted_iou": float(proposal.get("predicted_iou", 0.0) or 0.0),
            "stability_score": float(proposal.get("stability_score", 0.0) or 0.0),
            "border_fraction": border_fraction,
            "background_exclusion_overlap": background_overlap,
            "centroid": {"x": int(cx), "y": int(cy)},
        }
        if reason is not None:
            metadata["rejection_reason"] = reason
            report.append(metadata)
            continue

        metadata["selection_score"] = _proposal_score(proposal, area_ratio, border_fraction)
        metadata["mask"] = mask
        candidates.append(metadata)
        report.append({key: value for key, value in metadata.items() if key != "mask"})

    candidates.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
    if save_candidates:
        candidate_dir = output_mask_dir.parent / "sam2_auto_candidates"
        for candidate in candidates:
            candidate_path = candidate_dir / f"proposal_{int(candidate['proposal_index']):03d}.png"
            _save_mask_png(candidate["mask"], candidate_path)
            candidate["mask_path"] = str(candidate_path.resolve())
    return candidates, report, model, image


def _overlay_mask_contours(ax: Any, masks: list[np.ndarray]) -> None:
    if cv2 is None:
        return
    for mask in masks:
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        contours, _hierarchy = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            contour_xy = contour[:, 0, :]
            ax.plot(contour_xy[:, 0], contour_xy[:, 1], color="#00ffff", linewidth=1.0, alpha=0.85)


def _draw_sam2_auto_label_image(
    image_path: Path,
    candidates: list[dict[str, Any]],
    out_path: Path,
    max_labels: int = 80,
) -> None:
    points_with_ids: list[tuple[int, int, int]] = []
    masks: list[np.ndarray] = []
    for index, candidate in enumerate(candidates[:max_labels], start=1):
        centroid = candidate.get("centroid", {})
        x = int(centroid.get("x", 0))
        y = int(centroid.get("y", 0))
        points_with_ids.append((index, x, y))
        masks.append(np.asarray(candidate["mask"], dtype=bool))

    with Image.open(image_path).convert("RGB") as image:
        _draw_labeled_image_matplotlib(image=image, points_with_ids=points_with_ids, out_png_path=out_path)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            _overlay_mask_contours(ax, masks)
            for obj_id, x, y in points_with_ids:
                ax.text(
                    x,
                    y,
                    str(obj_id),
                    color="yellow",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                )
            ax.axis("off")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
        except Exception:
            return


def _save_sam2_rgb_parts_sheet(
    image_path: Path,
    candidates: list[dict[str, Any]],
    out_dir: Path,
    max_labels: int = 40,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    parts_dir = out_dir / "sam2_rgb_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_images: list[tuple[int, Image.Image]] = []

    for index, candidate in enumerate(candidates[:max_labels], start=1):
        mask = np.asarray(candidate["mask"], dtype=bool)
        x, y, width, height = _mask_bbox(mask)
        if width <= 0 or height <= 0:
            continue
        pad = max(8, int(round(max(width, height) * 0.08)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image_np.shape[1], x + width + pad)
        y1 = min(image_np.shape[0], y + height + pad)
        crop = image_np[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]
        white = np.full_like(crop, 255)
        visible = np.where(crop_mask[..., None], crop, white)
        part_image = Image.fromarray(visible, mode="RGB")
        part_path = parts_dir / f"part_{index:03d}.png"
        part_image.save(part_path)
        part_images.append((index, part_image))

    if not part_images:
        sheet_path = out_dir / "sam2_rgb_parts_sheet.png"
        Image.new("RGB", (256, 256), "white").save(sheet_path)
        return sheet_path

    thumb_size = 160
    label_height = 26
    columns = 5
    rows = int(np.ceil(len(part_images) / columns))
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "white")
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 2.3))
        axes_arr = np.asarray(axes).reshape(rows, columns)
        for ax in axes_arr.flat:
            ax.axis("off")
        for item_index, (part_id, part_image) in enumerate(part_images):
            row = item_index // columns
            col = item_index % columns
            ax = axes_arr[row, col]
            ax.imshow(part_image)
            ax.set_title(str(part_id), fontsize=10, fontweight="bold")
            ax.axis("off")
        sheet_path = out_dir / "sam2_rgb_parts_sheet.png"
        fig.savefig(sheet_path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        return sheet_path
    except Exception:
        for item_index, (part_id, part_image) in enumerate(part_images):
            row = item_index // columns
            col = item_index % columns
            thumb = part_image.copy()
            thumb.thumbnail((thumb_size, thumb_size))
            x = col * thumb_size + (thumb_size - thumb.width) // 2
            y = row * (thumb_size + label_height) + label_height
            sheet.paste(thumb, (x, y))
        sheet_path = out_dir / "sam2_rgb_parts_sheet.png"
        sheet.save(sheet_path)
        return sheet_path


def _extract_json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            line for line in stripped.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def _image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    media_type = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif suffix == ".webp":
        media_type = "image/webp"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{data}"


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def _openai_client(api_key_env: str, base_url: str | None, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The OpenAI Python package is not installed in the smartgrasp environment.") from exc

    client_kwargs: dict[str, Any] = {"timeout": timeout}
    api_key = os.environ.get(api_key_env)
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _normalize_scene_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("Scene inventory response must contain an `objects` list.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_objects, start=1):
        if not isinstance(item, dict):
            continue
        description = unescape(str(item.get("description") or item.get("label") or "")).strip()
        if not description:
            continue
        visible_parts = item.get("visible_parts", [])
        if not isinstance(visible_parts, list):
            visible_parts = [visible_parts]
        relative_position = unescape(str(item.get("relative_position") or item.get("position") or "")).strip()
        normalized.append(
            {
                "id": int(item.get("id") or index),
                "description": description,
                "relative_position": relative_position,
                "visible_parts": [
                    unescape(str(part)).strip()
                    for part in visible_parts
                    if unescape(str(part)).strip()
                ],
            }
        )
    if not normalized:
        raise ValueError("Scene inventory response returned no valid objects.")
    return normalized


def _openai_list_scene_objects(
    image_path: Path,
    model_id: str,
    api_key_env: str,
    base_url: str | None,
    timeout: float,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    prompt = (
        "List every visible graspable physical object in this scene. "
        "Ignore the tray, table, bin, support surface, background, shadows, and reflections. "
        "Scan the image region by region from top-left to bottom-right before answering. "
        "Include small objects such as loose screws, bolts, nuts, washers, clips, caps, pins, and tiny blue or metallic pieces even if they partly touch or overlap a larger object. "
        "Treat a single tool or package as one object even when it has multiple colors or materials. "
        "For example, pliers with red/yellow handles and black jaws are one object, not separate red, yellow, and black objects. "
        "Use explicit relative position words for every object, especially repeated similar objects. "
        "For each object, describe the complete object and list its visible parts or colors when useful for later mask merging. "
        "If a small blue bolt/screw-like object appears in the lower-left area, list it as its own object instead of merging it into a nearby yellow or blue tool. "
        "Pay special attention to the lower-left cluster: a blue vertical bolt/screw-like piece below the pliers should be a separate object from any yellow propeller-shaped knob or blue-yellow tool above it. "
        "Return only JSON with this schema: "
        "{\"objects\":[{\"id\":1,\"description\":\"red and yellow handled pliers with black jaws on the right\","
        "\"relative_position\":\"lower right\",\"visible_parts\":[\"red handle\",\"yellow handle\",\"black jaws\"]}]}."
    )

    client = _openai_client(api_key_env=api_key_env, base_url=base_url, timeout=timeout)
    response = client.responses.create(
        model=model_id,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _image_data_url(image_path)},
                ],
            }
        ],
        max_output_tokens=1600,
        reasoning={"effort": "medium"},
        store=False,
    )
    raw_output = _response_text(response)
    (out_dir / "openai_scene_objects_raw.txt").write_text(raw_output, encoding="utf-8")
    scene_objects = _normalize_scene_objects(_extract_json_from_text(raw_output))
    scene_payload = {
        "model_id": model_id,
        "review_backend": "openai_responses",
        "image": {"path": str(image_path.resolve())},
        "raw_model_output": raw_output,
        "objects": scene_objects,
    }
    (out_dir / "openai_scene_objects.json").write_text(json.dumps(scene_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return scene_objects, raw_output


def _openai_review_sam2_candidates(
    image_path: Path,
    label_image_path: Path,
    parts_sheet_path: Path,
    candidates: list[dict[str, Any]],
    model_id: str,
    api_key_env: str,
    base_url: str | None,
    timeout: float,
    out_dir: Path,
    max_labels: int = 40,
) -> tuple[list[dict[str, Any]], str]:
    t_r0 = time.time()
    candidate_lines: list[str] = []
    for index, candidate in enumerate(candidates[:max_labels], start=1):
        bbox = candidate.get("bbox", [])
        area_ratio = float(candidate.get("area_ratio", 0.0))
        candidate_lines.append(f"{index}: bbox={bbox}, area_ratio={area_ratio:.5f}")

    scene_objects, scene_raw_output = _openai_list_scene_objects(
        image_path=image_path,
        model_id=model_id,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout=timeout,
        out_dir=out_dir,
    )
    _log_step("    ②b1 api_scene_objects (1img)", t_r0)
    scene_lines: list[str] = []
    for obj in scene_objects:
        parts = obj.get("visible_parts", [])
        parts_text = f"; visible_parts={parts}" if parts else ""
        position = obj.get("relative_position") or "unspecified position"
        scene_lines.append(f"{int(obj['id'])}: {obj['description']} at {position}{parts_text}")

    prompt = (
        "You are assigning automatic SAM2 mask parts to a known scene object list. "
        "First use the original scene image to understand complete physical objects, then use the numbered scene overlay and the contact sheet of numbered RGB cutouts to choose the SAM2 parts for each object. "
        "Ignore the green tray/box, table, bin, background, shadows, reflections, and support surfaces. "
        "Important: color alone is not a valid reason to merge parts. "
        "Objects can have multiple colors or materials, such as pliers with red/yellow handles and black jaws; include all parts that belong to that one complete physical object. "
        "Also do not merge two separate objects just because their parts share the same color, material, category, or shape. "
        "For each scene object, output one record with all corresponding SAM2 ids in `sam2_ids`. "
        "Every object from the known scene object list must appear in the output exactly once; do not drop small bolts/screws just because they are close to a larger tool. "
        "In the lower-left cluster, SAM2 ids 20, 21, and 24 often correspond to a blue bolt/screw-like object. Evaluate those ids as a separate object before assigning them to any blue-yellow tool or yellow knob. "
        "If ids 20/21/24 form a separate blue vertical piece, output a separate object for it and do not include those ids in the nearby larger tool. "
        "If SAM2 missed part of that object, keep the whole-object description and set status to `incomplete`. "
        "If a listed scene object has no usable SAM2 part, include it with an empty `sam2_ids` list and status `missing`. "
        "Return only JSON with this schema: "
        "{\"objects\":[{\"id\":1,\"scene_object_id\":1,\"description\":\"red and yellow handled pliers with black jaws on the right\","
        "\"sam2_ids\":[3,7,12],\"status\":\"complete|incomplete|missing\"}]}. "
        "The `description` must describe the final complete object mask, not just one color part. "
        "Known scene objects:\n"
        + "\n".join(scene_lines)
        + "\nAvailable SAM2 mask ids:\n"
        + "\n".join(candidate_lines)
        + "\nImage order: original scene, numbered SAM2 overlay, numbered RGB cutout sheet."
    )

    client = _openai_client(api_key_env=api_key_env, base_url=base_url, timeout=timeout)
    response = client.responses.create(
        model=model_id,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _image_data_url(image_path)},
                    {"type": "input_image", "image_url": _image_data_url(label_image_path)},
                    {"type": "input_image", "image_url": _image_data_url(parts_sheet_path)},
                ],
            }
        ],
        max_output_tokens=2200,
        reasoning={"effort": "medium"},
        store=False,
    )
    raw_output = _response_text(response)
    (out_dir / "openai_sam2_review_raw.txt").write_text(raw_output, encoding="utf-8")
    _log_step("    ②b2 api_sam2_review (3img)", t_r0)

    payload = _extract_json_from_text(raw_output)
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Object SAM2 review response must contain an `objects` list.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(objects, start=1):
        if not isinstance(item, dict):
            continue
        description = unescape(str(item.get("description") or item.get("label") or "")).strip()
        if not description:
            continue
        try:
            scene_object_id = int(item.get("scene_object_id") or item.get("object_id") or item.get("id") or index)
        except Exception:
            scene_object_id = index
        raw_ids = item.get("sam2_ids", [])
        if raw_ids is None:
            raw_ids = []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        sam2_ids: list[int] = []
        for raw_id in raw_ids:
            try:
                sam2_id = int(raw_id)
            except Exception:
                continue
            if 1 <= sam2_id <= min(len(candidates), max_labels):
                sam2_ids.append(sam2_id)
        normalized.append(
            {
                "id": int(item.get("id") or index),
                "scene_object_id": scene_object_id,
                "description": description,
                "sam2_ids": sorted(set(sam2_ids)),
                "status": str(item.get("status") or "incomplete"),
            }
        )
    if not normalized:
        raise ValueError("Object SAM2 review returned no valid objects.")

    review_payload = {
        "model_id": model_id,
        "review_backend": "openai_responses",
        "image": {
            "path": str(image_path.resolve()),
            "sam2_label_path": str(label_image_path.resolve()),
            "sam2_rgb_parts_sheet_path": str(parts_sheet_path.resolve()),
        },
        "scene_objects_raw_model_output": scene_raw_output,
        "scene_objects": scene_objects,
        "raw_model_output": raw_output,
        "objects": normalized,
    }
    (out_dir / "sam2_review.json").write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "openai_sam2_review.json").write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized, raw_output


LANGSAM_MIN_AREA_RATIO = 0.0007  # Minimum fraction of image area for a valid LangSAM mask (0.07%)


def _select_langsam_mask_for_review_object(
    masks: list[np.ndarray],
    scores: list[float],
    review_object: dict[str, Any],
    candidates: list[dict[str, Any]],
    background_exclusion_mask: np.ndarray | None,
    mask_clean_kernel: int,
    image_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    anchor_masks: list[np.ndarray] = []
    for sam2_id in review_object.get("sam2_ids", []):
        candidate_index = int(sam2_id) - 1
        if 0 <= candidate_index < len(candidates):
            anchor_masks.append(np.asarray(candidates[candidate_index]["mask"], dtype=bool))
    anchor_mask = np.any(np.stack(anchor_masks, axis=0), axis=0) if anchor_masks else None

    candidate_records: list[dict[str, Any]] = []
    for index, raw_mask in enumerate(masks):
        mask = _clean_mask(raw_mask, mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        background_overlap = _background_overlap_fraction(mask, background_exclusion_mask)
        anchor_covered = _mask_overlap_fraction(anchor_mask, mask) if anchor_mask is not None else 0.0
        semantic_inside_anchor = _mask_overlap_fraction(mask, anchor_mask) if anchor_mask is not None else 0.0
        selection_score = (
            3.0 * score
            + 2.0 * anchor_covered
            + 0.5 * semantic_inside_anchor
            - 2.0 * background_overlap
        )
        accepted = background_overlap < LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD
        if anchor_mask is not None:
            accepted = accepted and (anchor_covered >= 0.35 or semantic_inside_anchor >= 0.25)
        candidate_records.append(
            {
                "candidate_index": int(index),
                "mask": mask,
                "area": area,
                "semantic_score": score,
                "background_exclusion_overlap": float(background_overlap),
                "anchor_covered": float(anchor_covered),
                "semantic_inside_anchor": float(semantic_inside_anchor),
                "accepted": bool(accepted),
                "selection_score": float(selection_score),
            }
        )

    metadata = [{key: value for key, value in candidate.items() if key != "mask"} for candidate in candidate_records]
    accepted_candidates = [candidate for candidate in candidate_records if candidate["accepted"]]
    if not accepted_candidates:
        return anchor_mask, {"fallback": "sam2_anchor_union" if anchor_mask is not None else "no_mask"}, metadata
    best = max(accepted_candidates, key=lambda item: item["selection_score"])
    best_mask = np.asarray(best["mask"], dtype=bool)
    selected = {key: value for key, value in best.items() if key != "mask"}

    # If the best LangSAM mask is too small, fallback to the SAM2 anchor union.
    image_area = max(1, float(image_shape[0] * image_shape[1])) if image_shape else None
    if image_area is not None:
        langsam_area_ratio = float(int(np.count_nonzero(best_mask))) / image_area
        if langsam_area_ratio < LANGSAM_MIN_AREA_RATIO:
            if anchor_mask is not None and int(np.count_nonzero(anchor_mask)) > 0:
                selected["fallback"] = "langsam_too_small_keep_sam2_anchor"
                selected["langsam_area_ratio"] = float(langsam_area_ratio)
                selected["min_area_ratio_threshold"] = float(LANGSAM_MIN_AREA_RATIO)
                return anchor_mask, selected, metadata
            return None, {"fallback": "langsam_too_small_no_anchor"}, metadata

    if anchor_mask is not None:
        anchor_area = max(1, int(np.count_nonzero(anchor_mask)))
        best_area = max(1, int(np.count_nonzero(best_mask)))
        anchor_covered = _mask_overlap_fraction(anchor_mask, best_mask)
        best_inside_anchor = _mask_overlap_fraction(best_mask, anchor_mask)
        area_ratio = float(best_area / anchor_area)
        if anchor_covered < 0.25 and best_inside_anchor < 0.15:
            selected["fallback"] = "langsam_not_aligned_keep_sam2_anchor"
            return anchor_mask, selected, metadata
        if area_ratio > 3.0 and best_inside_anchor < 0.35:
            selected["fallback"] = "langsam_too_large_keep_sam2_anchor"
            return anchor_mask, selected, metadata
        merged = anchor_mask | best_mask
        selected["merge_policy"] = "sam2_anchor_union_langsam"
        selected["anchor_area"] = int(anchor_area)
        selected["langsam_area"] = int(best_area)
        selected["merged_area"] = int(np.count_nonzero(merged))
        return merged, selected, metadata
    return best_mask, selected, metadata


UNCLAIMED_MIN_AREA_RATIO = 0.0007  # Minimum fraction of image area for an unclaimed SAM2 candidate (0.07%)


def _append_unclaimed_sam2_candidates(
    mask_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    claimed_sam2_ids: set[int],
    image_np: np.ndarray,
    output_mask_dir: Path,
    max_unclaimed: int,
    depth_map: np.ndarray | None = None,
    depth_threshold: float = 0.012,
    min_contact_pixels: int = 4,
    max_area_ratio: float = 0.18,
    min_area_ratio: float = 0.0007,
    iou_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    if max_unclaimed <= 0:
        return mask_records

    image_area = float(image_np.shape[0] * image_np.shape[1])
    existing_masks = [np.asarray(record["mask_array"], dtype=bool) for record in mask_records]
    seed_items: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        if candidate_index in claimed_sam2_ids:
            continue
        mask = np.asarray(candidate["mask"], dtype=bool)
        if any(_mask_iou(mask, existing) > iou_threshold for existing in existing_masks):
            continue
        seed_items.append({"candidate_index": candidate_index, "candidate": candidate, "mask": mask})

    if not seed_items:
        return mask_records

    parent = list(range(len(seed_items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    dilated = [_dilate_mask(item["mask"], 3) for item in seed_items]
    for left in range(len(seed_items)):
        for right in range(left + 1, len(seed_items)):
            left_mask = seed_items[left]["mask"]
            right_mask = seed_items[right]["mask"]
            if not (np.any(dilated[left] & right_mask) or np.any(dilated[right] & left_mask)):
                continue
            should_merge = True
            if depth_map is not None:
                depth_delta, contact_pixels = _mask_boundary_depth_delta(
                    left_mask,
                    right_mask,
                    depth_map,
                    boundary_kernel_size=3,
                )
                should_merge = (
                    depth_delta is not None
                    and contact_pixels >= min_contact_pixels
                    and depth_delta <= depth_threshold
                )
            if should_merge:
                union(left, right)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, item in enumerate(seed_items):
        groups.setdefault(find(index), []).append(item)

    grouped_items: list[dict[str, Any]] = []
    for members in groups.values():
        merged_mask = np.any(np.stack([member["mask"] for member in members], axis=0), axis=0)
        area = int(np.count_nonzero(merged_mask))
        area_ratio = float(area / max(1.0, image_area))
        if area_ratio > max_area_ratio or area_ratio < min_area_ratio:
            continue
        best_member = max(members, key=lambda item: float(item["candidate"].get("selection_score", 0.0)))
        grouped_items.append(
            {
                "mask": merged_mask,
                "area": area,
                "area_ratio": area_ratio,
                "candidate_indices": [int(member["candidate_index"]) for member in members],
                "members": [_candidate_summary(member["candidate"]) for member in members],
                "selection_score": max(float(member["candidate"].get("selection_score", 0.0)) for member in members),
                "best_candidate": best_member["candidate"],
            }
        )

    grouped_items.sort(key=lambda item: (int(item["area"]), float(item["selection_score"])), reverse=True)

    appended = 0
    next_id = max([int(record.get("object_id", 0)) for record in mask_records] + [0]) + 1
    for item in grouped_items:
        mask = np.asarray(item["mask"], dtype=bool)
        if any(_mask_iou(mask, existing) > iou_threshold for existing in existing_masks):
            continue
        label = _proposal_label(image_np, mask)
        cx, cy = _mask_centroid_xy(mask)
        object_id = next_id + appended
        proposal_name = "_".join(str(index) for index in item["candidate_indices"][:8])
        filename = f"{object_id:03d}_sam2_group_{proposal_name}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(mask, mask_path)
        mask_records.append(
            {
                "node_id": len(mask_records),
                "object_id": object_id,
                "label": f"sam2 group {proposal_name}",
                "description": label,
                "point": {"x": int(cx), "y": int(cy)},
                "segmentation_backend": "sam2_auto_unclaimed_depth_grouped",
                "sam2_ids": item["candidate_indices"],
                "selected_sam2_candidate": _candidate_summary(item["best_candidate"]),
                "sam2_group_members": item["members"],
                "sam2_group_area_ratio": float(item["area_ratio"]),
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(mask)),
                "mask_array": mask,
            }
        )
        existing_masks.append(mask)
        appended += 1
        if appended >= max_unclaimed:
            break
    return mask_records


def generate_masks_with_sam2_langsam_pipeline(
    image_path: Path,
    output_mask_dir: Path,
    review_model_id: str,
    review_api_key_env: str,
    review_base_url: str | None,
    review_timeout: float,
    min_area_ratio: float,
    max_area_ratio: float,
    border_fraction_threshold: float,
    mask_clean_kernel: int,
    save_candidates: bool,
    device: str | None,
    background_exclusion_mask: np.ndarray | None,
    depth_map: np.ndarray | None = None,
    sam2_points_per_side: int | None = None,
    sam2_crop_n_layers: int | None = None,
    sam2_pred_iou_thresh: float | None = None,
    sam2_stability_score_thresh: float | None = None,
    preserve_unclaimed_sam2: int = 24,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    t_p0 = _log_step("  ②a sam2_auto", None)

    candidates, sam2_report, model, image = _sam2_auto_candidate_pool(
        image_path=image_path,
        output_mask_dir=output_mask_dir,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        border_fraction_threshold=border_fraction_threshold,
        mask_clean_kernel=mask_clean_kernel,
        save_candidates=save_candidates,
        device=device,
        background_exclusion_mask=background_exclusion_mask,
        points_per_side=sam2_points_per_side,
        crop_n_layers=sam2_crop_n_layers,
        pred_iou_thresh=sam2_pred_iou_thresh,
        stability_score_thresh=sam2_stability_score_thresh,
    )
    t_p1 = _log_step("  ②a sam2_auto", t_p0)
    out_dir = output_mask_dir.parent
    label_path = out_dir / "label_1_sam2auto.png"
    _draw_sam2_auto_label_image(image_path, candidates, label_path)
    parts_sheet_path = _save_sam2_rgb_parts_sheet(image_path, candidates, out_dir)
    try:
        review_objects, raw_review = _openai_review_sam2_candidates(
            image_path=image_path,
            label_image_path=label_path,
            parts_sheet_path=parts_sheet_path,
            candidates=candidates,
            model_id=review_model_id,
            api_key_env=review_api_key_env,
            base_url=review_base_url,
            timeout=review_timeout,
            out_dir=out_dir,
        )
    except Exception as exc:
        _log_step("  ②b openai_review FAILED", t_p1)
        raise RuntimeError(f"OpenAI review failed — API error. Aborting. Error: {exc}") from exc

    if not review_objects:
        _log_step("  ②b openai_review returned empty", t_p1)
        raise RuntimeError("OpenAI review returned no objects — empty response. Aborting.")

    mask_records: list[dict[str, Any]] = []
    claimed_sam2_ids: set[int] = set()
    with Image.open(image_path).convert("RGB") as source_image:
        image_np = np.asarray(source_image)
        for item in review_objects:
            claimed_sam2_ids.update(int(value) for value in item.get("sam2_ids", []) if int(value) > 0)
            object_id = int(item["id"])
            description = str(item["description"])
            prompt = (
                f"{description}. Segment the complete physical object instance. "
                "Include all visible parts of this one object. Do not include the tray, table, bin, or adjacent objects."
            )
            masks, scores, boxes = _langsam_predict(model, source_image, prompt)
            best_mask, selected_candidate, candidate_metadata = _select_langsam_mask_for_review_object(
                masks=masks,
                scores=scores,
                review_object=item,
                candidates=candidates,
                background_exclusion_mask=background_exclusion_mask,
                mask_clean_kernel=mask_clean_kernel,
                image_shape=source_image.size[::-1],  # (H, W) from PIL (W, H)
            )
            if best_mask is None:
                continue
            best_mask = _clean_mask(best_mask, mask_clean_kernel)
            if int(np.count_nonzero(best_mask)) == 0:
                continue
            cx, cy = _mask_centroid_xy(best_mask)
            filename = f"{object_id:03d}_langsam_{_safe_label(description)}.png"
            mask_path = output_mask_dir / filename
            _save_mask_png(best_mask, mask_path)
            mask_records.append(
                {
                    "node_id": len(mask_records),
                    "object_id": object_id,
                    "label": description,
                    "description": description,
                    "point": {"x": int(cx), "y": int(cy)},
                    "segmentation_backend": "sam2_auto_review_langsam",
                    "sam2_ids": item.get("sam2_ids", []),
                    "review_status": item.get("status"),
                    "semantic_prompt": prompt,
                    "selected_langsam_candidate": selected_candidate,
                    "langsam_candidates": candidate_metadata,
                    "mask_path": str(mask_path.resolve()),
                    "mask_area": int(np.count_nonzero(best_mask)),
                    "mask_array": best_mask,
                }
            )

        mask_records = _append_unclaimed_sam2_candidates(
            mask_records=mask_records,
            candidates=candidates,
            claimed_sam2_ids=claimed_sam2_ids,
            image_np=image_np,
            output_mask_dir=output_mask_dir,
            max_unclaimed=preserve_unclaimed_sam2,
            depth_map=depth_map,
        )

    report = [
        {
            "stage": "sam2_auto_initial",
            "label_png": str(label_path.resolve()),
            "rgb_parts_sheet_png": str(parts_sheet_path.resolve()),
            "candidates": sam2_report[:200],
        },
        {"stage": "openai_sam2_review", "objects": review_objects, "raw_model_output": raw_review, "error": None},
        {"stage": "sam2_unclaimed_preservation", "max_unclaimed": int(preserve_unclaimed_sam2), "claimed_sam2_ids": sorted(claimed_sam2_ids)},
    ]
    _log_step("  ②c langsam_refine + unclaimed", t_p1)
    return mask_records, report

def _log_step(step: str, start: float | None = None) -> float:
    """Log a pipeline step with elapsed time. Returns current time."""
    now = time.time()
    if start is not None:
        elapsed = now - start
        print(f"[perception]  {step}  ({elapsed:.1f}s)", file=sys.stderr, flush=True)
    else:
        print(f"[perception]  {step} ...", file=sys.stderr, flush=True)
    return now


def build_org_json(
    image_path: Path,
    depth_path: Path,
    output_json_path: Path,
    output_mask_dir: Path,
    review_model_id: str = "gpt-5.5",
    review_api_key_env: str = "OPENAI_API_KEY",
    review_base_url: str | None = None,
    review_timeout: float = 120.0,
    epsilon: float = 0.05,
    kernel_size: int = 5,
    min_contact_pixels: int = 50,
    min_contact_ratio: float = 0.002,
    mask_clean_kernel: int = 3,
    proposal_min_area_ratio: float = 0.006,
    proposal_max_area_ratio: float = 0.11,
    proposal_border_fraction_threshold: float = 0.18,
    save_candidates: bool = False,
    device: str | None = None,
    sam2_points_per_side: int | None = 24,
    sam2_crop_n_layers: int | None = 0,
    sam2_pred_iou_thresh: float | None = 0.7,
    sam2_stability_score_thresh: float | None = 0.88,
    preserve_unclaimed_sam2: int = 24,
) -> dict[str, Any]:
    t0 = _log_step("start", None)

    _prepare_mask_output_dir(output_mask_dir, save_candidates)
    depth_map = _load_depth_map(depth_path)
    background_exclusion_mask: np.ndarray | None = None
    try:
        background_exclusion_mask = _generate_background_exclusion_mask(
            depth_map=depth_map,
            image=Image.open(image_path).convert("RGB"),
            mask_clean_kernel=mask_clean_kernel,
        )
    except Exception as exc:
        print(f"Background exclusion mask generation failed: {exc}", file=sys.stderr, flush=True)
    t1 = _log_step("① background_mask", t0)

    mask_records, anchor_report = generate_masks_with_sam2_langsam_pipeline(
        image_path=image_path,
        output_mask_dir=output_mask_dir,
        review_model_id=review_model_id,
        review_api_key_env=review_api_key_env,
        review_base_url=review_base_url,
        review_timeout=review_timeout,
        min_area_ratio=proposal_min_area_ratio,
        max_area_ratio=proposal_max_area_ratio,
        border_fraction_threshold=proposal_border_fraction_threshold,
        mask_clean_kernel=mask_clean_kernel,
        save_candidates=save_candidates,
        device=device,
        background_exclusion_mask=background_exclusion_mask,
        depth_map=depth_map,
        sam2_points_per_side=sam2_points_per_side,
        sam2_crop_n_layers=sam2_crop_n_layers,
        sam2_pred_iou_thresh=sam2_pred_iou_thresh,
        sam2_stability_score_thresh=sam2_stability_score_thresh,
        preserve_unclaimed_sam2=preserve_unclaimed_sam2,
    )
    t2 = _log_step("② sam2+vlm+langsam_pipeline", t1)

    mask_records = _renumber_masks(mask_records, output_mask_dir)
    _draw_mask_records_label(
        image_path=image_path,
        mask_records=mask_records,
        out_path=output_mask_dir.parent / "label_2_VLM_langsam.png",
    )

    final_mask_quality_report: dict[str, Any] = {}
    mask_records, final_mask_quality_report = _finalize_independent_scene_masks(
        mask_records=mask_records,
        output_mask_dir=output_mask_dir,
        background_exclusion_mask=background_exclusion_mask,
        image_shape=tuple(depth_map.shape),
    )
    mask_records = _renumber_masks(mask_records, output_mask_dir)
    t3 = _log_step("③ finalize_non_overlap", t2)

    if background_exclusion_mask is not None and np.count_nonzero(background_exclusion_mask) > 0:
        foreground_union = np.any(
            np.stack([np.asarray(record["mask_array"], dtype=bool) for record in mask_records], axis=0),
            axis=0,
        ) if mask_records else np.zeros_like(background_exclusion_mask, dtype=bool)
        _save_mask_png(np.asarray(background_exclusion_mask, dtype=bool) & ~foreground_union, output_mask_dir / "000_background.png")

    _draw_mask_records_label(
        image_path=image_path,
        mask_records=mask_records,
        out_path=output_mask_dir.parent / "label_3_final.png",
    )

    masks = np.stack([record["mask_array"] for record in mask_records], axis=0)

    if masks.shape[1:] != depth_map.shape:
        raise ValueError(
            f"Mask shape {masks.shape[1:]} does not match depth map shape {depth_map.shape}."
        )

    graph, adjacency = build_occlusion_graph(
        masks=masks,
        depth_map=depth_map,
        epsilon=epsilon,
        kernel_size=kernel_size,
        min_contact_pixels=min_contact_pixels,
        min_contact_ratio=min_contact_ratio,
    )

    node_records: list[dict[str, Any]] = []
    for record in mask_records:
        node_record = dict(record)
        node_record.pop("mask_array", None)
        node_records.append(node_record)

    graph_payload = graph_to_jsonable(graph, adjacency, node_records=node_records)
    with Image.open(image_path) as img:
        width, height = img.size
    payload = {
        "image": {
            "path": str(image_path.resolve()),
            "width": int(width),
            "height": int(height),
        },
        "depth_map": {
            "path": str(depth_path.resolve()),
            "shape": [int(depth_map.shape[0]), int(depth_map.shape[1])],
        },
        "segmentation_backend": "sam2-langsam",
        "anchor_report": anchor_report,
        "final_mask_quality_report": final_mask_quality_report,
        "save_candidates": bool(save_candidates),
        "graph": graph_payload,
    }

    for edge in payload["graph"]["edges"]:
        source_node = node_records[edge["source"]]
        target_node = node_records[edge["target"]]
        edge["source_object_id"] = int(source_node["object_id"])
        edge["target_object_id"] = int(target_node["object_id"])
        edge["source_label"] = str(source_node["label"])
        edge["target_label"] = str(target_node["label"])

    _write_json(output_json_path, payload)

    t4 = _log_step("④ occlusion_graph", t3)
    _log_step("total", t0)
    return payload
