"""End-to-end pipeline: Molmo points -> SAM masks -> occlusion graph JSON."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

SMARTGRASP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SMARTGRASP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MOLMO_OUT = SMARTGRASP_ROOT / "perception" / "molmo" / "out"

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
from transformers import GenerationConfig

from SmartGrasp.perception.occul_map.org import build_occlusion_graph, graph_to_jsonable

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only in cv2-less environments
    cv2 = None

_SAM_CACHE: dict[tuple[str, str], tuple[SamProcessor, SamModel]] = {}
_LANGSAM_CACHE: dict[tuple[str, str], Any] = {}
GROUNDING_DINO_LOCAL_MODEL_PATH = Path(
    "/home/data/models/huggingface/hub/models--IDEA-Research--grounding-dino-base/"
    "snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
)


@dataclass(frozen=True)
class MolmoPoint:
    molmo_id: int
    x: int
    y: int
    label: str


def _safe_label(label: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in label.strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "object"


LANGSAM_GUIDELINE = (
    " Segment the complete object. An object may have multiple parts, colors, "
    "or irregular shapes — judge by overall form, usage, and color to include all of it in one mask. "
    "Separate this object from adjacent objects even if they touch: "
    "only return the mask for this one object."
)


def _semantic_prompt(label: str) -> str:
    prompt = " ".join(part for part in _sanitize_label(label).split("_") if part)
    return f"{prompt}.{LANGSAM_GUIDELINE}" if prompt else f"object.{LANGSAM_GUIDELINE}"


VAGUE_LABEL_TERMS = {
    "unknown",
    "unknownproduct",
    "unknown_product",
    "object",
    "item",
    "product",
    "thing",
}


def _label_tokens(label: str) -> set[str]:
    return set(_safe_label(label).split("_"))


def _sanitize_label(label: str) -> str:
    tokens = [token for token in _safe_label(label).split("_") if token]
    filtered = [token for token in tokens if token not in VAGUE_LABEL_TERMS]
    if filtered:
        return "_".join(filtered)
    return "_".join(tokens)


def _filter_points(points: list[MolmoPoint]) -> tuple[list[MolmoPoint], list[dict[str, Any]]]:
    kept: list[MolmoPoint] = []
    for point in points:
        sanitized = _sanitize_label(point.label)
        kept.append(MolmoPoint(point.molmo_id, point.x, point.y, sanitized or point.label))
    return kept, []


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


def _load_points(points_json_path: Path) -> tuple[dict[str, Any], list[MolmoPoint], Path]:
    payload = _load_json(points_json_path)
    raw_points = payload.get("points", [])
    if not raw_points:
        raise ValueError(f"No points found in {points_json_path}.")

    points: list[MolmoPoint] = []
    for point in raw_points:
        points.append(
            MolmoPoint(
                molmo_id=int(point["molmo_id"]),
                x=int(point["x"]),
                y=int(point["y"]),
                label=str(point.get("label", f"object_{point['molmo_id']}")),
            )
        )

    image_meta = payload.get("image", {})
    image_path_value = image_meta.get("path")
    if not image_path_value:
        raise ValueError(f"Missing image.path in {points_json_path}.")
    image_path = _resolve_path(points_json_path, image_path_value)
    return payload, points, image_path


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


def _prompt_offsets(radius: int, mode: str) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    if radius <= 0:
        return offsets
    if mode == "cross":
        offsets.extend([(radius, 0), (-radius, 0), (0, radius), (0, -radius)])
    elif mode == "grid":
        offsets.extend(
            (dx, dy)
            for dy in (-radius, 0, radius)
            for dx in (-radius, 0, radius)
            if dx != 0 or dy != 0
        )
    elif mode == "ring":
        offsets.extend(
            [
                (radius, 0),
                (-radius, 0),
                (0, radius),
                (0, -radius),
                (radius, radius),
                (radius, -radius),
                (-radius, radius),
                (-radius, -radius),
            ]
        )
    else:
        raise ValueError(f"Unsupported SAM prompt mode: {mode}")
    return offsets


def _clip_unique_points(
    coords: list[tuple[int, int]],
    width: int,
    height: int,
) -> list[list[int]]:
    points: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for x_raw, y_raw in coords:
        x = int(np.clip(x_raw, 0, width - 1))
        y = int(np.clip(y_raw, 0, height - 1))
        if (x, y) not in seen:
            seen.add((x, y))
            points.append([x, y])
    return points


def _positive_prompt_points(
    point: MolmoPoint,
    width: int,
    height: int,
    radius: int,
    mode: str,
) -> list[list[int]]:
    coords = [(point.x + dx, point.y + dy) for dx, dy in _prompt_offsets(radius, mode)]
    return _clip_unique_points(coords, width, height)


def _negative_prompt_points(
    target: MolmoPoint,
    points: list[MolmoPoint],
    width: int,
    height: int,
    max_points: int,
) -> list[list[int]]:
    if max_points <= 0:
        return []
    other_points = [point for point in points if point.molmo_id != target.molmo_id]
    other_points.sort(key=lambda point: (point.x - target.x) ** 2 + (point.y - target.y) ** 2)
    coords = [(point.x, point.y) for point in other_points[:max_points]]
    return _clip_unique_points(coords, width, height)


def _build_sam_prompt(
    target: MolmoPoint,
    points: list[MolmoPoint],
    width: int,
    height: int,
    radius: int,
    mode: str,
    negative_points: int,
) -> tuple[list[list[int]], list[int], list[list[int]], list[list[int]]]:
    positives = _positive_prompt_points(target, width, height, radius, mode)
    negatives = _negative_prompt_points(target, points, width, height, negative_points)
    input_points = positives + negatives
    input_labels = [1 for _ in positives] + [0 for _ in negatives]
    return input_points, input_labels, positives, negatives


def _prompt_variants(
    target: MolmoPoint,
    points: list[MolmoPoint],
    width: int,
    height: int,
    radius: int,
    mode: str,
    negative_points: int,
) -> list[dict[str, Any]]:
    if mode != "auto":
        input_points, input_labels, positives, negatives = _build_sam_prompt(
            target=target,
            points=points,
            width=width,
            height=height,
            radius=radius,
            mode=mode,
            negative_points=negative_points,
        )
        return [
            {
                "mode": mode,
                "radius": radius,
                "negative_points": negative_points,
                "input_points": input_points,
                "input_labels": input_labels,
                "positive_points": positives,
                "negative_prompt_points": negatives,
            }
        ]

    radii = [0]
    if radius > 0:
        radii.extend([radius, radius * 2])

    specs: list[tuple[str, int, int]] = []
    for prompt_radius in radii:
        specs.append(("cross", prompt_radius, 0))
        if prompt_radius > 0:
            specs.append(("grid", prompt_radius, 0))
            specs.append(("ring", prompt_radius, 0))
    if negative_points > 0 and radius > 0:
        specs.append(("grid", radius, negative_points))

    variants: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for prompt_mode, prompt_radius, prompt_negative_points in specs:
        key = (prompt_mode, prompt_radius, prompt_negative_points)
        if key in seen:
            continue
        seen.add(key)
        input_points, input_labels, positives, negatives = _build_sam_prompt(
            target=target,
            points=points,
            width=width,
            height=height,
            radius=prompt_radius,
            mode=prompt_mode,
            negative_points=prompt_negative_points,
        )
        variants.append(
            {
                "mode": prompt_mode,
                "radius": prompt_radius,
                "negative_points": prompt_negative_points,
                "input_points": input_points,
                "input_labels": input_labels,
                "positive_points": positives,
                "negative_prompt_points": negatives,
            }
        )
    return variants


def _point_grid(point: MolmoPoint, width: int, height: int, radius: int) -> list[list[int]]:
    """Backward-compatible cross prompt used by older callers/tests."""
    offsets = _prompt_offsets(radius, "cross")
    coords = [(point.x + dx, point.y + dy) for dx, dy in offsets]
    return _clip_unique_points(coords, width, height)


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


def _normalize_processed_masks(processed_masks: Any) -> np.ndarray:
    masks_np = np.asarray(processed_masks)
    if masks_np.ndim == 4 and masks_np.shape[0] == 1:
        masks_np = masks_np[0]
    if masks_np.ndim != 3:
        raise ValueError(f"Unexpected mask tensor shape after post-processing: {masks_np.shape}.")
    return masks_np > 0


def _point_hits(mask: np.ndarray, input_points: list[list[int]], radius: int = 3) -> int:
    height, width = mask.shape
    hits = 0
    for x, y in input_points:
        x0 = max(0, int(x) - radius)
        x1 = min(width, int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(height, int(y) + radius + 1)
        if np.any(mask[y0:y1, x0:x1]):
            hits += 1
    return hits


def _select_best_mask(
    processed_masks: Any,
    iou_scores: torch.Tensor,
    input_points: list[list[int]],
    mask_clean_kernel: int,
    negative_points: list[list[int]] | None = None,
    prompt_box: list[int] | None = None,
    prompt_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    masks_np = _normalize_processed_masks(processed_masks)

    scores_np = iou_scores.detach().float().cpu().numpy().reshape(-1)
    if len(scores_np) < masks_np.shape[0]:
        scores_np = np.pad(scores_np, (0, masks_np.shape[0] - len(scores_np)), constant_values=0.0)

    image_area = float(masks_np.shape[1] * masks_np.shape[2])
    raw_areas = np.array([np.count_nonzero(mask) for mask in masks_np], dtype=np.float32)
    max_area = float(max(1.0, raw_areas.max(initial=1.0)))

    candidates: list[dict[str, Any]] = []
    for idx, raw_mask in enumerate(masks_np):
        cleaned_mask = _clean_mask(raw_mask, mask_clean_kernel)
        area = int(np.count_nonzero(cleaned_mask))
        area_ratio = float(area / image_area)
        prompt_hits = _point_hits(cleaned_mask, input_points)
        prompt_coverage = float(prompt_hits / max(1, len(input_points)))
        negative_hits = _point_hits(cleaned_mask, negative_points or [])
        box_coverage = 0.0
        outside_box_ratio = 0.0
        if prompt_box is not None:
            x0, y0, x1, y1 = prompt_box
            box_mask = np.zeros_like(cleaned_mask, dtype=bool)
            box_mask[y0:y1, x0:x1] = True
            box_area = max(1, int(np.count_nonzero(box_mask)))
            mask_area = max(1, area)
            box_coverage = float(np.count_nonzero(cleaned_mask & box_mask) / box_area)
            outside_box_ratio = float(np.count_nonzero(cleaned_mask & ~box_mask) / mask_area)
        too_small_penalty = max(0.0, 0.001 - area_ratio) * 200.0
        large_penalty = max(0.0, area_ratio - 0.08) * 4.0
        giant_penalty = max(0.0, area_ratio - 0.18) * 10.0
        empty_penalty = 1.0 if area == 0 else 0.0
        prompt_penalty = 0.6 if prompt_hits == 0 else 0.0
        negative_penalty = 1.5 * negative_hits
        box_bonus = 0.4 * box_coverage if prompt_box is not None else 0.0
        box_penalty = 2.5 * outside_box_ratio if prompt_box is not None else 0.0
        selection_score = (
            float(scores_np[idx])
            + 0.35 * prompt_coverage
            + box_bonus
            - too_small_penalty
            - large_penalty
            - giant_penalty
            - empty_penalty
            - prompt_penalty
            - negative_penalty
            - box_penalty
        )
        candidate = {
            "candidate_index": idx,
            "mask": cleaned_mask,
            "area": area,
            "area_ratio": area_ratio,
            "predicted_iou": float(scores_np[idx]),
            "prompt_hits": int(prompt_hits),
            "prompt_coverage": prompt_coverage,
            "negative_hits": int(negative_hits),
            "box_coverage": float(box_coverage),
            "outside_box_ratio": float(outside_box_ratio),
            "selection_score": float(selection_score),
        }
        if prompt_metadata:
            candidate.update(prompt_metadata)
        candidates.append(candidate)

    best = max(candidates, key=lambda item: item["selection_score"])
    metadata = {key: value for key, value in best.items() if key != "mask"}
    candidate_metadata = [{key: value for key, value in item.items() if key != "mask"} for item in candidates]
    return best["mask"], metadata, candidate_metadata


def _load_sam(sam_model_id: str, device: str) -> tuple[SamProcessor, SamModel]:
    cache_key = (sam_model_id, device)
    if cache_key not in _SAM_CACHE:
        processor = SamProcessor.from_pretrained(sam_model_id)
        model = SamModel.from_pretrained(sam_model_id).to(device)
        model.eval()
        _SAM_CACHE[cache_key] = (processor, model)
    return _SAM_CACHE[cache_key]


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


def _point_inside_mask(mask: np.ndarray, point: MolmoPoint, radius: int = 3) -> bool:
    height, width = mask.shape
    x0 = max(0, int(point.x) - radius)
    x1 = min(width, int(point.x) + radius + 1)
    y0 = max(0, int(point.y) - radius)
    y1 = min(height, int(point.y) + radius + 1)
    return bool(np.any(mask[y0:y1, x0:x1]))


def _other_points_inside_mask(
    mask: np.ndarray,
    target: MolmoPoint,
    points: list[MolmoPoint],
    radius: int = 3,
) -> list[MolmoPoint]:
    return [
        point
        for point in points
        if point.molmo_id != target.molmo_id and _point_inside_mask(mask, point, radius=radius)
    ]


def _point_inside_xy(mask: np.ndarray, x: int, y: int, radius: int = 3) -> bool:
    height, width = mask.shape
    x0 = max(0, int(x) - radius)
    x1 = min(width, int(x) + radius + 1)
    y0 = max(0, int(y) - radius)
    y1 = min(height, int(y) + radius + 1)
    return bool(np.any(mask[y0:y1, x0:x1]))


def _mask_centroid_distance(mask: np.ndarray, point: MolmoPoint) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float("inf")
    cx = float(xs.mean())
    cy = float(ys.mean())
    return float(np.hypot(cx - point.x, cy - point.y))


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


def _box_contains_point(box: list[int] | None, point: MolmoPoint, margin: int = 3) -> bool:
    if box is None:
        return False
    x0, y0, x1, y1 = box
    return bool(x0 - margin <= point.x <= x1 + margin and y0 - margin <= point.y <= y1 + margin)


def _other_points_inside_box(
    box: list[int] | None,
    target: MolmoPoint,
    points: list[MolmoPoint],
    margin: int = 3,
) -> list[MolmoPoint]:
    if box is None:
        return []
    return [
        point
        for point in points
        if point.molmo_id != target.molmo_id and _box_contains_point(box, point, margin=margin)
    ]


def _clip_box_xyxy(box: list[int], width: int, height: int) -> list[int] | None:
    x0, y0, x1, y1 = box
    clipped = [
        int(np.clip(x0, 0, width - 1)),
        int(np.clip(y0, 0, height - 1)),
        int(np.clip(x1, 1, width)),
        int(np.clip(y1, 1, height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _expand_box_xyxy(box: list[int], width: int, height: int, margin_ratio: float = 0.04) -> list[int] | None:
    x0, y0, x1, y1 = box
    margin = int(round(max(x1 - x0, y1 - y0) * margin_ratio))
    return _clip_box_xyxy([x0 - margin, y0 - margin, x1 + margin, y1 + margin], width, height)


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    x, y, width, height = _mask_bbox(mask)
    if width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


def _box_area_ratio_xyxy(box: list[int], width: int, height: int) -> float:
    x0, y0, x1, y1 = box
    return float(((x1 - x0) * (y1 - y0)) / max(1, width * height))


def _select_text_box_candidate(
    masks: list[np.ndarray],
    scores: list[float],
    boxes: list[list[int] | None],
    point: MolmoPoint,
    points: list[MolmoPoint],
    width: int,
    height: int,
) -> tuple[list[int] | None, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    candidate_count = max(len(boxes), len(masks))
    for idx in range(candidate_count):
        raw_box = boxes[idx] if idx < len(boxes) else None
        if raw_box is None and idx < len(masks):
            raw_box = _mask_bbox_xyxy(masks[idx])
        if raw_box is None:
            continue

        box = _clip_box_xyxy(raw_box, width, height)
        if box is None:
            continue

        contains_point = _box_contains_point(box, point, margin=5)
        other_points = _other_points_inside_box(box, point, points, margin=5)
        area_ratio = _box_area_ratio_xyxy(box, width, height)
        semantic_score = float(scores[idx]) if idx < len(scores) else 0.0
        size_penalty = max(0.0, 0.002 - area_ratio) * 100.0 + max(0.0, area_ratio - 0.35) * 4.0
        selection_score = (
            semantic_score
            + (3.0 if contains_point else -2.0)
            - 2.5 * len(other_points)
            - size_penalty
        )
        candidates.append(
            {
                "candidate_index": idx,
                "box": box,
                "area_ratio": float(area_ratio),
                "contains_point": bool(contains_point),
                "other_points_in_box": [
                    {"molmo_id": other.molmo_id, "x": other.x, "y": other.y, "label": other.label}
                    for other in other_points
                ],
                "semantic_score": semantic_score,
                "selection_score": float(selection_score),
            }
        )

    if not candidates:
        return None, {}, []

    point_candidates = [candidate for candidate in candidates if candidate["contains_point"]]
    best_pool = point_candidates or candidates
    best = max(best_pool, key=lambda item: item["selection_score"])
    box = _expand_box_xyxy(best["box"], width, height)
    selected = {key: value for key, value in best.items() if key != "box"}
    selected["box"] = box
    selected["selection_pool"] = "point" if point_candidates else "all"
    return box, selected, candidates


def _detect_langsam_text_box(
    model: Any,
    image: Image.Image,
    point: MolmoPoint,
    points: list[MolmoPoint],
) -> tuple[list[int] | None, dict[str, Any], list[dict[str, Any]]]:
    width, height = image.size
    prompt = _semantic_prompt(point.label)
    masks, scores, boxes = _langsam_predict(model, image, prompt)
    selected_box, selected_metadata, candidates = _select_text_box_candidate(
        masks=masks,
        scores=scores,
        boxes=boxes,
        point=point,
        points=points,
        width=width,
        height=height,
    )
    if selected_metadata:
        selected_metadata["semantic_prompt"] = prompt
    return selected_box, selected_metadata, candidates


def _select_langsam_mask(
    masks: list[np.ndarray],
    scores: list[float],
    boxes: list[list[int] | None],
    point: MolmoPoint,
    points: list[MolmoPoint],
    mask_clean_kernel: int,
    previous_masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    previous_masks = previous_masks or []
    for idx, raw_mask in enumerate(masks):
        mask = _clean_mask(raw_mask, mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        contains_point = _point_inside_mask(mask, point)
        other_points_in_mask = _other_points_inside_mask(mask, point, points)
        max_previous_iou = max((_mask_iou(mask, previous) for previous in previous_masks), default=0.0)
        score = float(scores[idx]) if idx < len(scores) else 0.0
        selection_score = (
            4.0 * score
            + (3.0 if contains_point else -3.0)
            - 3.0 * len(other_points_in_mask)
            - 2.0 * max_previous_iou
        )
        candidates.append(
            {
                "candidate_index": idx,
                "mask": mask,
                "area": area,
                "contains_point": contains_point,
                "other_points_in_mask": [
                    {"molmo_id": other.molmo_id, "x": other.x, "y": other.y, "label": other.label}
                    for other in other_points_in_mask
                ],
                "max_previous_iou": float(max_previous_iou),
                "semantic_score": score,
                "selection_score": selection_score,
            }
        )

    if not candidates:
        return None, {}, []

    point_candidates = [candidate for candidate in candidates if candidate["contains_point"]]
    best_pool_name = "point" if point_candidates else "all"
    best_pool = point_candidates or candidates
    best = max(best_pool, key=lambda item: item["selection_score"])
    metadata = {key: value for key, value in best.items() if key != "mask"}
    metadata["selection_pool"] = best_pool_name
    candidate_metadata = [{key: value for key, value in item.items() if key != "mask"} for item in candidates]
    return best["mask"], metadata, candidate_metadata


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


def _labels_compatible(label_a: str, label_b: str) -> bool:
    tokens_a = _label_tokens(label_a) - VAGUE_LABEL_TERMS
    tokens_b = _label_tokens(label_b) - VAGUE_LABEL_TERMS
    if not tokens_a or not tokens_b:
        return True
    return bool(tokens_a & tokens_b)


def _draw_mask_records_label(
    image_path: Path,
    mask_records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    from SmartGrasp.perception.molmo.molmo_annotator.draw import draw_labeled_image_matplotlib

    points_with_ids: list[tuple[int, int, int]] = []
    for record in mask_records:
        point = record.get("point", {})
        x = int(point.get("x", -1))
        y = int(point.get("y", -1))
        if x < 0 or y < 0:
            continue
        molmo_id = int(record.get("molmo_id", record.get("node_id", len(points_with_ids) + 1)))
        points_with_ids.append((molmo_id, x, y))
    with Image.open(image_path) as image:
        draw_labeled_image_matplotlib(
            image=image,
            points_with_ids=points_with_ids,
            out_png_path=str(out_path),
        )


def _drop_contained_duplicate_masks(
    mask_records: list[dict[str, Any]],
    containment_threshold: float = 0.3,
    duplicate_dir: Path | None = None,
    mode: str = "either",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in mask_records:
        mask = np.asarray(record["mask_array"], dtype=bool)
        duplicate_of: dict[str, Any] | None = None
        for kept_record in kept:
            kept_mask = np.asarray(kept_record["mask_array"], dtype=bool)
            new_in_kept = _mask_overlap_fraction(mask, kept_mask)
            kept_in_new = _mask_overlap_fraction(kept_mask, mask)
            if mode == "new_in_existing":
                is_duplicate = new_in_kept >= containment_threshold and int(np.count_nonzero(mask)) <= int(np.count_nonzero(kept_mask))
            else:
                is_duplicate = new_in_kept >= containment_threshold or kept_in_new >= containment_threshold
            if not is_duplicate:
                continue
            duplicate_of = kept_record
            break

        if duplicate_of is None:
            kept.append(record)
        else:
            removed_record = {key: value for key, value in record.items() if key != "mask_array"}
            removed_record["duplicate_of_molmo_id"] = int(duplicate_of.get("molmo_id", -1))
            removed_record["duplicate_reason"] = "mask_contained_in_existing_instance"
            old_mask_path = Path(str(record.get("mask_path", "")))
            if duplicate_dir is not None and old_mask_path.exists():
                duplicate_dir.mkdir(parents=True, exist_ok=True)
                new_mask_path = duplicate_dir / old_mask_path.name
                shutil.move(str(old_mask_path), str(new_mask_path))
                removed_record["mask_path"] = str(new_mask_path.resolve())
                removed_record["mask_status"] = "archived_duplicate"
            removed.append(removed_record)

    for node_id, record in enumerate(kept):
        record["node_id"] = node_id
    return kept, removed


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
        0 if str(record.get("segmentation_backend", "")).startswith("sam2_auto_molmo_review") else 1
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
                    "molmo_id": int(kept[index].get("molmo_id", index + 1)),
                    "removed_overlap_pixels": overlap_pixels,
                    "removed_overlap_fraction": float(overlap_pixels / max(1, areas[index])),
                    "overlap_with": [
                        {
                            "molmo_id": int(kept[int(owner_id)].get("molmo_id", int(owner_id) + 1)),
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
    for index, record in enumerate(kept):
        mask = masks[index]
        area = int(np.count_nonzero(mask))
        if area == 0:
            removed = {key: value for key, value in record.items() if key != "mask_array"}
            removed["duplicate_reason"] = "removed_after_overlap_exclusivity"
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
    """Renumber molmo_id/node_id sequentially (1,2,3...), rename mask files on disk, keep labels in sync."""
    renumbered: list[dict[str, Any]] = []
    for index, record in enumerate(mask_records, start=1):
        old_path = Path(str(record.get("mask_path", "")))
        label = _safe_label(str(record.get("label", f"object_{index}")))
        backend = record.get("segmentation_backend", "")
        source = "sam2" if "sam2" in backend else "molmo"
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
        record["molmo_id"] = index
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


def _bbox_gap(bbox_a: list[int], bbox_b: list[int]) -> float:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    ax1 = ax + aw
    ay1 = ay + ah
    bx1 = bx + bw
    by1 = by + bh
    dx = max(bx - ax1, ax - bx1, 0)
    dy = max(by - ay1, ay - by1, 0)
    return float(np.hypot(dx, dy))


def _bbox_union(bboxes: list[list[int]]) -> list[int]:
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[0] + bbox[2] for bbox in bboxes)
    y1 = max(bbox[1] + bbox[3] for bbox in bboxes)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _bbox_area_ratio(bbox: list[int], image_area: float) -> float:
    return float((bbox[2] * bbox[3]) / max(1.0, image_area))


def _proposal_bboxes_should_cluster(
    bbox_a: list[int],
    bbox_b: list[int],
    image_area: float,
    max_area_ratio: float,
) -> bool:
    union_bbox = _bbox_union([bbox_a, bbox_b])
    if _bbox_area_ratio(union_bbox, image_area) > max_area_ratio:
        return False

    gap = _bbox_gap(bbox_a, bbox_b)
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_y = min(ay + ah, by + bh) - max(ay, by)
    shared_axis = overlap_x > -24 or overlap_y > -24
    close = gap <= 32.0
    return bool(close and shared_axis)


def _fill_cluster_mask(mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return mask
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return mask

    points = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(points)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    kernel = np.ones((7, 7), dtype=np.uint8)
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def _cluster_proposal_candidates(
    candidates: list[dict[str, Any]],
    image_area: float,
    max_area_ratio: float,
) -> list[dict[str, Any]]:
    remaining = sorted(candidates, key=lambda item: item["selection_score"], reverse=True)
    clustered: list[dict[str, Any]] = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            next_remaining: list[dict[str, Any]] = []
            for candidate in remaining:
                if any(
                    _proposal_bboxes_should_cluster(
                        list(member["bbox"]),
                        list(candidate["bbox"]),
                        image_area,
                        max_area_ratio,
                    )
                    for member in cluster
                ):
                    cluster.append(candidate)
                    changed = True
                else:
                    next_remaining.append(candidate)
            remaining = next_remaining

        if len(cluster) == 1:
            clustered.append(seed)
            continue

        raw_union = np.any(np.stack([np.asarray(item["mask"], dtype=bool) for item in cluster], axis=0), axis=0)
        merged_mask = _fill_cluster_mask(raw_union)
        area = int(np.count_nonzero(merged_mask))
        bbox = _bbox_union([list(item["bbox"]) for item in cluster])
        cx, cy = _mask_centroid_xy(merged_mask)
        best = max(cluster, key=lambda item: item["selection_score"])
        merged = {key: value for key, value in best.items() if key != "mask"}
        merged.update(
            {
                "proposal_index": int(best["proposal_index"]),
                "proposal_indices": [int(item["proposal_index"]) for item in cluster],
                "cluster_size": len(cluster),
                "area": area,
                "area_ratio": float(area / image_area),
                "bbox": bbox,
                "centroid": {"x": cx, "y": cy},
                "selection_score": float(max(item["selection_score"] for item in cluster) + 0.02 * len(cluster)),
                "mask": merged_mask,
                "clustered_from": [
                    {key: value for key, value in item.items() if key != "mask"}
                    for item in cluster
                ],
            }
        )
        clustered.append(merged)

    clustered.sort(key=lambda item: item["selection_score"], reverse=True)
    return clustered


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
LANGSAM_OTHER_POINT_FALLBACK_THRESHOLD = 2


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


def merge_physically_connected_masks(
    masks: np.ndarray,
    depth_map: np.ndarray,
    depth_threshold: float = 0.015,
    boundary_kernel_size: int = 3,
    min_contact_pixels: int = 8,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """Merge adjacent mask fragments whose boundary depth is physically smooth.

    The function is intentionally standalone and only depends on numpy/cv2. It
    treats masks as instances, finds touching or near-touching pairs through a
    one-step dilation, then measures the median depth on both sides of the
    shared boundary. If the depth delta is below ``depth_threshold`` meters, the
    pair is considered parts of the same rigid object and is unioned.
    """
    masks_np = np.asarray(masks, dtype=bool)
    if masks_np.ndim != 3:
        raise ValueError(f"Expected masks with shape (N, H, W), got {masks_np.shape}.")
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.shape != masks_np.shape[1:]:
        raise ValueError(f"Depth map shape {depth.shape} does not match mask shape {masks_np.shape[1:]}.")

    count = masks_np.shape[0]
    if count <= 1:
        report: list[dict[str, Any]] = []
        return (masks_np, report) if return_report else masks_np

    parent = list(range(count))

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

    dilated_masks = [_dilate_mask(mask, boundary_kernel_size) for mask in masks_np]
    report: list[dict[str, Any]] = []
    for left in range(count):
        for right in range(left + 1, count):
            adjacent = bool(np.any(dilated_masks[left] & masks_np[right]) or np.any(dilated_masks[right] & masks_np[left]))
            if not adjacent:
                continue

            depth_delta, contact_pixels = _mask_boundary_depth_delta(
                masks_np[left],
                masks_np[right],
                depth,
                boundary_kernel_size=boundary_kernel_size,
            )
            should_merge = (
                depth_delta is not None
                and contact_pixels >= min_contact_pixels
                and depth_delta <= depth_threshold
            )
            entry = {
                "left_index": int(left),
                "right_index": int(right),
                "contact_pixels": int(contact_pixels),
                "depth_delta": None if depth_delta is None else float(depth_delta),
                "depth_threshold": float(depth_threshold),
                "merged": bool(should_merge),
            }
            report.append(entry)
            if should_merge:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    merged_masks = []
    for members in groups.values():
        merged_masks.append(np.any(masks_np[members], axis=0))
    merged_np = np.stack(merged_masks, axis=0) if merged_masks else masks_np[:0]
    return (merged_np, report) if return_report else merged_np


def _merge_mask_records_by_depth(
    mask_records: list[dict[str, Any]],
    depth_map: np.ndarray,
    output_mask_dir: Path,
    depth_threshold: float = 0.015,
    boundary_kernel_size: int = 3,
    min_contact_pixels: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(mask_records) <= 1 or depth_threshold <= 0:
        return mask_records, []

    masks = np.stack([np.asarray(record["mask_array"], dtype=bool) for record in mask_records], axis=0)
    _merged_masks, report = merge_physically_connected_masks(
        masks=masks,
        depth_map=depth_map,
        depth_threshold=depth_threshold,
        boundary_kernel_size=boundary_kernel_size,
        min_contact_pixels=min_contact_pixels,
        return_report=True,
    )

    parent = list(range(len(mask_records)))

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

    for entry in report:
        if entry.get("merged"):
            union(int(entry["left_index"]), int(entry["right_index"]))

    groups: dict[int, list[int]] = {}
    for index in range(len(mask_records)):
        groups.setdefault(find(index), []).append(index)

    merged_records: list[dict[str, Any]] = []
    for members in groups.values():
        base = dict(mask_records[members[0]])
        member_records = [mask_records[index] for index in members]
        merged_mask = np.any(
            np.stack([np.asarray(record["mask_array"], dtype=bool) for record in member_records], axis=0),
            axis=0,
        )
        cx, cy = _mask_centroid_xy(merged_mask)
        base["mask_array"] = merged_mask
        base["mask_area"] = int(np.count_nonzero(merged_mask))
        base["point"] = {"x": int(cx), "y": int(cy)}
        if len(members) > 1:
            labels = [str(record.get("label", "")).strip() for record in member_records if str(record.get("label", "")).strip()]
            base["label"] = labels[0] if labels else str(base.get("label", "object"))
            base["segmentation_backend"] = f"{base.get('segmentation_backend', 'mask')}_depth_merged"
            base["depth_merged_from"] = [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"mask_array", "sam_candidates", "semantic_candidates", "text_box_candidates"}
                }
                for record in member_records
            ]

        old_path = Path(str(base.get("mask_path", "")))
        if not old_path.exists():
            old_path = output_mask_dir / f"depth_merge_{len(merged_records) + 1:03d}_{_safe_label(str(base.get('label', 'object')))}.png"
        _save_mask_png(merged_mask, old_path)
        base["mask_path"] = str(old_path.resolve())
        merged_records.append(base)

    for node_id, record in enumerate(merged_records):
        record["node_id"] = node_id
    return merged_records, report


def _box_xywh_to_xyxy(box: list[int]) -> list[int] | None:
    if len(box) < 4:
        return None
    x, y, width, height = [int(value) for value in box[:4]]
    if width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


def _box_mask(shape: tuple[int, int], box_xyxy: list[int] | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if box_xyxy is None:
        return mask
    height, width = shape
    clipped = _clip_box_xyxy(box_xyxy, width, height)
    if clipped is None:
        return mask
    x0, y0, x1, y1 = clipped
    mask[y0:y1, x0:x1] = True
    return mask


def _mask_box_overlap_fraction(mask: np.ndarray, box_xyxy: list[int] | None) -> float:
    if box_xyxy is None:
        return 0.0
    mask_bool = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask_bool))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(mask_bool & _box_mask(mask_bool.shape, box_xyxy)) / area)


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"mask"}
    }


def _contains_other_molmo_point(mask: np.ndarray, target: MolmoPoint, points: list[MolmoPoint]) -> list[MolmoPoint]:
    return [
        point
        for point in points
        if point.molmo_id != target.molmo_id and _point_inside_mask(mask, point, radius=3)
    ]


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


def _anchor_candidate_score(
    candidate: dict[str, Any],
    point: MolmoPoint,
    text_box: list[int] | None,
) -> float:
    mask = np.asarray(candidate["mask"], dtype=bool)
    contains_point = _point_inside_mask(mask, point, radius=3)
    text_box_overlap = _mask_box_overlap_fraction(mask, text_box)
    cx = float(candidate.get("centroid", {}).get("x", point.x))
    cy = float(candidate.get("centroid", {}).get("y", point.y))
    distance = float(np.hypot(cx - point.x, cy - point.y))
    height, width = mask.shape
    distance_prior = max(0.0, 1.0 - distance / max(1.0, 0.35 * np.hypot(width, height)))
    return (
        float(candidate.get("selection_score", 0.0))
        + (4.0 if contains_point else 0.0)
        + 2.0 * text_box_overlap
        + 0.5 * distance_prior
    )


def _candidate_allowed_for_anchor(
    candidate: dict[str, Any],
    point: MolmoPoint,
    points: list[MolmoPoint],
    text_box: list[int] | None,
) -> tuple[bool, str | None]:
    mask = np.asarray(candidate["mask"], dtype=bool)
    other_points = _contains_other_molmo_point(mask, point, points)
    if other_points:
        return False, "contains_other_molmo_point"

    if _point_inside_mask(mask, point, radius=3):
        return True, None

    if text_box is None:
        return True, None

    if _mask_box_overlap_fraction(mask, text_box) >= 0.05:
        return True, None

    bbox_xyxy = candidate.get("bbox_xyxy")
    if bbox_xyxy is not None:
        candidate_box_mask = _box_mask(mask.shape, bbox_xyxy)
        if _mask_box_overlap_fraction(candidate_box_mask, text_box) >= 0.15:
            return True, None

    return False, "outside_anchor_text_box"


def _anchor_group_candidates(
    candidates: list[dict[str, Any]],
    point: MolmoPoint,
    points: list[MolmoPoint],
    depth_map: np.ndarray,
    text_box: list[int] | None,
    depth_threshold: float,
    boundary_kernel_size: int,
    min_contact_pixels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    for candidate in candidates:
        ok, reason = _candidate_allowed_for_anchor(candidate, point, points, text_box)
        if ok:
            allowed.append(candidate)
        else:
            report.append(
                {
                    "proposal_index": int(candidate["proposal_index"]),
                    "accepted": False,
                    "reason": reason,
                }
            )

    seeds = [candidate for candidate in allowed if _point_inside_mask(candidate["mask"], point, radius=4)]
    if not seeds and text_box is not None:
        seeds = [candidate for candidate in allowed if _mask_box_overlap_fraction(candidate["mask"], text_box) >= 0.25]
    if not seeds:
        return [], report

    seed = max(seeds, key=lambda item: _anchor_candidate_score(item, point, text_box))
    selected: list[dict[str, Any]] = [seed]
    selected_ids = {int(seed["proposal_index"])}
    changed = True
    while changed:
        changed = False
        union_mask = np.any(np.stack([np.asarray(item["mask"], dtype=bool) for item in selected], axis=0), axis=0)
        for candidate in sorted(allowed, key=lambda item: _anchor_candidate_score(item, point, text_box), reverse=True):
            proposal_index = int(candidate["proposal_index"])
            if proposal_index in selected_ids:
                continue

            candidate_mask = np.asarray(candidate["mask"], dtype=bool)
            adjacent = bool(np.any(_dilate_mask(union_mask, boundary_kernel_size) & candidate_mask))
            if not adjacent:
                report.append(
                    {
                        "proposal_index": proposal_index,
                        "accepted": False,
                        "reason": "not_adjacent_to_anchor_group",
                    }
                )
                continue

            depth_delta, contact_pixels = _mask_boundary_depth_delta(
                union_mask,
                candidate_mask,
                depth_map,
                boundary_kernel_size=boundary_kernel_size,
            )
            should_add = (
                depth_delta is not None
                and contact_pixels >= min_contact_pixels
                and depth_delta <= depth_threshold
            )
            report.append(
                {
                    "proposal_index": proposal_index,
                    "accepted": bool(should_add),
                    "reason": None if should_add else "depth_discontinuity",
                    "contact_pixels": int(contact_pixels),
                    "depth_delta": None if depth_delta is None else float(depth_delta),
                    "depth_threshold": float(depth_threshold),
                }
            )
            if should_add:
                selected.append(candidate)
                selected_ids.add(proposal_index)
                changed = True

    return selected, report


def _anchor_description(point: MolmoPoint, image_np: np.ndarray, mask: np.ndarray) -> str:
    color = _proposal_color_name(image_np, mask)
    shape = _proposal_shape_name(mask)
    visual_parts = [part for part in (color, shape) if part and part != "visible"]
    visual = " ".join(visual_parts)
    label = point.label.strip() or "object"
    if visual:
        return f"{label}, {visual} object"
    return label


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
    from SmartGrasp.perception.molmo.molmo_annotator.draw import draw_labeled_image_matplotlib

    points_with_ids: list[tuple[int, int, int]] = []
    masks: list[np.ndarray] = []
    for index, candidate in enumerate(candidates[:max_labels], start=1):
        centroid = candidate.get("centroid", {})
        x = int(centroid.get("x", 0))
        y = int(centroid.get("y", 0))
        points_with_ids.append((index, x, y))
        masks.append(np.asarray(candidate["mask"], dtype=bool))

    with Image.open(image_path).convert("RGB") as image:
        draw_labeled_image_matplotlib(image=image, points_with_ids=points_with_ids, out_png_path=str(out_path))

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

    payload = _extract_json_from_text(raw_output)
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Molmo SAM2 review response must contain an `objects` list.")

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
        raise ValueError("Molmo SAM2 review returned no valid objects.")

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
    (out_dir / "molmo_sam2_review.json").write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "openai_sam2_review.json").write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized, raw_output


def _select_langsam_mask_for_review_object(
    masks: list[np.ndarray],
    scores: list[float],
    review_object: dict[str, Any],
    candidates: list[dict[str, Any]],
    background_exclusion_mask: np.ndarray | None,
    mask_clean_kernel: int,
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
        if area_ratio > max_area_ratio:
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
    next_id = max([int(record.get("molmo_id", 0)) for record in mask_records] + [0]) + 1
    for item in grouped_items:
        mask = np.asarray(item["mask"], dtype=bool)
        if any(_mask_iou(mask, existing) > iou_threshold for existing in existing_masks):
            continue
        label = _proposal_label(image_np, mask)
        cx, cy = _mask_centroid_xy(mask)
        molmo_id = next_id + appended
        proposal_name = "_".join(str(index) for index in item["candidate_indices"][:8])
        filename = f"{molmo_id:03d}_sam2_group_{proposal_name}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(mask, mask_path)
        mask_records.append(
            {
                "node_id": len(mask_records),
                "molmo_id": molmo_id,
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


def generate_masks_with_sam2_molmo_langsam_pipeline(
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
    out_dir = output_mask_dir.parent
    label_path = out_dir / "label_1_sam2_auto.png"
    _draw_sam2_auto_label_image(image_path, candidates, label_path)
    parts_sheet_path = _save_sam2_rgb_parts_sheet(image_path, candidates, out_dir)
    review_error: str | None = None
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
        review_error = str(exc)
        raw_review = ""
        review_objects = []
        fallback_payload = {
            "model_id": review_model_id,
            "review_backend": "openai_responses",
            "image": {
                "path": str(image_path.resolve()),
                "sam2_label_path": str(label_path.resolve()),
                "sam2_rgb_parts_sheet_path": str(parts_sheet_path.resolve()),
            },
            "error": review_error,
            "objects": [],
            "fallback": "preserve_sam2_auto_candidates",
        }
        (out_dir / "molmo_sam2_review.json").write_text(json.dumps(fallback_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "openai_sam2_review.json").write_text(json.dumps(fallback_payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
                    "molmo_id": object_id,
                    "label": description,
                    "description": description,
                    "point": {"x": int(cx), "y": int(cy)},
                    "segmentation_backend": "sam2_auto_molmo_review_langsam",
                    "sam2_ids": item.get("sam2_ids", []),
                    "molmo_status": item.get("status"),
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
        {"stage": "openai_sam2_review", "objects": review_objects, "raw_model_output": raw_review, "error": review_error},
        {"stage": "sam2_unclaimed_preservation", "max_unclaimed": int(preserve_unclaimed_sam2), "claimed_sam2_ids": sorted(claimed_sam2_ids)},
    ]
    return mask_records, report


def _refine_anchor_mask_with_langsam(
    model: Any,
    image: Image.Image,
    point: MolmoPoint,
    points: list[MolmoPoint],
    anchor_mask: np.ndarray,
    description: str,
    mask_clean_kernel: int,
    background_exclusion_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any] | None, list[dict[str, Any]]]:
    prompt = _semantic_prompt(description)
    masks, scores, boxes = _langsam_predict(model, image, prompt)
    candidates: list[dict[str, Any]] = []
    anchor_area = max(1, int(np.count_nonzero(anchor_mask)))
    for idx, raw_mask in enumerate(masks):
        mask = _clean_mask(raw_mask, mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        contains_point = _point_inside_mask(mask, point, radius=4)
        other_points = _other_points_inside_mask(mask, point, points, radius=3)
        iou = _mask_iou(mask, anchor_mask)
        anchor_covered = _mask_overlap_fraction(anchor_mask, mask)
        semantic_inside_anchor = _mask_overlap_fraction(mask, anchor_mask)
        area_ratio = float(area / anchor_area)
        background_overlap = _background_overlap_fraction(mask, background_exclusion_mask)
        score = float(scores[idx]) if idx < len(scores) else 0.0
        accepted = (
            contains_point
            and not other_points
            and background_overlap < LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD
            and 0.45 <= area_ratio <= 2.2
            and (iou >= 0.25 or anchor_covered >= 0.55)
        )
        selection_score = (
            3.0 * score
            + 2.5 * iou
            + 1.5 * anchor_covered
            + (2.0 if contains_point else -2.0)
            - 2.0 * len(other_points)
            - max(0.0, area_ratio - 1.8)
            - background_overlap
        )
        candidates.append(
            {
                "candidate_index": int(idx),
                "box": boxes[idx] if idx < len(boxes) else None,
                "area": area,
                "area_ratio_to_anchor": area_ratio,
                "contains_point": bool(contains_point),
                "other_points_in_mask": [
                    {"molmo_id": other.molmo_id, "x": other.x, "y": other.y, "label": other.label}
                    for other in other_points
                ],
                "iou_with_anchor": float(iou),
                "anchor_covered": float(anchor_covered),
                "semantic_inside_anchor": float(semantic_inside_anchor),
                "background_exclusion_overlap": float(background_overlap),
                "semantic_score": score,
                "accepted": bool(accepted),
                "selection_score": float(selection_score),
                "mask": mask,
            }
        )

    accepted_candidates = [candidate for candidate in candidates if candidate["accepted"]]
    metadata_candidates = [{key: value for key, value in candidate.items() if key != "mask"} for candidate in candidates]
    if not accepted_candidates:
        return anchor_mask, None, metadata_candidates

    best = max(accepted_candidates, key=lambda item: item["selection_score"])
    selected_metadata = {key: value for key, value in best.items() if key != "mask"}
    selected_metadata["semantic_prompt"] = prompt
    return np.asarray(best["mask"], dtype=bool), selected_metadata, metadata_candidates


def generate_masks_with_sam2_anchor_pipeline(
    image_path: Path,
    points: list[MolmoPoint],
    depth_map: np.ndarray,
    output_mask_dir: Path,
    min_area_ratio: float = 0.006,
    max_area_ratio: float = 0.11,
    border_fraction_threshold: float = 0.18,
    mask_clean_kernel: int = 3,
    save_candidates: bool = False,
    device: str | None = None,
    background_exclusion_mask: np.ndarray | None = None,
    anchor_merge_depth_threshold: float = 0.015,
    anchor_merge_boundary_kernel: int = 3,
    anchor_merge_min_contact_pixels: int = 8,
    refine_with_langsam: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, proposal_report, model, image = _sam2_auto_candidate_pool(
        image_path=image_path,
        output_mask_dir=output_mask_dir,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        border_fraction_threshold=border_fraction_threshold,
        mask_clean_kernel=mask_clean_kernel,
        save_candidates=save_candidates,
        device=device,
        background_exclusion_mask=background_exclusion_mask,
        points_per_side=None,
        crop_n_layers=None,
        pred_iou_thresh=None,
        stability_score_thresh=None,
    )
    image_np = np.asarray(image)
    mask_records: list[dict[str, Any]] = []
    anchor_report: list[dict[str, Any]] = [
        {"stage": "sam2_auto_pool", "candidates": proposal_report[:200], "candidate_count": len(candidates)}
    ]

    for point in points:
        text_box: list[int] | None = None
        selected_text_box: dict[str, Any] | None = None
        text_box_candidates: list[dict[str, Any]] = []
        text_box_error: str | None = None
        try:
            text_box, selected_text_box, text_box_candidates = _detect_langsam_text_box(model, image, point, points)
        except Exception as exc:
            text_box_error = str(exc)

        selected_parts, group_report = _anchor_group_candidates(
            candidates=candidates,
            point=point,
            points=points,
            depth_map=depth_map,
            text_box=text_box,
            depth_threshold=anchor_merge_depth_threshold,
            boundary_kernel_size=anchor_merge_boundary_kernel,
            min_contact_pixels=anchor_merge_min_contact_pixels,
        )
        if not selected_parts:
            anchor_report.append(
                {
                    "molmo_id": int(point.molmo_id),
                    "label": point.label,
                    "status": "no_anchor_group",
                    "text_box": selected_text_box,
                    "text_box_error": text_box_error,
                    "group_report": group_report[:100],
                }
            )
            continue

        anchor_mask = np.any(
            np.stack([np.asarray(part["mask"], dtype=bool) for part in selected_parts], axis=0),
            axis=0,
        )
        anchor_mask = _clean_mask(anchor_mask, mask_clean_kernel)
        if not _point_inside_mask(anchor_mask, point, radius=4):
            anchor_report.append(
                {
                    "molmo_id": int(point.molmo_id),
                    "label": point.label,
                    "status": "anchor_mask_misses_point",
                    "selected_parts": [_candidate_summary(part) for part in selected_parts],
                }
            )
            continue

        other_points_in_anchor = _other_points_inside_mask(anchor_mask, point, points, radius=3)
        background_overlap = _background_overlap_fraction(anchor_mask, background_exclusion_mask)
        if other_points_in_anchor or background_overlap >= LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD:
            anchor_report.append(
                {
                    "molmo_id": int(point.molmo_id),
                    "label": point.label,
                    "status": "anchor_mask_rejected",
                    "other_points_in_anchor": [
                        {"molmo_id": other.molmo_id, "x": other.x, "y": other.y, "label": other.label}
                        for other in other_points_in_anchor
                    ],
                    "background_exclusion_overlap": float(background_overlap),
                    "selected_parts": [_candidate_summary(part) for part in selected_parts],
                }
            )
            continue

        description = _anchor_description(point, image_np, anchor_mask)
        refined_mask = anchor_mask
        selected_refinement: dict[str, Any] | None = None
        refinement_candidates: list[dict[str, Any]] = []
        if refine_with_langsam:
            refined_mask, selected_refinement, refinement_candidates = _refine_anchor_mask_with_langsam(
                model=model,
                image=image,
                point=point,
                points=points,
                anchor_mask=anchor_mask,
                description=description,
                mask_clean_kernel=mask_clean_kernel,
                background_exclusion_mask=background_exclusion_mask,
            )

        final_mask = _clean_mask(refined_mask, mask_clean_kernel)
        filename = f"{point.molmo_id:03d}_anchor_{_safe_label(point.label)}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(final_mask, mask_path)
        cx, cy = _mask_centroid_xy(final_mask)
        mask_records.append(
            {
                "node_id": len(mask_records),
                "molmo_id": point.molmo_id,
                "label": point.label,
                "description": description,
                "point": {"x": int(cx), "y": int(cy)},
                "molmo_anchor_point": {"x": int(point.x), "y": int(point.y)},
                "segmentation_backend": (
                    "sam2_auto_molmo_anchor_langsam_refine"
                    if selected_refinement is not None
                    else "sam2_auto_molmo_anchor"
                ),
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(final_mask)),
                "anchor_text_box": selected_text_box,
                "text_box_candidates": text_box_candidates,
                "text_box_error": text_box_error,
                "anchor_sam2_parts": [_candidate_summary(part) for part in selected_parts],
                "anchor_group_report": group_report[:100],
                "anchor_background_overlap": float(background_overlap),
                "selected_langsam_refinement": selected_refinement,
                "langsam_refinement_candidates": refinement_candidates,
                "mask_array": final_mask,
            }
        )
        anchor_report.append(
            {
                "molmo_id": int(point.molmo_id),
                "label": point.label,
                "status": "accepted",
                "part_count": len(selected_parts),
                "description": description,
                "selected_refinement": selected_refinement,
            }
        )

    return mask_records, anchor_report


def complete_masks_with_sam2_auto_proposals(
    image_path: Path,
    mask_records: list[dict[str, Any]],
    output_mask_dir: Path,
    min_area_ratio: float = 0.006,
    max_area_ratio: float = 0.11,
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.6,
    border_fraction_threshold: float = 0.18,
    max_masks: int = 3,
    mask_clean_kernel: int = 3,
    save_candidates: bool = False,
    device: str | None = None,
    reserved_molmo_ids: set[int] | None = None,
    background_exclusion_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_masks <= 0:
        return mask_records, []
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    image_area = float(image_np.shape[0] * image_np.shape[1])
    existing_masks = [np.asarray(record["mask_array"], dtype=bool) for record in mask_records]
    existing_union = np.any(np.stack(existing_masks, axis=0), axis=0) if existing_masks else np.zeros(image_np.shape[:2], dtype=bool)
    existing_points = [
        (int(record.get("point", {}).get("x", -1)), int(record.get("point", {}).get("y", -1)))
        for record in mask_records
    ]

    model = _load_langsam(device)
    raw_proposals = _sam2_auto_generate(model, image)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, proposal in enumerate(raw_proposals):
        raw_mask = proposal.get("segmentation")
        if raw_mask is None:
            continue
        mask = _clean_mask(_as_numpy_mask(raw_mask), mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        area_ratio = float(area / image_area)
        border_fraction = _border_touch_fraction(mask)
        max_iou = max((_mask_iou(mask, existing) for existing in existing_masks), default=0.0)
        containment = _mask_overlap_fraction(mask, existing_union)
        cx, cy = _mask_centroid_xy(mask)
        contains_existing_point = any(_point_inside_xy(mask, x, y) for x, y in existing_points if x >= 0 and y >= 0)
        background_overlap = 0.0
        has_background_exclusion = background_exclusion_mask is not None and np.count_nonzero(background_exclusion_mask) > 0
        if has_background_exclusion:
            background_overlap = float(np.count_nonzero(mask & background_exclusion_mask) / max(1, area))

        reason: str | None = None
        if area_ratio < min_area_ratio:
            reason = "too_small"
        elif area_ratio > max_area_ratio:
            reason = "too_large"
        elif border_fraction > border_fraction_threshold:
            reason = "touches_image_border"
        elif _is_support_like_horizontal_strip(mask) or _is_tray_or_background_like_proposal(image_np, mask, background_exclusion_mask):
            reason = "support_or_tray_like"
        elif max_iou > iou_threshold:
            reason = "duplicates_existing_mask"
        elif containment > containment_threshold:
            reason = "mostly_inside_existing_mask"
        elif contains_existing_point:
            reason = "contains_existing_molmo_point"
        elif background_overlap > 0.5:
            reason = "overlaps_background_exclusion"

        metadata = {
            "proposal_index": idx,
            "area": area,
            "area_ratio": area_ratio,
            "bbox": [int(value) for value in proposal.get("bbox", _mask_bbox(mask))],
            "predicted_iou": float(proposal.get("predicted_iou", 0.0) or 0.0),
            "stability_score": float(proposal.get("stability_score", 0.0) or 0.0),
            "border_fraction": border_fraction,
            "max_existing_iou": max_iou,
            "existing_containment": containment,
            "background_exclusion_overlap": background_overlap,
            "centroid": {"x": cx, "y": cy},
        }
        if reason:
            metadata["rejection_reason"] = reason
            rejected.append(metadata)
            continue

        metadata["selection_score"] = _proposal_score(proposal, area_ratio, border_fraction)
        metadata["mask"] = mask
        candidates.append(metadata)

    candidates = _cluster_proposal_candidates(candidates, image_area, max_area_ratio)
    selected: list[dict[str, Any]] = []
    selected_masks: list[np.ndarray] = []
    reserved_ids = reserved_molmo_ids or set()
    next_molmo_id = max([int(record.get("molmo_id", 0)) for record in mask_records] + list(reserved_ids) + [0]) + 1

    for candidate in candidates:
        mask = candidate["mask"]
        if any(_mask_iou(mask, selected_mask) > iou_threshold for selected_mask in selected_masks):
            rejected.append(
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "mask"
                }
                | {"rejection_reason": "duplicates_selected_proposal"}
            )
            continue

        label = _proposal_label(image_np, mask)
        molmo_id = next_molmo_id + len(selected)
        filename = f"{molmo_id:03d}_sam2_{_safe_label(label)}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(mask, mask_path)
        cx = int(candidate["centroid"]["x"])
        cy = int(candidate["centroid"]["y"])
        record_metadata = {key: value for key, value in candidate.items() if key != "mask"}
        mask_records.append(
            {
                "node_id": len(mask_records),
                "molmo_id": molmo_id,
                "label": _sanitize_label(label).replace("_", " "),
                "point": {"x": cx, "y": cy},
                "segmentation_backend": "sam2_auto_proposal",
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(mask)),
                "proposal_metadata": record_metadata,
                "mask_array": mask,
            }
        )
        selected.append(record_metadata)
        selected_masks.append(mask)
        if len(selected) >= max_masks:
            break

    if save_candidates:
        candidate_dir = output_mask_dir.parent / "sam2_auto_candidates"
        for candidate in candidates:
            mask = candidate["mask"]
            proposal_name = "_".join(str(index) for index in candidate.get("proposal_indices", [candidate["proposal_index"]]))
            candidate_path = candidate_dir / f"proposal_{proposal_name}.png"
            _save_mask_png(mask, candidate_path)
            candidate["mask_path"] = str(candidate_path.resolve())

    proposal_report = [
        {key: value for key, value in item.items() if key != "mask"}
        for item in candidates
    ]
    proposal_report.extend(rejected[:100])
    return mask_records, proposal_report


def generate_masks_with_langsam(
    image_path: Path,
    points: list[MolmoPoint],
    output_mask_dir: Path,
    sam_model_id: str = "facebook/sam-vit-base",
    sam_point_grid_radius: int = 0,
    sam_prompt_mode: str = "cross",
    sam_negative_points: int = 0,
    mask_clean_kernel: int = 3,
    save_candidates: bool = False,
    device: str | None = None,
    background_exclusion_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_langsam(device)

    mask_records: list[dict[str, Any]] = []
    for point in points:
        prompt = _semantic_prompt(point.label)
        previous_masks = [np.asarray(record["mask_array"], dtype=bool) for record in mask_records]
        masks, scores, boxes = _langsam_predict(model, image, prompt)
        best_mask, selected_candidate, candidate_metadata = _select_langsam_mask(
            masks,
            scores,
            boxes,
            point,
            points,
            mask_clean_kernel,
            previous_masks=previous_masks,
        )
        if best_mask is None:
            raise ValueError(f"LangSAM returned no masks for point {point.molmo_id} prompt={prompt!r}.")

        max_previous_iou = max((_mask_iou(best_mask, previous) for previous in previous_masks), default=0.0)
        point_hit = _point_inside_mask(best_mask, point)
        other_points_in_best_mask = _other_points_inside_mask(best_mask, point, points)
        semantic_background_overlap = _background_overlap_fraction(best_mask, background_exclusion_mask)
        fallback_reason: str | None = None
        if not point_hit:
            fallback_reason = "semantic_mask_misses_point"
        elif max_previous_iou > 0.3:
            fallback_reason = "semantic_mask_duplicates_previous_instance"
        elif len(other_points_in_best_mask) > LANGSAM_OTHER_POINT_FALLBACK_THRESHOLD:
            fallback_reason = "semantic_mask_contains_other_instances"
        elif semantic_background_overlap >= LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD:
            fallback_reason = "semantic_mask_overlaps_background"

        fallback_record: dict[str, Any] | None = None
        fallback_accepted = False
        if fallback_reason is not None:
            fallback_records = generate_masks_with_sam(
                image_path=image_path,
                points=points,
                target_points=[point],
                output_mask_dir=output_mask_dir,
                sam_model_id=sam_model_id,
                point_grid_radius=sam_point_grid_radius,
                prompt_mode=sam_prompt_mode,
                negative_points=max(sam_negative_points, 8),
                mask_clean_kernel=mask_clean_kernel,
                save_candidates=save_candidates,
                device=device,
                use_text_box_prompt=False,
            )
            fallback_record = fallback_records[0]
            fallback_mask = np.asarray(fallback_record["mask_array"], dtype=bool)
            fallback_hit = _point_inside_mask(fallback_mask, point)
            fallback_max_previous_iou = max((_mask_iou(fallback_mask, previous) for previous in previous_masks), default=0.0)
            fallback_background_overlap = _background_overlap_fraction(fallback_mask, background_exclusion_mask)
            if (
                fallback_hit
                and fallback_max_previous_iou <= 0.3
                and fallback_background_overlap <= LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD
            ):
                best_mask = fallback_mask
                semantic_background_overlap = fallback_background_overlap
                fallback_record["background_exclusion_overlap"] = float(fallback_background_overlap)
                fallback_accepted = True
            else:
                fallback_record = None
        if fallback_reason is not None and not fallback_accepted:
            continue

        # Drop mask if it does not contain its own Molmo point after all fallbacks
        final_point_hit = _point_inside_mask(best_mask, point)
        if not final_point_hit:
            continue

        filename = f"{point.molmo_id:03d}_molmo_{_safe_label(point.label)}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(best_mask, mask_path)

        if save_candidates:
            candidate_dir = output_mask_dir.parent / "langsam_candidates"
            for candidate in candidate_metadata:
                candidate_idx = int(candidate["candidate_index"])
                candidate_path = (
                    candidate_dir
                    / f"mask_{point.molmo_id:03d}_{_safe_label(point.label)}_cand_{candidate_idx}.png"
                )
                _save_mask_png(masks[candidate_idx], candidate_path)
                candidate["mask_path"] = str(candidate_path.resolve())

        mask_records.append(
            {
                "node_id": len(mask_records),
                "molmo_id": point.molmo_id,
                "label": point.label,
                "point": {"x": point.x, "y": point.y},
                "segmentation_backend": "langsam_sam_fallback" if fallback_record is not None else "langsam",
                "semantic_prompt": prompt,
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(best_mask)),
                "selected_semantic_candidate": selected_candidate,
                "semantic_candidates": candidate_metadata,
                "semantic_point_hit": bool(point_hit),
                "semantic_max_previous_iou": float(max_previous_iou),
                "semantic_background_overlap": float(semantic_background_overlap),
                "fallback_reason": fallback_reason,
                "fallback_sam_record": (
                    {key: value for key, value in fallback_record.items() if key != "mask_array"}
                    if fallback_record is not None
                    else None
                ),
                "mask_array": best_mask,
            }
        )

    return mask_records


def generate_masks_with_sam(
    image_path: Path,
    points: list[MolmoPoint],
    output_mask_dir: Path,
    sam_model_id: str,
    target_points: list[MolmoPoint] | None = None,
    point_grid_radius: int = 0,
    prompt_mode: str = "cross",
    negative_points: int = 0,
    mask_clean_kernel: int = 3,
    save_candidates: bool = False,
    device: str | None = None,
    use_text_box_prompt: bool = True,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = _load_sam(sam_model_id, device)
    langsam_model: Any | None = None
    if use_text_box_prompt:
        try:
            langsam_model = _load_langsam(device)
        except Exception as exc:
            print(f"LangSAM text-box prompt unavailable; using point-only SAM prompts: {exc}", file=sys.stderr, flush=True)

    mask_records: list[dict[str, Any]] = []
    points_to_segment = target_points or points
    for point in points_to_segment:
        text_box: list[int] | None = None
        selected_text_box: dict[str, Any] | None = None
        text_box_candidates: list[dict[str, Any]] = []
        text_box_error: str | None = None
        if langsam_model is not None:
            try:
                text_box, selected_text_box, text_box_candidates = _detect_langsam_text_box(
                    langsam_model,
                    image,
                    point,
                    points,
                )
            except Exception as exc:
                text_box_error = str(exc)

        variants = _prompt_variants(
            target=point,
            points=points,
            width=width,
            height=height,
            radius=point_grid_radius,
            mode=prompt_mode,
            negative_points=negative_points,
        )
        all_candidates: list[dict[str, Any]] = []
        best_mask: np.ndarray | None = None
        selected_candidate: dict[str, Any] | None = None
        selected_positive_points: list[list[int]] = []
        selected_negative_points: list[list[int]] = []

        for variant_idx, variant in enumerate(variants):
            input_points = variant["input_points"]
            input_labels = variant["input_labels"]
            positive_points = variant["positive_points"]
            processor_kwargs: dict[str, Any] = {
                "input_points": [[input_points]],
                "input_labels": [[input_labels]],
                "return_tensors": "pt",
            }
            if text_box is not None:
                processor_kwargs["input_boxes"] = [[text_box]]
            inputs = processor(image, **processor_kwargs)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, multimask_output=True)

            processed_masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
            )[0]
            masks_np = _normalize_processed_masks(processed_masks)
            variant_best_mask, variant_selected, variant_candidates = _select_best_mask(
                masks_np,
                outputs.iou_scores[0],
                positive_points,
                mask_clean_kernel,
                negative_points=variant["negative_prompt_points"],
                prompt_box=text_box,
                prompt_metadata={
                    "variant_index": variant_idx,
                    "prompt_mode": variant["mode"],
                    "prompt_radius": int(variant["radius"]),
                    "prompt_negative_points": int(variant["negative_points"]),
                    "prompt_box": text_box,
                },
            )

            if save_candidates:
                candidate_dir = output_mask_dir.parent / "mask_candidates"
                for candidate in variant_candidates:
                    candidate_idx = int(candidate["candidate_index"])
                    candidate_mask = _clean_mask(masks_np[candidate_idx], mask_clean_kernel)
                    candidate_path = (
                        candidate_dir
                        / f"mask_{point.molmo_id:03d}_{_safe_label(point.label)}_var_{variant_idx}_cand_{candidate_idx}.png"
                    )
                    _save_mask_png(candidate_mask, candidate_path)
                    candidate["mask_path"] = str(candidate_path.resolve())

            all_candidates.extend(variant_candidates)
            if selected_candidate is None or variant_selected["selection_score"] > selected_candidate["selection_score"]:
                best_mask = variant_best_mask
                selected_candidate = variant_selected
                selected_positive_points = variant["positive_points"]
                selected_negative_points = variant["negative_prompt_points"]

        if best_mask is None or selected_candidate is None:
            raise ValueError(f"SAM did not return a usable mask for point {point.molmo_id}.")

        filename = f"{point.molmo_id:03d}_molmo_{_safe_label(point.label)}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(best_mask, mask_path)

        mask_records.append(
            {
                "node_id": len(mask_records),
                "molmo_id": point.molmo_id,
                "label": point.label,
                "point": {"x": point.x, "y": point.y},
                "segmentation_backend": "sam",
                "sam_prompt_mode": selected_candidate["prompt_mode"],
                "sam_prompt_radius": int(selected_candidate["prompt_radius"]),
                "sam_positive_points": [{"x": int(x), "y": int(y)} for x, y in selected_positive_points],
                "sam_negative_points": [{"x": int(x), "y": int(y)} for x, y in selected_negative_points],
                "sam_prompt_box": text_box,
                "text_box_backend": "langsam_grounding_dino" if text_box is not None else "none",
                "selected_text_box": selected_text_box,
                "text_box_candidates": text_box_candidates,
                "text_box_error": text_box_error,
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(best_mask)),
                "predicted_iou": float(selected_candidate["predicted_iou"]),
                "selected_sam_candidate": selected_candidate,
                "sam_candidates": all_candidates,
                "mask_array": best_mask,
            }
        )

    return mask_records


def build_org_json(
    points_json_path: Path,
    depth_path: Path,
    output_json_path: Path,
    output_mask_dir: Path,
    segmentation_backend: str = "sam2-molmo-langsam",
    sam_model_id: str = "facebook/sam-vit-base",
    molmo_model_id: str = "allenai/Molmo-7B-D-0924",
    review_model_id: str = "gpt-5.5",
    review_api_key_env: str = "OPENAI_API_KEY",
    review_base_url: str | None = None,
    review_timeout: float = 120.0,
    epsilon: float = 0.05,
    kernel_size: int = 5,
    min_contact_pixels: int = 50,
    min_contact_ratio: float = 0.002,
    sam_point_grid_radius: int = 0,
    sam_prompt_mode: str = "cross",
    sam_negative_points: int = 0,
    mask_clean_kernel: int = 3,
    proposal_backend: str = "sam2-auto",
    proposal_min_area_ratio: float = 0.006,
    proposal_max_area_ratio: float = 0.11,
    proposal_iou_threshold: float = 0.3,
    proposal_containment_threshold: float = 0.6,
    proposal_border_fraction_threshold: float = 0.18,
    max_proposal_masks: int = 3,
    save_candidates: bool = False,
    device: str | None = None,
    use_text_box_prompt: bool = True,
    depth_merge_threshold: float = 0.0,
    depth_merge_boundary_kernel: int = 3,
    depth_merge_min_contact_pixels: int = 8,
    anchor_merge_depth_threshold: float = 0.015,
    anchor_refine_with_langsam: bool = True,
    sam2_points_per_side: int | None = 24,
    sam2_crop_n_layers: int | None = 0,
    sam2_pred_iou_thresh: float | None = 0.7,
    sam2_stability_score_thresh: float | None = 0.88,
    preserve_unclaimed_sam2: int = 24,
) -> dict[str, Any]:
    _prepare_mask_output_dir(output_mask_dir, save_candidates)
    if segmentation_backend == "sam2-molmo-langsam":
        points_payload = _load_json(points_json_path)
        image_path_value = points_payload.get("image", {}).get("path")
        if not image_path_value:
            raise ValueError(f"Missing image.path in {points_json_path}.")
        image_path = _resolve_path(points_json_path, str(image_path_value))
        points: list[MolmoPoint] = []
        filtered_points: list[dict[str, Any]] = []
        raw_molmo_ids: set[int] = set()
    else:
        points_payload, points, image_path = _load_points(points_json_path)
        raw_molmo_ids = {int(point.molmo_id) for point in points}
        points, filtered_points = _filter_points(points)
        if not points:
            raise ValueError("No Molmo points are available for mask generation.")
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

    effective_backend = segmentation_backend
    anchor_report: list[dict[str, Any]] = []
    if segmentation_backend == "sam2-molmo-langsam":
        mask_records, anchor_report = generate_masks_with_sam2_molmo_langsam_pipeline(
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
        effective_backend = "sam2-molmo-langsam"
    elif segmentation_backend == "sam2-anchor":
        mask_records, anchor_report = generate_masks_with_sam2_anchor_pipeline(
            image_path=image_path,
            points=points,
            depth_map=depth_map,
            output_mask_dir=output_mask_dir,
            min_area_ratio=proposal_min_area_ratio,
            max_area_ratio=proposal_max_area_ratio,
            border_fraction_threshold=proposal_border_fraction_threshold,
            mask_clean_kernel=mask_clean_kernel,
            save_candidates=save_candidates,
            device=device,
            background_exclusion_mask=background_exclusion_mask,
            anchor_merge_depth_threshold=anchor_merge_depth_threshold,
            anchor_merge_boundary_kernel=depth_merge_boundary_kernel,
            anchor_merge_min_contact_pixels=depth_merge_min_contact_pixels,
            refine_with_langsam=anchor_refine_with_langsam,
        )
        effective_backend = "sam2-anchor"
    elif segmentation_backend in {"langsam", "auto"}:
        try:
            mask_records = generate_masks_with_langsam(
                image_path=image_path,
                points=points,
                output_mask_dir=output_mask_dir,
                sam_model_id=sam_model_id,
                sam_point_grid_radius=sam_point_grid_radius,
                sam_prompt_mode=sam_prompt_mode,
                sam_negative_points=sam_negative_points,
                mask_clean_kernel=mask_clean_kernel,
                save_candidates=save_candidates,
                device=device,
                background_exclusion_mask=background_exclusion_mask,
            )
            effective_backend = "langsam"
        except Exception as exc:
            if segmentation_backend == "langsam":
                raise
            print(f"LangSAM unavailable or failed; falling back to SAM prompts: {exc}", file=sys.stderr, flush=True)
            mask_records = generate_masks_with_sam(
                image_path=image_path,
                points=points,
                output_mask_dir=output_mask_dir,
                sam_model_id=sam_model_id,
                point_grid_radius=sam_point_grid_radius,
                prompt_mode=sam_prompt_mode,
                negative_points=sam_negative_points,
                mask_clean_kernel=mask_clean_kernel,
                save_candidates=save_candidates,
                device=device,
                use_text_box_prompt=use_text_box_prompt,
            )
            effective_backend = "sam"
    else:
        mask_records = generate_masks_with_sam(
            image_path=image_path,
            points=points,
            output_mask_dir=output_mask_dir,
            sam_model_id=sam_model_id,
            point_grid_radius=sam_point_grid_radius,
            prompt_mode=sam_prompt_mode,
            negative_points=sam_negative_points,
            mask_clean_kernel=mask_clean_kernel,
            save_candidates=save_candidates,
            device=device,
            use_text_box_prompt=use_text_box_prompt,
        )

    duplicate_mask_report: list[dict[str, Any]] = []
    if segmentation_backend != "sam2-molmo-langsam":
        mask_records, duplicate_mask_report = _drop_contained_duplicate_masks(
            mask_records,
        )
    mask_records = _renumber_masks(mask_records, output_mask_dir)

    # --- first-stage label image (after LangSAM/SAM dedup) ---
    _draw_mask_records_label(
        image_path=image_path,
        mask_records=mask_records,
        out_path=output_mask_dir.parent / "label_2_langsam.png",
    )

    # --- background exclusion mask (depth tray-bottom plane) ---
    if proposal_backend == "sam2-auto":
        if background_exclusion_mask is not None and np.count_nonzero(background_exclusion_mask) > 0:
            _save_mask_png(background_exclusion_mask, output_mask_dir / "000_background.png")

    proposal_report: list[dict[str, Any]] = []
    effective_proposal_backend = "none"
    if segmentation_backend == "sam2-molmo-langsam":
        effective_proposal_backend = "sam2-auto-initial-reviewed-by-molmo"
    elif segmentation_backend == "sam2-anchor":
        effective_proposal_backend = "sam2-auto-anchor-pool"
    elif proposal_backend == "sam2-auto":
        try:
            before_count = len(mask_records)
            mask_records, proposal_report = complete_masks_with_sam2_auto_proposals(
                image_path=image_path,
                mask_records=mask_records,
                output_mask_dir=output_mask_dir,
                min_area_ratio=proposal_min_area_ratio,
                max_area_ratio=proposal_max_area_ratio,
                iou_threshold=proposal_iou_threshold,
                containment_threshold=proposal_containment_threshold,
                border_fraction_threshold=proposal_border_fraction_threshold,
                max_masks=max_proposal_masks,
                mask_clean_kernel=mask_clean_kernel,
                save_candidates=save_candidates,
                device=device,
                reserved_molmo_ids=raw_molmo_ids,
                background_exclusion_mask=background_exclusion_mask,
            )
            if len(mask_records) > before_count:
                effective_proposal_backend = "sam2-auto"
        except Exception as exc:
            print(f"SAM2 automatic proposal completion failed; continuing with prompted masks: {exc}", file=sys.stderr, flush=True)
            effective_proposal_backend = "sam2-auto-failed"
    elif proposal_backend != "none":
        raise ValueError(f"Unsupported proposal backend: {proposal_backend}")

    proposal_duplicate_report: list[dict[str, Any]] = []
    if segmentation_backend != "sam2-molmo-langsam":
        mask_records, proposal_duplicate_report = _drop_contained_duplicate_masks(
            mask_records,
        )
    mask_records = _renumber_masks(mask_records, output_mask_dir)
    duplicate_mask_report.extend(proposal_duplicate_report)

    depth_merge_report: list[dict[str, Any]] = []
    if depth_merge_threshold > 0:
        mask_records, depth_merge_report = _merge_mask_records_by_depth(
            mask_records=mask_records,
            depth_map=depth_map,
            output_mask_dir=output_mask_dir,
            depth_threshold=depth_merge_threshold,
            boundary_kernel_size=depth_merge_boundary_kernel,
            min_contact_pixels=depth_merge_min_contact_pixels,
        )
        mask_records = _renumber_masks(mask_records, output_mask_dir)

    final_mask_quality_report: dict[str, Any] = {}
    mask_records, final_mask_quality_report = _finalize_independent_scene_masks(
        mask_records=mask_records,
        output_mask_dir=output_mask_dir,
        background_exclusion_mask=background_exclusion_mask,
        image_shape=tuple(depth_map.shape),
    )
    mask_records = _renumber_masks(mask_records, output_mask_dir)
    duplicate_mask_report.extend(final_mask_quality_report.get("dedup_report", []))
    if background_exclusion_mask is not None and np.count_nonzero(background_exclusion_mask) > 0:
        foreground_union = np.any(
            np.stack([np.asarray(record["mask_array"], dtype=bool) for record in mask_records], axis=0),
            axis=0,
        ) if mask_records else np.zeros_like(background_exclusion_mask, dtype=bool)
        _save_mask_png(np.asarray(background_exclusion_mask, dtype=bool) & ~foreground_union, output_mask_dir / "000_background.png")

    # --- second-stage label image (after proposal dedup) ---
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
    payload = {
        "image": {
            "path": str(image_path.resolve()),
            "width": int(points_payload.get("image", {}).get("width", masks.shape[2])),
            "height": int(points_payload.get("image", {}).get("height", masks.shape[1])),
        },
        "depth_map": {
            "path": str(depth_path.resolve()),
            "shape": [int(depth_map.shape[0]), int(depth_map.shape[1])],
        },
        "points_source": str(points_json_path.resolve()),
        "molmo_points": points_payload.get("points", []),
        "filtered_molmo_points": filtered_points,
        "segmentation_backend": effective_backend,
        "proposal_backend": effective_proposal_backend,
        "duplicate_mask_report": duplicate_mask_report,
        "proposal_report": proposal_report,
        "anchor_report": anchor_report,
        "depth_merge_report": depth_merge_report,
        "depth_merge_threshold": float(depth_merge_threshold),
        "anchor_merge_depth_threshold": float(anchor_merge_depth_threshold),
        "anchor_refine_with_langsam": bool(anchor_refine_with_langsam),
        "final_mask_quality_report": final_mask_quality_report,
        "save_candidates": bool(save_candidates),
        "sam_model_id": sam_model_id,
        "sam_prompt_mode": sam_prompt_mode,
        "sam_negative_points": int(sam_negative_points),
        "use_text_box_prompt": bool(use_text_box_prompt),
        "graph": graph_payload,
    }

    for edge in payload["graph"]["edges"]:
        source_node = node_records[edge["source"]]
        target_node = node_records[edge["target"]]
        edge["source_molmo_id"] = int(source_node["molmo_id"])
        edge["target_molmo_id"] = int(target_node["molmo_id"])
        edge["source_label"] = str(source_node["label"])
        edge["target_label"] = str(target_node["label"])

    _write_json(output_json_path, payload)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SAM masks from Molmo points and build an occlusion graph JSON.",
    )
    parser.add_argument(
        "--points-json",
        default=str(DEFAULT_MOLMO_OUT / "molmo_points.json"),
        help="Path to Molmo points JSON.",
    )
    parser.add_argument(
        "--depth-map",
        required=True,
        help="Path to the depth map (.npy, .npz, or image file).",
    )
    parser.add_argument(
        "--mask-dir",
        default=str(DEFAULT_MOLMO_OUT / "mask"),
        help="Directory where per-object mask PNGs will be written.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_MOLMO_OUT / "occlusion_graph.json"),
        help="Path to the final occlusion graph JSON file.",
    )
    parser.add_argument(
        "--segmentation-backend",
        choices=["sam2-molmo-langsam", "sam2-anchor", "sam", "langsam", "auto"],
        default="sam2-molmo-langsam",
        help="Mask generator: SAM2-auto -> Molmo review -> LangSAM, legacy SAM2-anchor, SAM point prompts, LangSAM semantic prompts, or LangSAM with SAM fallback.",
    )
    parser.add_argument(
        "--molmo-model-id",
        default="allenai/Molmo-7B-D-0924",
        help="Hugging Face Molmo model id used by legacy Molmo-point paths.",
    )
    parser.add_argument("--review-model-id", default="gpt-5.5")
    parser.add_argument("--review-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--review-base-url", default=None)
    parser.add_argument("--review-timeout", type=float, default=120.0)
    parser.add_argument(
        "--sam-model-id",
        default="facebook/sam-vit-base",
        help="Hugging Face SAM model id.",
    )
    parser.add_argument("--epsilon", type=float, default=0.05, help="Depth margin for occlusion decisions.")
    parser.add_argument("--kernel-size", type=int, default=5, help="Dilation kernel size.")
    parser.add_argument(
        "--min-contact-pixels",
        type=int,
        default=50,
        help="Ignore contact areas smaller than this many pixels.",
    )
    parser.add_argument(
        "--min-contact-ratio",
        type=float,
        default=0.002,
        help="Ignore contacts smaller than this fraction of the smaller object mask.",
    )
    parser.add_argument(
        "--sam-point-grid-radius",
        type=int,
        default=0,
        help="Pixel radius used by SAM multi-point positive prompts.",
    )
    parser.add_argument(
        "--sam-prompt-mode",
        choices=["cross", "grid", "ring", "auto"],
        default="cross",
        help="Positive SAM prompt layout around each Molmo point.",
    )
    parser.add_argument(
        "--sam-negative-points",
        type=int,
        default=0,
        help="Use this many nearest other Molmo points as negative SAM prompts.",
    )
    parser.add_argument(
        "--no-text-box-prompt",
        action="store_true",
        help="Disable LangSAM/GroundingDINO text-to-box prompts before SAM mask extraction.",
    )
    parser.add_argument(
        "--mask-clean-kernel",
        type=int,
        default=3,
        help="Morphological cleanup kernel for SAM masks; use 1 to disable.",
    )
    parser.add_argument(
        "--proposal-backend",
        choices=["none", "sam2-auto"],
        default="sam2-auto",
        help="Optionally add SAM2 automatic masks for missed objects after Molmo/LangSAM masks.",
    )
    parser.add_argument(
        "--proposal-min-area-ratio",
        type=float,
        default=0.006,
        help="Minimum image area ratio for SAM2 automatic proposal masks.",
    )
    parser.add_argument(
        "--proposal-max-area-ratio",
        type=float,
        default=0.11,
        help="Maximum image area ratio for SAM2 automatic proposal masks.",
    )
    parser.add_argument(
        "--proposal-iou-threshold",
        type=float,
        default=0.3,
        help="Reject proposals whose IoU with an existing/selected mask exceeds this value.",
    )
    parser.add_argument(
        "--proposal-containment-threshold",
        type=float,
        default=0.6,
        help="Reject proposals mostly covered by already selected masks.",
    )
    parser.add_argument(
        "--proposal-border-fraction-threshold",
        type=float,
        default=0.18,
        help="Reject proposals with too much mask area touching the image border.",
    )
    parser.add_argument(
        "--max-proposal-masks",
        type=int,
        default=3,
        help="Maximum number of SAM2 automatic proposal masks to add.",
    )
    parser.add_argument(
        "--depth-merge-threshold",
        type=float,
        default=0.0,
        help="Legacy global merge for adjacent final masks; disabled by default. Use only for debugging.",
    )
    parser.add_argument(
        "--anchor-merge-depth-threshold",
        type=float,
        default=0.015,
        help="Within each Molmo anchor, merge adjacent SAM2-auto parts when boundary median depth differs by at most this many meters.",
    )
    parser.add_argument(
        "--depth-merge-boundary-kernel",
        type=int,
        default=3,
        help="Dilation kernel size used to find touching mask boundaries for depth-based merging.",
    )
    parser.add_argument(
        "--depth-merge-min-contact-pixels",
        type=int,
        default=8,
        help="Minimum boundary pixels required before depth-based merging can join a mask pair.",
    )
    parser.add_argument(
        "--save-candidates",
        action="store_true",
        help="Save intermediate candidate masks for debugging.",
    )
    parser.add_argument("--sam2-points-per-side", type=int, default=24)
    parser.add_argument("--sam2-crop-n-layers", type=int, default=0)
    parser.add_argument("--sam2-pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--sam2-stability-score-thresh", type=float, default=0.88)
    parser.add_argument("--preserve-unclaimed-sam2", type=int, default=24)
    parser.add_argument(
        "--no-anchor-langsam-refine",
        action="store_true",
        help="Disable guarded LangSAM refinement after SAM2-auto Molmo-anchor merging.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cuda, cuda:0, or cpu.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    payload = build_org_json(
        points_json_path=Path(args.points_json).resolve(),
        depth_path=Path(args.depth_map).resolve(),
        output_json_path=Path(args.output_json).resolve(),
        output_mask_dir=Path(args.mask_dir).resolve(),
        segmentation_backend=args.segmentation_backend,
        sam_model_id=args.sam_model_id,
        molmo_model_id=args.molmo_model_id,
        review_model_id=args.review_model_id,
        review_api_key_env=args.review_api_key_env,
        review_base_url=args.review_base_url,
        review_timeout=args.review_timeout,
        epsilon=args.epsilon,
        kernel_size=args.kernel_size,
        min_contact_pixels=args.min_contact_pixels,
        min_contact_ratio=args.min_contact_ratio,
        sam_point_grid_radius=args.sam_point_grid_radius,
        sam_prompt_mode=args.sam_prompt_mode,
        sam_negative_points=args.sam_negative_points,
        mask_clean_kernel=args.mask_clean_kernel,
        proposal_backend=args.proposal_backend,
        proposal_min_area_ratio=args.proposal_min_area_ratio,
        proposal_max_area_ratio=args.proposal_max_area_ratio,
        proposal_iou_threshold=args.proposal_iou_threshold,
        proposal_containment_threshold=args.proposal_containment_threshold,
        proposal_border_fraction_threshold=args.proposal_border_fraction_threshold,
        max_proposal_masks=args.max_proposal_masks,
        save_candidates=args.save_candidates,
        device=args.device,
        use_text_box_prompt=not args.no_text_box_prompt,
        depth_merge_threshold=args.depth_merge_threshold,
        depth_merge_boundary_kernel=args.depth_merge_boundary_kernel,
        depth_merge_min_contact_pixels=args.depth_merge_min_contact_pixels,
        anchor_merge_depth_threshold=args.anchor_merge_depth_threshold,
        anchor_refine_with_langsam=not args.no_anchor_langsam_refine,
        sam2_points_per_side=args.sam2_points_per_side,
        sam2_crop_n_layers=args.sam2_crop_n_layers,
        sam2_pred_iou_thresh=args.sam2_pred_iou_thresh,
        sam2_stability_score_thresh=args.sam2_stability_score_thresh,
        preserve_unclaimed_sam2=args.preserve_unclaimed_sam2,
    )
    print(f"Saved occlusion graph JSON to: {args.output_json}")
    print(f"Saved {len(payload['graph']['nodes'])} masks to: {args.mask_dir}")


if __name__ == "__main__":
    main()
