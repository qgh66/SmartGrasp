"""End-to-end pipeline: Molmo points -> SAM masks -> occlusion graph JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
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

from SmartGrasp.perception.occul_map.org import build_occlusion_graph, graph_to_jsonable

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only in cv2-less environments
    cv2 = None

_SAM_CACHE: dict[tuple[str, str], tuple[SamProcessor, SamModel]] = {}
_LANGSAM_CACHE: dict[tuple[str, str], Any] = {}


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


def _semantic_prompt(label: str) -> str:
    prompt = " ".join(part for part in _sanitize_label(label).split("_") if part)
    return f"{prompt}." if prompt else "object."


BACKGROUND_LABEL_TERMS = {
    "background",
    "table",
    "tray",
    "bin",
    "box",
    "container",
    "surface",
    "holder",
    "support",
}
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


def _is_background_or_vague_label(label: str) -> tuple[bool, str | None]:
    normalized = _sanitize_label(label)
    compact = normalized.replace("_", "")
    tokens = set(normalized.split("_"))
    if compact in VAGUE_LABEL_TERMS or tokens & VAGUE_LABEL_TERMS:
        return True, "vague_label"
    if tokens & BACKGROUND_LABEL_TERMS:
        return True, "background_or_container_label"
    return False, None


def _sanitize_label(label: str) -> str:
    tokens = [token for token in _safe_label(label).split("_") if token]
    filtered = [token for token in tokens if token not in VAGUE_LABEL_TERMS]
    if filtered:
        return "_".join(filtered)
    return "_".join(tokens)


def _filter_points(points: list[MolmoPoint]) -> tuple[list[MolmoPoint], list[dict[str, Any]]]:
    kept: list[MolmoPoint] = []
    removed: list[dict[str, Any]] = []
    for point in points:
        should_remove, reason = _is_background_or_vague_label(point.label)
        if should_remove:
            removed.append(
                {
                    "molmo_id": point.molmo_id,
                    "x": point.x,
                    "y": point.y,
                    "label": point.label,
                    "reason": reason,
                }
            )
        else:
            sanitized = _sanitize_label(point.label)
            kept.append(MolmoPoint(point.molmo_id, point.x, point.y, sanitized or point.label))
    return kept, removed


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
        too_small_penalty = max(0.0, 0.001 - area_ratio) * 200.0
        large_penalty = max(0.0, area_ratio - 0.08) * 4.0
        giant_penalty = max(0.0, area_ratio - 0.18) * 10.0
        empty_penalty = 1.0 if area == 0 else 0.0
        prompt_penalty = 0.6 if prompt_hits == 0 else 0.0
        selection_score = (
            float(scores_np[idx])
            + 0.35 * prompt_coverage
            - too_small_penalty
            - large_penalty
            - giant_penalty
            - empty_penalty
            - prompt_penalty
        )
        candidate = {
            "candidate_index": idx,
            "mask": cleaned_mask,
            "area": area,
            "area_ratio": area_ratio,
            "predicted_iou": float(scores_np[idx]),
            "prompt_hits": int(prompt_hits),
            "prompt_coverage": prompt_coverage,
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


def _select_langsam_mask(
    masks: list[np.ndarray],
    scores: list[float],
    point: MolmoPoint,
    mask_clean_kernel: int,
) -> tuple[np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for idx, raw_mask in enumerate(masks):
        mask = _clean_mask(raw_mask, mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        contains_point = _point_inside_mask(mask, point)
        centroid_distance = _mask_centroid_distance(mask, point)
        score = float(scores[idx]) if idx < len(scores) else 0.0
        selection_score = score + (2.0 if contains_point else 0.0) - min(centroid_distance / 1000.0, 1.0)
        candidates.append(
            {
                "candidate_index": idx,
                "mask": mask,
                "area": area,
                "contains_point": contains_point,
                "centroid_distance": centroid_distance,
                "semantic_score": score,
                "selection_score": selection_score,
            }
        )

    if not candidates:
        return None, {}, []

    best = max(candidates, key=lambda item: item["selection_score"])
    metadata = {key: value for key, value in best.items() if key != "mask"}
    candidate_metadata = [{key: value for key, value in item.items() if key != "mask"} for item in candidates]
    return best["mask"], metadata, candidate_metadata


def _langsam_predict(model: Any, image: Image.Image, prompt: str) -> tuple[list[np.ndarray], list[float]]:
    result = model.predict([image], [prompt])
    if isinstance(result, list):
        if not result:
            return [], []
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
        else:
            raw_masks = item
            raw_scores = []
    elif isinstance(result, dict):
        raw_masks = result.get("masks")
        if raw_masks is None:
            raw_masks = []
        raw_scores = result.get("mask_scores")
        if raw_scores is None:
            raw_scores = result.get("scores")
        if raw_scores is None:
            raw_scores = []
    else:
        raw_masks = getattr(result, "masks", [])
        raw_scores = getattr(result, "mask_scores", getattr(result, "scores", []))

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
    return masks, scores


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in mask_records:
        mask = np.asarray(record["mask_array"], dtype=bool)
        duplicate_of: dict[str, Any] | None = None
        for kept_record in kept:
            kept_mask = np.asarray(kept_record["mask_array"], dtype=bool)
            containment = _mask_overlap_fraction(mask, kept_mask)
            if containment < containment_threshold:
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


def _is_tray_or_background_like_proposal(image_np: np.ndarray, mask: np.ndarray) -> bool:
    x, y, bbox_width, bbox_height = _mask_bbox(mask)
    if bbox_width <= 0 or bbox_height <= 0:
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


def _generate_background_exclusion_mask(
    model: Any,
    image: Image.Image,
    existing_foreground_union: np.ndarray,
    mask_clean_kernel: int = 3,
) -> np.ndarray:
    """Paint foreground black, ask LangSAM for \"background.\", exclude foreground.

    If LangSAM fails, returns an empty mask (no exclusion).
    """
    import numpy as np

    # Paint foreground areas black so LangSAM focuses on remaining regions
    image_np = np.asarray(image).copy()
    image_np[existing_foreground_union] = 0
    masked_image = Image.fromarray(image_np)

    image_area = float(masked_image.size[0] * masked_image.size[1])
    background_union = np.zeros((masked_image.size[1], masked_image.size[0]), dtype=bool)

    try:
        masks, _scores = _langsam_predict(model, masked_image, "background area around the objects.")
    except Exception:
        return background_union

    for raw_mask in masks:
        mask = _clean_mask(raw_mask, mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        area_ratio = float(area / image_area)
        if area_ratio < 0.01 or area_ratio > 0.7:
            continue
        background_union |= mask

    if int(np.count_nonzero(background_union)) == 0:
        return background_union

    # Safety: exclude any regions inside foreground masks
    background_union &= ~existing_foreground_union
    return background_union


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
        elif _is_support_like_horizontal_strip(mask) or _is_tray_or_background_like_proposal(image_np, mask):
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
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_langsam(device)

    mask_records: list[dict[str, Any]] = []
    for point in points:
        prompt = _semantic_prompt(point.label)
        masks, scores = _langsam_predict(model, image, prompt)
        best_mask, selected_candidate, candidate_metadata = _select_langsam_mask(
            masks,
            scores,
            point,
            mask_clean_kernel,
        )
        if best_mask is None:
            raise ValueError(f"LangSAM returned no masks for point {point.molmo_id} prompt={prompt!r}.")

        previous_masks = [np.asarray(record["mask_array"], dtype=bool) for record in mask_records]
        max_previous_iou = max((_mask_iou(best_mask, previous) for previous in previous_masks), default=0.0)
        point_hit = _point_inside_mask(best_mask, point)
        fallback_reason: str | None = None
        if not point_hit:
            fallback_reason = "semantic_mask_misses_point"
        elif max_previous_iou > 0.3:
            fallback_reason = "semantic_mask_duplicates_previous_instance"

        fallback_record: dict[str, Any] | None = None
        if fallback_reason is not None:
            fallback_records = generate_masks_with_sam(
                image_path=image_path,
                points=[point],
                output_mask_dir=output_mask_dir,
                sam_model_id=sam_model_id,
                point_grid_radius=sam_point_grid_radius,
                prompt_mode=sam_prompt_mode,
                negative_points=0,
                mask_clean_kernel=mask_clean_kernel,
                save_candidates=save_candidates,
                device=device,
            )
            fallback_record = fallback_records[0]
            fallback_mask = np.asarray(fallback_record["mask_array"], dtype=bool)
            fallback_hit = _point_inside_mask(fallback_mask, point)
            fallback_max_previous_iou = max((_mask_iou(fallback_mask, previous) for previous in previous_masks), default=0.0)
            if fallback_hit and fallback_max_previous_iou <= 0.3:
                best_mask = fallback_mask
            else:
                fallback_record = None

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
    point_grid_radius: int = 0,
    prompt_mode: str = "cross",
    negative_points: int = 0,
    mask_clean_kernel: int = 3,
    save_candidates: bool = False,
    device: str | None = None,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = _load_sam(sam_model_id, device)

    mask_records: list[dict[str, Any]] = []
    for point in points:
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
            inputs = processor(
                image,
                input_points=[[input_points]],
                input_labels=[[input_labels]],
                return_tensors="pt",
            )
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
                prompt_metadata={
                    "variant_index": variant_idx,
                    "prompt_mode": variant["mode"],
                    "prompt_radius": int(variant["radius"]),
                    "prompt_negative_points": int(variant["negative_points"]),
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
    segmentation_backend: str = "sam",
    sam_model_id: str = "facebook/sam-vit-base",
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
) -> dict[str, Any]:
    _prepare_mask_output_dir(output_mask_dir, save_candidates)
    points_payload, points, image_path = _load_points(points_json_path)
    raw_molmo_ids = {int(point.molmo_id) for point in points}
    points, filtered_points = _filter_points(points)
    if not points:
        raise ValueError("All Molmo points were filtered as background or vague labels.")
    depth_map = _load_depth_map(depth_path)

    effective_backend = segmentation_backend
    if segmentation_backend in {"langsam", "auto"}:
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
        )

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

    # --- background exclusion mask (paint foreground black, ask LangSAM "background.") ---
    background_exclusion_mask: np.ndarray | None = None
    if proposal_backend == "sam2-auto":
        foreground_masks = [np.asarray(record["mask_array"], dtype=bool) for record in mask_records]
        foreground_union = np.any(np.stack(foreground_masks, axis=0), axis=0) if foreground_masks else np.zeros(depth_map.shape, dtype=bool)
        try:
            langsam_model = _load_langsam(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            background_exclusion_mask = _generate_background_exclusion_mask(
                model=langsam_model,
                image=Image.open(image_path).convert("RGB"),
                existing_foreground_union=foreground_union,
                mask_clean_kernel=mask_clean_kernel,
            )
        except Exception as exc:
            print(f"Background exclusion mask generation failed: {exc}", file=sys.stderr, flush=True)

        if background_exclusion_mask is not None and np.count_nonzero(background_exclusion_mask) > 0:
            _save_mask_png(background_exclusion_mask, output_mask_dir / "000_background.png")

    proposal_report: list[dict[str, Any]] = []
    effective_proposal_backend = "none"
    if proposal_backend == "sam2-auto":
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

    mask_records, proposal_duplicate_report = _drop_contained_duplicate_masks(
        mask_records,
    )
    mask_records = _renumber_masks(mask_records, output_mask_dir)
    duplicate_mask_report.extend(proposal_duplicate_report)

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
        "save_candidates": bool(save_candidates),
        "sam_model_id": sam_model_id,
        "sam_prompt_mode": sam_prompt_mode,
        "sam_negative_points": int(sam_negative_points),
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
        choices=["sam", "langsam", "auto"],
        default="sam",
        help="Mask generator: SAM point prompts, LangSAM semantic prompts, or LangSAM with SAM fallback.",
    )
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
        "--save-candidates",
        action="store_true",
        help="Save intermediate candidate masks for debugging.",
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
    )
    print(f"Saved occlusion graph JSON to: {args.output_json}")
    print(f"Saved {len(payload['graph']['nodes'])} masks to: {args.mask_dir}")


if __name__ == "__main__":
    main()
