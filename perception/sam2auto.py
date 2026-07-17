"""SAM2 automatic mask generation: model loading, candidate pool, scoring, visualization, pipeline orchestrator."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from SmartGrasp.perception._shared import (
    _as_numpy_mask, _box_xywh_to_xyxy,
    _clean_mask, _draw_labeled_image_matplotlib, _log_step,
    _mask_bbox, _mask_centroid_xy, _nearest_mask_point_xy,
    _safe_label, _save_mask_png,
)
from SmartGrasp.perception.background import (
    background_overlap_fraction,
    BACKGROUND_OVERLAP_REJECTION_THRESHOLD,
    exclude_background_pixels,
)
from SmartGrasp.perception.vlm import review_and_assign_sam2

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

import matplotlib
matplotlib.use("Agg")

_SAM2_WRAPPER_CACHE: dict[tuple[str, str], Any] = {}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SAM2_ROOT = Path(os.environ.get("SAM2_ROOT", _PROJECT_ROOT / "sam2_repo"))
if not _SAM2_ROOT.exists():
    for candidate in (
        _PROJECT_ROOT / "sam2",
        Path.home() / "sam2",
        Path.home() / "Gsam2" / "Grounded-SAM-2",
        Path.home() / "Grounded-SAM-2",
    ):
        if candidate.exists():
            _SAM2_ROOT = candidate
            break
if not _SAM2_ROOT.exists():
    raise FileNotFoundError("SAM2 root not found. Set env SAM2_ROOT or place sam2 in project root.")
if str(_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM2_ROOT))
DEFAULT_SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
DEFAULT_SAM2_CHECKPOINT = Path(
    os.environ.get(
        "SAM2_CHECKPOINT",
        str(_SAM2_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"),
    )
)


@dataclass
class Sam2AutoWrapper:
    model: Any
    mask_generator: Any

    def generate(self, image_np: np.ndarray) -> list[dict[str, Any]]:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            return list(self.mask_generator.generate(image_np))


def _load_sam2_wrapper(device: str) -> Any:
    cache_key = ("default", device)
    if cache_key not in _SAM2_WRAPPER_CACHE:
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 is not installed in the active environment. "
                "Install facebookresearch/sam2 before running perception."
            ) from exc

        if not DEFAULT_SAM2_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {DEFAULT_SAM2_CHECKPOINT}. "
                "Set SAM2_CHECKPOINT to a valid checkpoint path."
            )

        model = build_sam2(DEFAULT_SAM2_CONFIG, str(DEFAULT_SAM2_CHECKPOINT), device=device)
        mask_generator = SAM2AutomaticMaskGenerator(model)
        _SAM2_WRAPPER_CACHE[cache_key] = Sam2AutoWrapper(model=model, mask_generator=mask_generator)
    return _SAM2_WRAPPER_CACHE[cache_key]


def clear_sam2_image_state() -> None:
    """Release cached accelerator memory between consecutive scenes."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sam2_auto_generate(model: Any, image: Image.Image) -> list[dict[str, Any]]:
    # Fixed seeds for reproducible SAM2 output across Linux/CUDA
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    image_np = np.array(image.convert("RGB"))
    if not hasattr(model, "generate"):
        raise RuntimeError("Loaded SAM2 wrapper does not expose automatic mask generation.")
    return list(model.generate(image_np))


def _configure_sam2_auto_generator(
    model: Any,
    points_per_side: int | None = None,
    crop_n_layers: int | None = None,
    pred_iou_thresh: float | None = None,
    stability_score_thresh: float | None = None,
) -> dict[str, Any]:
    generator = getattr(model, "mask_generator", None)
    sam_model = getattr(model, "model", None)
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

        model.mask_generator = SAM2AutomaticMaskGenerator(sam_model, **kwargs)
    except Exception as exc:
        return {"configured": False, "reason": str(exc), "requested": kwargs}
    return {"configured": True, "settings": kwargs}


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





def _proposal_score(proposal: dict[str, Any]) -> float:
    predicted_iou = float(proposal.get("predicted_iou", proposal.get("iou", 0.0)) or 0.0)
    stability = float(proposal.get("stability_score", 0.0) or 0.0)
    return predicted_iou + 0.25 * stability


def _depth_map_to_near_white_image(depth_map: np.ndarray | None, image_shape: tuple[int, int]) -> Image.Image | None:
    if depth_map is None or depth_map.size == 0:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.shape[:2] != image_shape:
        return None
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return None
    near, far = np.percentile(depth[valid], [2, 98])
    if far <= near:
        return None
    normalized = (far - np.clip(depth, near, far)) / max(far - near, 1e-6)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = np.clip(normalized[valid] * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def _normalized_depth_gradient(
    depth_map: np.ndarray | None,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if depth_map is None or depth_map.size == 0 or cv2 is None:
        return None, None
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.shape[:2] != image_shape:
        return None, None
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return None, None
    fill_value = float(np.median(depth[valid]))
    depth_filled = np.where(valid, depth, fill_value).astype(np.float32)
    depth_smooth = cv2.GaussianBlur(depth_filled, (5, 5), 0)
    grad_x = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    scale = float(np.percentile(gradient[valid], 95))
    if scale <= 1e-6:
        return np.zeros_like(gradient, dtype=np.float32), valid
    return np.clip(gradient / scale, 0.0, 1.0).astype(np.float32), valid


def _internal_depth_edge_report(
    mask: np.ndarray,
    depth_gradient: np.ndarray | None,
    valid_depth_mask: np.ndarray | None,
) -> dict[str, Any]:
    if depth_gradient is None or valid_depth_mask is None or cv2 is None:
        return {"has_internal_depth_edge": False, "reason": "depth_gradient_unavailable"}

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    kernel = np.ones((3, 3), np.uint8)
    checked_components = 0
    strongest_edge_fraction = 0.0
    strongest_edge_p90 = 0.0

    for label in range(1, num_labels):
        component = labels == label
        if int(np.count_nonzero(component)) < 64:
            continue
        interior = cv2.erode(component.astype(np.uint8), kernel, iterations=2).astype(bool)
        interior &= valid_depth_mask
        if int(np.count_nonzero(interior)) < 32:
            continue
        values = depth_gradient[interior]
        edge_fraction = float(np.mean(values >= 0.45))
        edge_p90 = float(np.percentile(values, 90))
        checked_components += 1
        strongest_edge_fraction = max(strongest_edge_fraction, edge_fraction)
        strongest_edge_p90 = max(strongest_edge_p90, edge_p90)
        if edge_fraction >= 0.08 and edge_p90 >= 0.70:
            return {
                "has_internal_depth_edge": True,
                "checked_components": checked_components,
                "internal_edge_fraction": edge_fraction,
                "internal_edge_p90": edge_p90,
            }

    return {
        "has_internal_depth_edge": False,
        "checked_components": checked_components,
        "internal_edge_fraction": strongest_edge_fraction,
        "internal_edge_p90": strongest_edge_p90,
    }



def _resolve_overlaps_by_depth(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve overlapping region between mask pairs using spatial k-NN voting.

    The overlap region is sampled. Each sample finds its k nearest neighbors
    from A's exclusive region and k nearest from B's exclusive region, then
    votes for whichever side is closer. The entire overlap region is assigned
    to the winner of the majority vote.
    """
    if len(candidates) < 2:
        return candidates

    n = len(candidates)
    masks = [np.asarray(c["mask"], dtype=bool).copy() for c in candidates]
    modified = False
    n_sample = 500
    k = 7

    for i in range(n):
        for j in range(i + 1, n):
            overlap = masks[i] & masks[j]
            n_overlap = int(np.count_nonzero(overlap))
            if n_overlap == 0:
                continue

            excl_i = masks[i] & ~masks[j]
            excl_j = masks[j] & ~masks[i]
            n_i = int(np.count_nonzero(excl_i))
            n_j = int(np.count_nonzero(excl_j))

            if n_i == 0 and n_j == 0:
                continue
            if n_i == 0:
                masks[i][overlap] = False
                modified = True
                continue
            if n_j == 0:
                masks[j][overlap] = False
                modified = True
                continue

            # Sample exclusive region coords
            coords_i = np.argwhere(excl_i).astype(np.float32)
            coords_j = np.argwhere(excl_j).astype(np.float32)

            # Sample overlap coords for voting
            coords_o = np.argwhere(overlap).astype(np.float32)
            n_o_sample = min(n_sample, n_overlap)
            sample_o = coords_o[np.random.choice(n_overlap, n_o_sample, replace=False)]

            votes_i = 0
            for p in sample_o:
                # Find k nearest from A exclusive, mean distance
                dist_i = np.sum((coords_i - p) ** 2, axis=1)
                k_i = min(k, n_i) - 1
                top_i = np.partition(dist_i, k_i)[:k_i + 1] if k_i >= 0 else np.array([float('inf')])
                mean_i = np.mean(top_i) if len(top_i) > 0 else float('inf')

                # Find k nearest from B exclusive, mean distance
                dist_j = np.sum((coords_j - p) ** 2, axis=1)
                k_j = min(k, n_j) - 1
                top_j = np.partition(dist_j, k_j)[:k_j + 1] if k_j >= 0 else np.array([float('inf')])
                mean_j = np.mean(top_j) if len(top_j) > 0 else float('inf')

                if mean_i <= mean_j:
                    votes_i += 1

            if votes_i > n_o_sample // 2:
                masks[j][overlap] = False
            else:
                masks[i][overlap] = False
            modified = True

    if not modified:
        return candidates

    for idx, c in enumerate(candidates):
        c["mask"] = masks[idx]
        c["mask_area"] = int(np.count_nonzero(masks[idx]))

    return candidates


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
        background_overlap = background_overlap_fraction(mask, background_exclusion_mask)
        if background_overlap >= BACKGROUND_OVERLAP_REJECTION_THRESHOLD:
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


def _hard_filter_sam2_proposals(
    raw_proposals: list[dict[str, Any]],
    source: str,
    image_np: np.ndarray,
    image_area: float,
    min_area_ratio: float,
    max_area_ratio: float,
    border_fraction_threshold: float,
    mask_clean_kernel: int,
    background_exclusion_mask: np.ndarray | None,
    max_candidates: int | None,
    report: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    effective_min_area_ratio = max(0.0002, min_area_ratio * 0.15)
    effective_max_area_ratio = max(max_area_ratio, 0.45)
    candidates: list[dict[str, Any]] = []

    for index, proposal in enumerate(raw_proposals):
        raw_mask = proposal.get("segmentation")
        if raw_mask is None:
            continue
        mask = _clean_mask(_as_numpy_mask(raw_mask), mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        area_ratio = float(area / image_area)
        border_fraction = _border_touch_fraction(mask)
        background_overlap = background_overlap_fraction(mask, background_exclusion_mask)
        bbox_xywh = [int(value) for value in proposal.get("bbox", _mask_bbox(mask))]
        bbox_xyxy = _box_xywh_to_xyxy(bbox_xywh)
        cx, cy = _mask_centroid_xy(mask)

        reason: str | None = None
        if area_ratio < effective_min_area_ratio:
            reason = "too_small"
        elif area_ratio > effective_max_area_ratio:
            reason = "too_large"
        elif _is_support_like_horizontal_strip(mask) or _is_tray_or_background_like_proposal(image_np, mask, background_exclusion_mask):
            reason = "support_or_tray_like"
        elif background_overlap > 0.5:
            reason = "overlaps_background_exclusion"

        metadata = {
            "source": source,
            "proposal_index": int(index),
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

        # ── Background-pixel exclusion ───────────────────────────────────
        # Strip background pixels from every candidate that survived hard
        # filtering, so downstream VLM review / merging / morphology never
        # see background material.
        # (empty-after-exclusion is impossible: background_overlap ≤ 0.5
        #  is already enforced by the hard-filter checks above.)
        area_before_exclusion = area
        if background_exclusion_mask is not None and int(np.count_nonzero(background_exclusion_mask)) > 0:
            cleaned = exclude_background_pixels(mask, background_exclusion_mask)
            removed = area_before_exclusion - int(np.count_nonzero(cleaned))
            if removed > 0:
                metadata["area_before_background_exclusion"] = area_before_exclusion
                metadata["background_exclusion_pixels_removed"] = removed
                mask = cleaned
                # Recompute every geometry-dependent field from the cleaned mask
                area = int(np.count_nonzero(mask))
                area_ratio = float(area / image_area)
                border_fraction = _border_touch_fraction(mask)
                background_overlap = background_overlap_fraction(mask, background_exclusion_mask)
                bbox_xywh = _mask_bbox(mask)
                bbox_xyxy = _box_xywh_to_xyxy(bbox_xywh)
                cx, cy = _mask_centroid_xy(mask)
                metadata["area"] = area
                metadata["area_ratio"] = area_ratio
                metadata["bbox"] = bbox_xywh
                metadata["bbox_xyxy"] = bbox_xyxy
                metadata["border_fraction"] = border_fraction
                metadata["background_exclusion_overlap"] = background_overlap
                metadata["centroid"] = {"x": int(cx), "y": int(cy)}

        metadata["selection_score"] = _proposal_score(proposal)
        metadata["mask"] = mask
        candidates.append(metadata)
        report.append({key: value for key, value in metadata.items() if key != "mask"})

    candidates.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
    if max_candidates is None:
        return candidates
    return candidates[:max_candidates]


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": candidate.get("source"),
        "proposal_index": int(candidate.get("proposal_index", -1)),
        "area": int(candidate.get("area", 0)),
        "selection_score": float(candidate.get("selection_score", 0.0)),
    }


def _refresh_candidate_geometry(candidate: dict[str, Any], mask: np.ndarray, image_area: float) -> None:
    area = int(np.count_nonzero(mask))
    bbox_xywh = _mask_bbox(mask)
    bbox_xyxy = _box_xywh_to_xyxy(bbox_xywh)
    center_x, center_y = _mask_centroid_xy(mask)
    candidate["mask"] = mask
    candidate["area"] = area
    candidate["area_ratio"] = float(area / image_area)
    candidate["bbox"] = bbox_xywh
    candidate["bbox_xyxy"] = bbox_xyxy
    candidate["centroid"] = {"x": int(center_x), "y": int(center_y)}


def _merge_candidate_union(
    kept: list[dict[str, Any]],
    matched_indices: list[int],
    candidate: dict[str, Any],
    image_area: float,
    report: list[dict[str, Any]],
) -> None:
    primary = kept[matched_indices[0]]
    merged_mask = np.asarray(candidate["mask"], dtype=bool).copy()
    merged_sources = set(candidate.get("merged_sources", [candidate.get("source")]))
    merged_records = list(candidate.get("merged_candidates", [_candidate_summary(candidate)]))

    for kept_index in matched_indices:
        kept_candidate = kept[kept_index]
        merged_mask |= np.asarray(kept_candidate["mask"], dtype=bool)
        merged_sources.update(kept_candidate.get("merged_sources", [kept_candidate.get("source")]))
        merged_records.extend(kept_candidate.get("merged_candidates", [_candidate_summary(kept_candidate)]))

    for kept_index in reversed(matched_indices[1:]):
        kept.pop(kept_index)

    _refresh_candidate_geometry(primary, merged_mask, image_area)
    if len(merged_sources) > 1:
        merged_records.sort(key=lambda r: int(r.get("area", 0)), reverse=True)
        primary["source"] = "-".join(str(r.get("source", "?")) for r in merged_records) + "-merged"
    else:
        primary["source"] = next(iter(merged_sources))
    primary["merged_sources"] = sorted(source for source in merged_sources if source is not None)
    primary["merged_candidates"] = merged_records
    primary["selection_score"] = max(
        float(primary.get("selection_score", 0.0)),
        float(candidate.get("selection_score", 0.0)),
    )
    primary["predicted_iou"] = max(
        float(primary.get("predicted_iou", 0.0)),
        float(candidate.get("predicted_iou", 0.0)),
    )
    primary["stability_score"] = max(
        float(primary.get("stability_score", 0.0)),
        float(candidate.get("stability_score", 0.0)),
    )
    report.append({
        "stage": "union_merge",
        "merged_into": _candidate_summary(primary),
        "merged_from": _candidate_summary(candidate),
        "num_merged_candidates": len(merged_records),
    })


def _merge_candidates_with_depth_edges(
    candidates: list[dict[str, Any]],
    image_area: float,
    depth_gradient: np.ndarray | None,
    valid_depth_mask: np.ndarray | None,
    report: list[dict[str, Any]],
    initial_kept: list[dict[str, Any]] | None = None,
    reject_internal_depth_edges: bool = True,
    containment_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = list(initial_kept or [])
    cumulative_mask: np.ndarray | None = None
    for kept_candidate in kept:
        kept_mask = np.asarray(kept_candidate["mask"], dtype=bool)
        cumulative_mask = kept_mask.copy() if cumulative_mask is None else (cumulative_mask | kept_mask)

    for candidate in candidates:
        new_mask = np.asarray(candidate["mask"], dtype=bool)
        new_area = int(np.count_nonzero(new_mask))
        if new_area == 0:
            continue

        if reject_internal_depth_edges:
            depth_edge_report = _internal_depth_edge_report(new_mask, depth_gradient, valid_depth_mask)
            candidate["depth_edge_report"] = depth_edge_report
            if depth_edge_report.get("has_internal_depth_edge"):
                candidate["rejection_reason"] = "internal_depth_edge"
                report.append({key: value for key, value in candidate.items() if key != "mask"})
                continue

        num_components = 1
        if cv2 is not None:
            num_labels, _ = cv2.connectedComponents(new_mask.astype(np.uint8), connectivity=8)
            num_components = num_labels - 1
        if num_components > 1 and cumulative_mask is not None:
            coverage = float(np.count_nonzero(new_mask & cumulative_mask) / max(1, new_area))
            if coverage >= containment_threshold:
                candidate["rejection_reason"] = "discontiguous_covered_by_cumulative"
                candidate["cumulative_coverage"] = coverage
                report.append({key: value for key, value in candidate.items() if key != "mask"})
                continue

        matched_indices: list[int] = []
        for kept_index, kept_candidate in enumerate(kept):
            kept_mask = np.asarray(kept_candidate["mask"], dtype=bool)
            kept_area = int(np.count_nonzero(kept_mask))
            intersection = int(np.count_nonzero(new_mask & kept_mask))
            new_coverage = intersection / max(1, new_area)
            kept_coverage = intersection / max(1, kept_area)
            if new_coverage >= containment_threshold or kept_coverage >= containment_threshold:
                matched_indices.append(kept_index)

        if matched_indices:
            _merge_candidate_union(kept, matched_indices, candidate, image_area, report)
        else:
            kept.append(candidate)

        cumulative_mask = None
        for kept_candidate in kept:
            kept_mask = np.asarray(kept_candidate["mask"], dtype=bool)
            cumulative_mask = kept_mask.copy() if cumulative_mask is None else (cumulative_mask | kept_mask)

    kept.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
    return kept


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
    depth_map: np.ndarray | None = None,
    depth_points_per_side: int | None = None,
    depth_crop_n_layers: int | None = None,
    depth_pred_iou_thresh: float | None = None,
    depth_stability_score_thresh: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, Image.Image]:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    image_area = float(image_np.shape[0] * image_np.shape[1])
    model = _load_sam2_wrapper(device)
    generator_settings = _configure_sam2_auto_generator(
        model,
        points_per_side=points_per_side,
        crop_n_layers=crop_n_layers,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    report: list[dict[str, Any]] = [{"stage": "sam2_auto_generator", **generator_settings}]

    rgb_proposals = _sam2_auto_generate(model, image)
    rgb_candidates = _hard_filter_sam2_proposals(
        raw_proposals=rgb_proposals,
        source="rgb",
        image_np=image_np,
        image_area=image_area,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        border_fraction_threshold=border_fraction_threshold,
        mask_clean_kernel=mask_clean_kernel,
        background_exclusion_mask=background_exclusion_mask,
        max_candidates=None,
        report=report,
    )

    depth_candidates: list[dict[str, Any]] = []
    depth_image = _depth_map_to_near_white_image(depth_map, image_np.shape[:2])
    if depth_image is not None:
        depth_generator_settings = _configure_sam2_auto_generator(
            model,
            points_per_side=depth_points_per_side if depth_points_per_side is not None else points_per_side,
            crop_n_layers=depth_crop_n_layers if depth_crop_n_layers is not None else crop_n_layers,
            pred_iou_thresh=depth_pred_iou_thresh if depth_pred_iou_thresh is not None else pred_iou_thresh,
            stability_score_thresh=depth_stability_score_thresh if depth_stability_score_thresh is not None else stability_score_thresh,
        )
        report.append({"stage": "depth_sam2_auto_generator", **depth_generator_settings})
        depth_proposals = _sam2_auto_generate(model, depth_image)
        depth_candidates = _hard_filter_sam2_proposals(
            raw_proposals=depth_proposals,
            source="depth",
            image_np=image_np,
            image_area=image_area,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            border_fraction_threshold=border_fraction_threshold,
            mask_clean_kernel=mask_clean_kernel,
            background_exclusion_mask=background_exclusion_mask,
            max_candidates=None,
            report=report,
        )

    depth_gradient, valid_depth_mask = _normalized_depth_gradient(depth_map, image_np.shape[:2])
    rgb_candidates = _merge_candidates_with_depth_edges(
        candidates=rgb_candidates,
        image_area=image_area,
        depth_gradient=depth_gradient,
        valid_depth_mask=valid_depth_mask,
        report=report,
        reject_internal_depth_edges=False,
    )
    candidates = _merge_candidates_with_depth_edges(
        candidates=depth_candidates,
        image_area=image_area,
        depth_gradient=depth_gradient,
        valid_depth_mask=valid_depth_mask,
        report=report,
        initial_kept=rgb_candidates,
        reject_internal_depth_edges=True,
    )
    candidates = _resolve_overlaps_by_depth(candidates)
    if save_candidates:
        candidate_dir = output_mask_dir.parent / "sam2_auto_candidates"
        for candidate in candidates:
            source = str(candidate.get("source", "rgb"))
            candidate_path = candidate_dir / f"{source}_proposal_{int(candidate['proposal_index']):03d}.png"
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
        mask = np.asarray(candidate["mask"], dtype=bool)
        centroid_xy = _mask_centroid_xy(mask)
        x, y = _nearest_mask_point_xy(mask, centroid_xy)
        points_with_ids.append((index, x, y))
        masks.append(mask)

    with Image.open(image_path).convert("RGB") as image:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            for mask_index, mask in enumerate(masks):
                color = plt.cm.tab20(mask_index % 20)
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[mask] = [color[0], color[1], color[2], 0.28]
                ax.imshow(overlay)
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
            _draw_labeled_image_matplotlib(image=image, points_with_ids=points_with_ids, out_png_path=out_path)
            return


def _save_sam2_rgb_parts_sheet(
    image_path: Path,
    candidates: list[dict[str, Any]],
    out_dir: Path,
    max_labels: int = 35,
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

    try:
        label_font = ImageFont.truetype("Arial.ttf", 28)
    except Exception:
        label_font = ImageFont.load_default()

    label_height = 42
    columns = 5
    rows = int(np.ceil(len(part_images) / columns))
    cell_width = max(part_image.width for _part_id, part_image in part_images)
    cell_height = max(part_image.height for _part_id, part_image in part_images)
    sheet = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for item_index, (part_id, part_image) in enumerate(part_images):
        row = item_index // columns
        col = item_index % columns
        cell_x = col * cell_width
        cell_y = row * (cell_height + label_height)
        x = cell_x + (cell_width - part_image.width) // 2
        y = cell_y + label_height + (cell_height - part_image.height) // 2
        label = str(part_id)
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = x + (part_image.width - text_width) // 2
        text_y = cell_y + (label_height - text_height) // 2
        draw.text((text_x, text_y), label, fill="black", font=label_font)
        sheet.paste(part_image, (x, y))

    sheet_path = out_dir / "sam2_rgb_parts_sheet.png"
    sheet.save(sheet_path)
    return sheet_path


def generate_masks_with_sam2_vlm_pipeline(
    image_path: Path,
    output_mask_dir: Path,
    review_model_id: str,
    review_api_key_env: str,
    review_base_url: str | None,
    review_timeout: float,
    min_area_ratio: float,
    max_area_ratio: float,
    proposal_border_fraction_threshold: float,
    mask_clean_kernel: int,
    save_candidates: bool,
    device: str | None,
    background_exclusion_mask: np.ndarray | None,
    depth_map: np.ndarray | None = None,
    sam2_points_per_side: int | None = None,
    sam2_crop_n_layers: int | None = None,
    sam2_pred_iou_thresh: float | None = None,
    sam2_stability_score_thresh: float | None = None,
    depth_sam2_points_per_side: int | None = None,
    depth_sam2_crop_n_layers: int | None = None,
    depth_sam2_pred_iou_thresh: float | None = None,
    depth_sam2_stability_score_thresh: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = output_mask_dir.parent

    t_sam2 = _log_step("  ②a sam2_auto", None)
    candidates, sam2_report, model, image = _sam2_auto_candidate_pool(
        image_path=image_path,
        output_mask_dir=output_mask_dir,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        mask_clean_kernel=mask_clean_kernel,
        save_candidates=save_candidates,
        device=device,
        background_exclusion_mask=background_exclusion_mask,
        points_per_side=sam2_points_per_side,
        crop_n_layers=sam2_crop_n_layers,
        pred_iou_thresh=sam2_pred_iou_thresh,
        stability_score_thresh=sam2_stability_score_thresh,
        border_fraction_threshold=proposal_border_fraction_threshold,
        depth_map=depth_map,
        depth_points_per_side=depth_sam2_points_per_side,
        depth_crop_n_layers=depth_sam2_crop_n_layers,
        depth_pred_iou_thresh=depth_sam2_pred_iou_thresh,
        depth_stability_score_thresh=depth_sam2_stability_score_thresh,
    )
    _log_step("  ②a sam2_auto", t_sam2)

    label_path = out_dir / "label_1_sam2auto.png"
    _draw_sam2_auto_label_image(image_path, candidates, label_path)
    parts_sheet_path = _save_sam2_rgb_parts_sheet(image_path, candidates, out_dir)
    try:
        t_review = _log_step("  ②b vlm_review", None)
        review_objects = review_and_assign_sam2(
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
        _log_step("  ②b vlm_review", t_review)
    except Exception as exc:
        _log_step("  ②b vlm_review FAILED", t_sam2)
        raise RuntimeError(f"VLM review failed — API error. Aborting. Error: {exc}") from exc

    if not review_objects:
        _log_step("  ②b vlm_review returned empty", t_sam2)
        raise RuntimeError("VLM review returned no objects — empty response. Aborting.")

    t_finalize = _log_step("  ②c anchor_assembly", None)
    mask_records: list[dict[str, Any]] = []
    for item in review_objects:
        sam2_ids = [int(v) for v in item.get("sam2_ids", []) if int(v) > 0]
        object_id = int(item["id"])
        description = str(item["description"])

        # Build SAM2 anchor mask from assigned ids
        anchor_masks = []
        for sam2_id in sam2_ids:
            idx = sam2_id - 1
            if 0 <= idx < len(candidates):
                anchor_masks.append(np.asarray(candidates[idx]["mask"], dtype=bool))
        if not anchor_masks:
            continue
        best_mask = np.any(np.stack(anchor_masks, axis=0), axis=0)
        best_mask = _clean_mask(best_mask, mask_clean_kernel)
        if int(np.count_nonzero(best_mask)) == 0:
            continue
        cx, cy = _mask_centroid_xy(best_mask)
        filename = f"{object_id:03d}_anchor_{_safe_label(description)}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(best_mask, mask_path)
        mask_records.append(
            {
                "node_id": len(mask_records),
                "object_id": object_id,
                "label": description,
                "description": description,
                "point": {"x": int(cx), "y": int(cy)},
                "segmentation_backend": "anchor",
                "sam2_ids": sam2_ids,
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(best_mask)),
                "mask_array": best_mask,
            }
        )

    report = [
        {
            "stage": "sam2_auto_initial",
            "label_png": str(label_path.resolve()),
            "rgb_parts_sheet_png": str(parts_sheet_path.resolve()),
            "candidates": sam2_report[:200],
        },
        {"stage": "vlm_review", "objects": review_objects, "error": None},
    ]
    _log_step("  ②c anchor_assembly", t_finalize)
    return mask_records, report
