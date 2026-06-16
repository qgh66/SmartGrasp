"""LangSAM text-guided segmentation and mask selection for reviewed objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from SmartGrasp.perception._shared import (
    _as_numpy_mask, _clean_mask, _mask_bbox, _mask_centroid_xy,
    _mask_overlap_fraction, _normalize_box, _safe_label,
    _save_mask_png,
)
from SmartGrasp.perception.background import (
    background_overlap_fraction,
    LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD,
)
from SmartGrasp.perception._shared import _proposal_label

LANGSAM_MIN_AREA_RATIO = 0.0007  # Minimum fraction of image area for a valid LangSAM mask (0.07%)

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
        background_overlap = background_overlap_fraction(mask, background_exclusion_mask)
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

