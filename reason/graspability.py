from __future__ import annotations

from typing import Any

import numpy as np

from .vlm import VLMClient, get_default_client


_DEFAULT_CLIENT: VLMClient | None = None


def _client() -> VLMClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = get_default_client()
    return _DEFAULT_CLIENT


def score_current_objects(
    mids: list[int],
    perception,
    client: VLMClient | None = None,
) -> dict[str, Any]:
    """Score object-level and part-level graspability for current objects."""
    if not mids:
        return _fallback([], "no objects to score")

    labeled_rgb = getattr(perception, "labeled_rgb", None)
    if labeled_rgb is None:
        labeled_rgb = getattr(perception, "rgb", None)
    if labeled_rgb is None:
        return _fallback(mids, "no labeled RGB image; used fallback graspability")
    if not isinstance(labeled_rgb, np.ndarray):
        labeled_rgb = np.asarray(labeled_rgb)

    objects = []
    allowed_part_ids: dict[int, set[int]] = {}
    for mid in mids:
        label = f"object_{mid}"
        node = perception.molmo_to_node.get(mid)
        if node is not None:
            label = perception.node_info[node].get("label", label)
        part_ids = _part_ids(perception, mid)
        allowed_part_ids[mid] = set(part_ids)
        objects.append({"mid": mid, "label": label, "part_ids": part_ids})

    try:
        payload = (client or _client()).score_graspability_objects(
            objects=objects,
            labeled_rgb=labeled_rgb,
            parts_sheet_rgb=getattr(perception, "sam2_rgb_parts_sheet", None),
            object_sheet_rgb=getattr(perception, "final_objects_sheet", None),
            occlusion_graph_rgb=getattr(perception, "occlusion_graph_rgb", None),
        )
    except Exception as exc:
        return _fallback(mids, f"graspability VLM failed with {type(exc).__name__}")

    payload.setdefault("graspability", {})
    payload.setdefault("graspability_part_id", {})
    payload.setdefault("graspability_parts", {})
    payload.setdefault("reason", "")
    for mid in mids:
        payload["graspability"].setdefault(mid, 1.0)
        raw_part_scores = payload["graspability_parts"].get(
            mid,
            payload["graspability_parts"].get(str(mid), {}),
        )
        raw_part_scores = raw_part_scores if isinstance(raw_part_scores, dict) else {}
        validated_part_scores: dict[int, float] = {}
        for part_id in sorted(allowed_part_ids[mid]):
            value = raw_part_scores.get(
                part_id,
                raw_part_scores.get(str(part_id), 0.0),
            )
            try:
                validated_part_scores[part_id] = min(
                    1.0,
                    max(0.0, float(value)),
                )
            except (TypeError, ValueError):
                validated_part_scores[part_id] = 0.0
        payload["graspability_parts"][mid] = validated_part_scores
        positive_scores = {
            part_id: score
            for part_id, score in validated_part_scores.items()
            if score > 0.0
        }
        payload["graspability_part_id"][mid] = (
            max(
                positive_scores,
                key=lambda part_id: (
                    positive_scores[part_id],
                    -part_id,
                ),
            )
            if positive_scores
            else None
        )
    return payload


def _part_ids(perception, mid: int) -> list[int]:
    mapping = (
        getattr(perception, "object_id_to_part_ids", None)
        or getattr(perception, "object_id_to_sam2_part_ids", None)
        or {}
    )
    return list(mapping.get(mid, ()))


def _fallback(mids: list[int], reason: str) -> dict[str, Any]:
    return {
        "graspability": {mid: 1.0 for mid in mids},
        "graspability_part_id": {mid: None for mid in mids},
        "graspability_parts": {mid: {} for mid in mids},
        "reason": reason,
    }
