"""P_s for fully-invisible target.

Target is completely hidden — VLM looks at the scene and judges
which visible occluder is most likely hiding it.

Returns mutually-exclusive probabilities (normalized to sum=1).
"""
from __future__ import annotations

import numpy as np
from typing import Any

from ..graspability import score_current_objects
from ..vlm import VLMClient, get_default_client


_DEFAULT_CLIENT: VLMClient | None = None


def _client() -> VLMClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = get_default_client()
    return _DEFAULT_CLIENT


def compute_semantic_prior(
    occluder_mids: list[int],
    target_mid: int,
    perception,
    client: VLMClient | None = None,
) -> dict[int, float]:
    """Return VLM probabilities for which visible object hides the target."""
    payload = compute_semantic_prior_payload(
        occluder_mids, target_mid, perception, client
    )
    return payload["scores"]


def compute_semantic_prior_payload(
    occluder_mids: list[int],
    target_mid: int,
    perception,
    client: VLMClient | None = None,
) -> dict[str, Any]:
    """Return VLM probabilities for which visible object hides the target."""
    print(f"[PRIOR-INV] called: target={target_mid}, occluders={occluder_mids}")

    if not occluder_mids:
        print("[PRIOR-INV] no occluders -> empty")
        return {"scores": {}, "graspability": {}, "reason": "no occluders available"}

    # With exactly one candidate, its semantic probability is deterministic.
    # Avoid asking the VLM to estimate a probability that must be 1.0.  In
    # graspability mode, make one graspability-only request so both the
    # object-level coefficient and every part-level coefficient are retained.
    if len(occluder_mids) == 1:
        mid = occluder_mids[0]
        prompt_mode = getattr(perception, "prior_prompt_mode", "original")
        if prompt_mode == "graspability":
            graspability_payload = score_current_objects(
                [mid], perception, client=client
            )
            return {
                "scores": {mid: 1.0},
                "graspability": graspability_payload.get(
                    "graspability", {mid: 1.0}
                ),
                "graspability_part_id": graspability_payload.get(
                    "graspability_part_id", {mid: None}
                ),
                "graspability_parts": graspability_payload.get(
                    "graspability_parts", {mid: {}}
                ),
                "reason": (
                    "single candidate; assigned deterministic semantic prior. "
                    + str(graspability_payload.get("reason") or "")
                ).strip(),
            }
        return {
            "scores": {mid: 1.0},
            "graspability": {mid: 1.0},
            "graspability_part_id": {mid: None},
            "graspability_parts": {mid: {}},
            "reason": "single candidate; assigned deterministic semantic prior",
        }

    # The hidden target is usually described by the language annotation.
    target_label = "unknown target"
    if target_mid in perception.molmo_to_node:
        target_label = perception.node_info[
            perception.molmo_to_node[target_mid]
        ]["label"]
    elif getattr(perception, "annotation", None):
        target_label = str(perception.annotation)

    # Build candidate descriptors for the VLM prompt.
    occluders = []
    for mid in occluder_mids:
        info = perception.node_info[perception.molmo_to_node[mid]]
        occluders.append(
            {
                "mid": mid,
                "label": info["label"],
                "part_ids": _part_ids(perception, mid),
            }
        )

    # The current VLM prompt requires a labeled RGB image.
    labeled_rgb = getattr(perception, "labeled_rgb", None)
    if labeled_rgb is None:
        labeled_rgb = getattr(perception, "rgb", None)

    if labeled_rgb is None:
        print("[PRIOR-INV] no labeled_rgb -> uniform fallback")
        n = len(occluder_mids)
        scores = {mid: 1.0 / n for mid in occluder_mids}
        return {
            "scores": scores,
            "graspability": {mid: 1.0 for mid in occluder_mids},
            "reason": "no labeled RGB image; used uniform fallback scores",
        }

    if not isinstance(labeled_rgb, np.ndarray):
        labeled_rgb = np.asarray(labeled_rgb)

    print(f"[PRIOR-INV] labeled_rgb loaded, shape={labeled_rgb.shape} -> calling VLM")

    # Query the VLM and keep a normalized probability distribution.
    c = client or _client()
    prompt_mode = getattr(perception, "prior_prompt_mode", "original")
    raw = c.score_occluders_invisible(
        target_label=target_label,
        occluders=occluders,
        labeled_rgb=labeled_rgb,
        parts_sheet_rgb=getattr(perception, "sam2_rgb_parts_sheet", None),
        prompt_mode=prompt_mode,
        object_sheet_rgb=getattr(perception, "final_objects_sheet", None),
        occlusion_graph_rgb=getattr(perception, "occlusion_graph_rgb", None),
    )

    # Fill missing ids with a uniform fallback.
    raw.setdefault("scores", {})
    raw.setdefault("graspability", {})
    raw.setdefault("graspability_part_id", {})
    raw.setdefault("graspability_parts", {})
    raw.setdefault("reason", "")
    for mid in occluder_mids:
        raw["scores"].setdefault(mid, 1.0 / len(occluder_mids))
        raw["graspability"].setdefault(mid, 1.0)
        raw["graspability_part_id"].setdefault(mid, None)
        raw["graspability_parts"].setdefault(mid, {})

    return raw


def _part_ids(perception, mid: int) -> list[int]:
    mapping = (
        getattr(perception, "object_id_to_part_ids", None)
        or getattr(perception, "object_id_to_sam2_part_ids", None)
        or {}
    )
    return list(mapping.get(mid, ()))
