"""P_s for partially-visible target.

VLM scores ALL ancestors of the target in one call. Downstream code reuses
these cached scores when simulating object removals.
"""
from __future__ import annotations

import networkx as nx
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


def compute_semantic_prior_all_ancestors(
    target_mid: int,
    perception,
    client: VLMClient | None = None,
) -> dict[str, Any]:
    """Score every ancestor of the target with one VLM call.

    Returns independent [0, 1] scores keyed by molmo_id. Downstream code
    (e.g. counterfactual entropy) reuses these without extra API calls.
    """
    g = perception.occlusion_graph
    if target_mid not in perception.molmo_to_node:
        print(f"[PRIOR] target {target_mid} not in graph -> empty")
        return {"scores": {}, "graspability": {}, "reason": "target is not in graph"}

    t_node = perception.molmo_to_node[target_mid]
    ancestor_nodes = nx.ancestors(g, t_node)
    ancestor_mids = sorted(
        perception.node_info[n]["molmo_id"] for n in ancestor_nodes
    )

    print(f"[PRIOR] target={target_mid}, all ancestors={ancestor_mids}")

    if not ancestor_mids:
        return {"scores": {}, "graspability": {}, "reason": "target has no ancestors"}
    if len(ancestor_mids)<=1:
        prompt_mode = getattr(perception, "prior_prompt_mode", "original")
        if prompt_mode == "graspability":
            graspability_payload = score_current_objects(
                ancestor_mids, perception, client=client
            )
            return {
                "scores": {mid: 1.0 for mid in ancestor_mids},
                "graspability": graspability_payload.get("graspability", {}),
                "graspability_part_id": graspability_payload.get("graspability_part_id", {}),
                "graspability_parts": graspability_payload.get("graspability_parts", {}),
                "reason": (
                    "single ancestor; assigned deterministic prior. "
                    + str(graspability_payload.get("reason") or "")
                ).strip(),
            }
        return {
            "scores": {mid: 1.0 for mid in ancestor_mids},
            "graspability": {mid: 1.0 for mid in ancestor_mids},
            "graspability_part_id": {mid: None for mid in ancestor_mids},
            "graspability_parts": {mid: {} for mid in ancestor_mids},
            "reason": "single ancestor; assigned deterministic prior",
        }

    target_label = "unknown"
    if target_mid in perception.molmo_to_node:
        target_label = perception.node_info[
            perception.molmo_to_node[target_mid]
        ]["label"]

    # Build candidate descriptors for the prompt.
    occluders = [
        {
            "mid": mid,
            "label": perception.node_info[
                perception.molmo_to_node[mid]
            ]["label"],
            "part_ids": _part_ids(perception, mid),
            "is_top_layer": g.in_degree(perception.molmo_to_node[mid]) == 0,
        }
        for mid in ancestor_mids
    ]

    # Full graph edges as mid-to-mid relations.
    relations: list[tuple[int, int]] = []
    for u, v in g.edges:
        u_mid = perception.node_info[u]["molmo_id"]
        v_mid = perception.node_info[v]["molmo_id"]
        relations.append((u_mid, v_mid))

    labeled_rgb = getattr(perception, "labeled_rgb", None)
    if labeled_rgb is None:
        labeled_rgb = getattr(perception, "rgb", None)
    if labeled_rgb is None:
        print("[PRIOR] no labeled_rgb -> 0.5 fallback for all")
        return {
            "scores": {mid: 0.5 for mid in ancestor_mids},
            "graspability": {mid: 1.0 for mid in ancestor_mids},
            "reason": "no labeled RGB image; used fallback scores",
        }
    if not isinstance(labeled_rgb, np.ndarray):
        labeled_rgb = np.asarray(labeled_rgb)

    print(f"[PRIOR] labeled_rgb shape={labeled_rgb.shape} -> calling VLM")

    c = client or _client()
    prompt_mode = getattr(perception, "prior_prompt_mode", "original")
    raw = c.score_occluders_partial(
        target_mid=target_mid,
        target_label=target_label,
        occluders=occluders,
        labeled_rgb=labeled_rgb,
        occlusion_relations=relations,
        parts_sheet_rgb=getattr(perception, "sam2_rgb_parts_sheet", None),
        prompt_mode=prompt_mode,
        object_sheet_rgb=getattr(perception, "final_objects_sheet", None),
        occlusion_graph_rgb=getattr(perception, "occlusion_graph_rgb", None),
    )

    # Fill missing ids with a neutral fallback.
    raw.setdefault("scores", {})
    raw.setdefault("graspability", {})
    raw.setdefault("graspability_part_id", {})
    raw.setdefault("graspability_parts", {})
    raw.setdefault("reason", "")
    for mid in ancestor_mids:
        raw["scores"].setdefault(mid, 0.5)
        raw["graspability"].setdefault(mid, 1.0)
        raw["graspability_part_id"].setdefault(mid, None)
        raw["graspability_parts"].setdefault(mid, {})
    return raw


# Backward-compatible alias (so old handler imports still work).
def compute_semantic_prior(
    occluder_mids: list[int],
    target_mid: int,
    perception,
    client: VLMClient | None = None,
) -> dict[int, float]:
    """Compatibility wrapper: returns scores for the requested mids only."""
    all_scores = compute_semantic_prior_all_ancestors(target_mid, perception, client)
    scores = all_scores.get("scores", {})
    return {mid: scores.get(mid, 0.5) for mid in occluder_mids}


def _part_ids(perception, mid: int) -> list[int]:
    mapping = getattr(perception, "object_id_to_sam2_part_ids", None) or {}
    return list(mapping.get(mid, ()))
