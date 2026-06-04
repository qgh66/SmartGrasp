"""P_s for partially-visible target.

Calls a VLM with labeled_rgb + occlusion relations to score each occluder.
Returns independent [0, 1] scores (NOT normalized) — tbm_fusion normalizes later.
"""
from __future__ import annotations

import numpy as np

from ..vlm import VLMClient, get_default_client


# Reuse one client instance across calls.
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
    """Return VLM-based semantic scores for partially visible occluders."""
    print(f"[PRIOR] called: target={target_mid}, occluders={occluder_mids}")

    if not occluder_mids:
        print("[PRIOR] no occluders -> empty")
        return {}

    # Read the visible target label if it exists in the graph.
    target_label = "unknown"
    if target_mid in perception.molmo_to_node:
        target_label = perception.node_info[
            perception.molmo_to_node[target_mid]
        ]["label"]

    # Build compact descriptors for the VLM prompt.
    occluders = []
    for mid in occluder_mids:
        info = perception.node_info[perception.molmo_to_node[mid]]
        occluders.append({"mid": mid, "label": info["label"]})

    # Export graph edges as mid-to-mid relations.
    g = perception.occlusion_graph
    relations: list[tuple[int, int]] = []
    for u, v in g.edges:
        u_mid = perception.node_info[u]["molmo_id"]
        v_mid = perception.node_info[v]["molmo_id"]
        relations.append((u_mid, v_mid))

    # Labeled RGB is required by the current VLM prompt.
    labeled_rgb = getattr(perception, "labeled_rgb", None)
    if labeled_rgb is None:
        labeled_rgb = getattr(perception, "rgb", None)

    if labeled_rgb is None:
        print("[PRIOR] no labeled_rgb available -> 0.5 fallback")
        return {mid: 0.5 for mid in occluder_mids}

    if not isinstance(labeled_rgb, np.ndarray):
        labeled_rgb = np.asarray(labeled_rgb)

    print(f"[PRIOR] labeled_rgb loaded, shape={labeled_rgb.shape} -> calling VLM")

    # Query the VLM and keep scores in [0, 1] without normalization.
    c = client or _client()
    raw = c.score_occluders_partial(
        target_mid=target_mid,
        target_label=target_label,
        occluders=occluders,
        labeled_rgb=labeled_rgb,
        occlusion_relations=relations,
    )

    # Fill any missing candidate with a neutral fallback.
    for mid in occluder_mids:
        raw.setdefault(mid, 0.5)

    return raw
