"""Partial-occlusion branch: pick the best top-layer occluder to remove."""
from __future__ import annotations

import networkx as nx
import math

from ..schemas import PerceptionOutput, GraspDecision, Branch
from .prior import compute_semantic_prior_all_ancestors
from .geometry import compute_geometric_prior
from .scoring import (
    information_gain,
    tbm_fusion,
)


def handle(perception: PerceptionOutput) -> GraspDecision:
    """Choose the next visible occluder to remove for a partial target."""
    target_mid = perception.target_molmo_id
    g = perception.occlusion_graph

    if target_mid not in perception.molmo_to_node:
        return GraspDecision(
            branch=Branch.PARTIALLY_OCCLUDED,
            target_molmo_id=target_mid,
            is_terminal=False,
            success=False,
            message=f"target molmo_id={target_mid} not in graph",
        )

    t_node = perception.molmo_to_node[target_mid]
    ancestors = nx.ancestors(g, t_node)
    candidate_nodes = [n for n in ancestors if g.in_degree(n) == 0]
    if not candidate_nodes:
        return GraspDecision(
            branch=Branch.PARTIALLY_OCCLUDED,
            target_molmo_id=target_mid,
            is_terminal=False,
            success=False,
            message=f"no top-layer ancestor found for target {target_mid}",
        )

    candidate_mids = sorted(
        perception.node_info[n]["molmo_id"] for n in candidate_nodes
    )

    # One VLM call: score every ancestor and, optionally, graspability.
    prior_payload = compute_semantic_prior_all_ancestors(target_mid, perception)
    P_s_all = prior_payload["scores"]
    graspability_all = prior_payload["graspability"]
    graspability_part_id_all = prior_payload.get("graspability_part_id", {})
    graspability_parts_all = prior_payload.get("graspability_parts", {})
    vlm_reason = str(prior_payload.get("reason") or "")
    P_s_top = {mid: P_s_all.get(mid, 0.5) for mid in candidate_mids}
    graspability_top = {mid: graspability_all.get(mid, 1.0) for mid in candidate_mids}
    graspability_part_id_top = {
        mid: graspability_part_id_all.get(mid) for mid in candidate_mids
    }
    graspability_parts_top = {
        mid: graspability_parts_all.get(mid, {}) for mid in candidate_mids
    }

    # Initial belief on top-level candidates only.
    P_g_top = compute_geometric_prior(candidate_mids, target_mid, perception)
    P_prior = tbm_fusion(P_s_top, P_g_top)

    # Shannon IG per candidate; reuses P_s_all to avoid extra VLM calls.
    details: dict[int, dict] = {}
    ranking_score = getattr(perception, "ranking_score", "legacy")
    norm = math.log2(max(2, len(candidate_mids)))
    for mid in candidate_mids:
        ig_value = information_gain(mid, perception, P_prior, P_s_all)
        score_legacy = P_prior[mid] * ig_value
        score_ig = ig_value
        score_ig_graspability = ig_value * graspability_top[mid]
        ig_normalized = max(0.0, ig_value) / norm
        score_theory = graspability_top[mid] * P_prior[mid] * ig_normalized
        score = _select_score(
            ranking_score,
            legacy=score_legacy,
            ig=score_ig,
            ig_graspability=score_ig_graspability,
            theory=score_theory,
        )
        details[mid] = {
            "P_s": P_s_top[mid],
            "P_g": P_g_top[mid],
            "P": P_prior[mid],
            "IG": ig_value,
            "IG_normalized": ig_normalized,
            "graspability": graspability_top[mid],
            "graspability_part_id": graspability_part_id_top[mid],
            "graspability_parts": graspability_parts_top[mid],
            "score_legacy": score_legacy,
            "score_ig": score_ig,
            "score_ig_graspability": score_ig_graspability,
            "score_theory": score_theory,
            "score": score,
            "vlm_reason": vlm_reason,
        }

    best_mid = max(
        details,
        key=lambda m: (details[m]["score"], details[m]["IG"], details[m]["P"], -m),
    )
    best_node = perception.molmo_to_node[best_mid]
    best_label = perception.node_info[best_node]["label"]

    # Compact debug string.
    lines = [
        f"selected mid={best_mid} ({best_label})",
        f"vlm_reason={vlm_reason}" if vlm_reason else "vlm_reason=",
        "candidates:",
    ]
    for mid in candidate_mids:
        d = details[mid]
        mark = " <-- selected" if mid == best_mid else ""
        lines.append(
            f"  mid={mid}: P_s={d['P_s']:.3f} P_g={d['P_g']:.4f} "
            f"P={d['P']:.4f} IG={d['IG']:.4f} "
            f"G={d['graspability']:.3f} best_part={d['graspability_part_id']} "
            f"score={d['score']:.4f}{mark}"
        )
    message = "  ".join(lines)

    return GraspDecision(
        branch=Branch.PARTIALLY_OCCLUDED,
        grasp_id=best_mid,
        grasp_label=best_label,
        target_molmo_id=target_mid,
        is_terminal=False,
        success=True,
        message=message,
        details=details,
    )


def _select_score(
    ranking_score: str,
    *,
    legacy: float,
    ig: float,
    ig_graspability: float,
    theory: float,
) -> float:
    if ranking_score == "ig":
        return ig
    if ranking_score == "ig_graspability":
        return ig_graspability
    if ranking_score == "theory":
        return theory
    return legacy
