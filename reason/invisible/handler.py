from __future__ import annotations

import math

from ..schemas import PerceptionOutput, GraspDecision, Branch
from .geometry import compute_geometric_prior, precompute_geometry_cache
from .prior import compute_semantic_prior_payload
from .scoring import expected_information_gain, tbm_fusion


def handle(perception: PerceptionOutput) -> GraspDecision:
    """Choose which visible top-level object to remove for a hidden target."""
    g = perception.occlusion_graph
    target_mid = perception.target_molmo_id

    # Only top-level visible objects can be removed in this branch.
    candidate_nodes = [n for n in g.nodes if g.in_degree(n) == 0]
    if target_mid in perception.molmo_to_node:
        t_node = perception.molmo_to_node[target_mid]
        candidate_nodes = [n for n in candidate_nodes if n != t_node]

    if not candidate_nodes:
        return GraspDecision(
            branch=Branch.FULLY_OCCLUDED,
            target_molmo_id=target_mid,
            is_terminal=False,
            success=False,
            message="no top-layer candidates available",
        )

    candidate_mids = sorted(
        perception.node_info[n]["molmo_id"] for n in candidate_nodes
    )

    # Cache geometry once because IG builds counterfactual scenes.
    geom_cache = precompute_geometry_cache(perception)

    # Combine VLM prior and geometry prior into one belief.
    prior_payload = compute_semantic_prior_payload(candidate_mids, target_mid, perception)
    P_s = prior_payload["scores"]
    graspability = prior_payload["graspability"]
    graspability_part_id = prior_payload.get("graspability_part_id", {})
    graspability_parts = prior_payload.get("graspability_parts", {})
    vlm_reason = str(prior_payload.get("reason") or "")
    P_g = compute_geometric_prior(
        candidate_mids,
        target_mid,
        perception,
        geom_cache=geom_cache,
    )
    P = tbm_fusion(P_s, P_g)

    # Expected IG measures how much a miss would simplify the next step.
    details: dict[int, dict] = {}
    ranking_score = getattr(perception, "ranking_score", "legacy")
    norm = math.log2(max(2, len(candidate_mids)))
    for mid in candidate_mids:
        ig_value = expected_information_gain(mid, P, perception, geom_cache)
        score_legacy = ig_value
        score_ig = ig_value
        score_ig_graspability = ig_value * graspability.get(mid, 1.0)
        ig_normalized = max(0.0, ig_value) / norm
        score_theory = ig_normalized * graspability.get(mid, 1.0)
        score = _select_score(
            ranking_score,
            legacy=score_legacy,
            ig=score_ig,
            ig_graspability=score_ig_graspability,
            theory=score_theory,
        )
        details[mid] = {
            "P_s": P_s[mid],
            "P_g": P_g[mid],
            "P": P[mid],
            "IG": ig_value,
            "IG_normalized": ig_normalized,
            "graspability": graspability.get(mid, 1.0),
            "graspability_part_id": graspability_part_id.get(mid),
            "graspability_parts": graspability_parts.get(mid, {}),
            "score_legacy": score_legacy,
            "score_ig": score_ig,
            "score_ig_graspability": score_ig_graspability,
            "score_theory": score_theory,
            "score": score,
            "vlm_reason": vlm_reason,
        }

    # Pick the candidate with the strongest expected gain.
    best_mid = max(
        details,
        key=lambda m: (details[m]["score"], details[m]["IG"], details[m]["P"], -m),
    )
    best_node = perception.molmo_to_node[best_mid]
    best_label = perception.node_info[best_node]["label"]

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
        branch=Branch.FULLY_OCCLUDED,
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
    if ranking_score == "legacy":
        return legacy
    if ranking_score == "ig":
        return ig
    if ranking_score == "ig_graspability":
        return ig_graspability
    if ranking_score == "theory":
        return theory
    return legacy
