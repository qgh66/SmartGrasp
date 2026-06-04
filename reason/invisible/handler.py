from __future__ import annotations

from ..schemas import PerceptionOutput, GraspDecision, Branch
from .geometry import compute_geometric_prior, precompute_geometry_cache
from .prior import compute_semantic_prior
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
    P_s = compute_semantic_prior(candidate_mids, target_mid, perception)
    P_g = compute_geometric_prior(
        candidate_mids,
        target_mid,
        perception,
        geom_cache=geom_cache,
        removed_mids=None,
    )
    P = tbm_fusion(P_s, P_g)

    # Expected IG measures how much a miss would simplify the next step.
    details: dict[int, dict] = {}
    for mid in candidate_mids:
        ig_value = expected_information_gain(mid, P, perception, geom_cache)
        details[mid] = {
            "P_s": P_s[mid],
            "P_g": P_g[mid],
            "P": P[mid],
            "IG": ig_value,
        }

    # Pick the candidate with the strongest expected gain.
    best_mid = max(
        details,
        key=lambda m: (details[m]["IG"], details[m]["P"], -m),
    )
    best_node = perception.molmo_to_node[best_mid]
    best_label = perception.node_info[best_node]["label"]

    lines = [f"selected mid={best_mid} ({best_label})", "candidates:"]
    for mid in candidate_mids:
        d = details[mid]
        mark = " <-- selected" if mid == best_mid else ""
        lines.append(
            f"  mid={mid}: P_s={d['P_s']:.3f} P_g={d['P_g']:.4f} "
            f"P={d['P']:.4f} IG={d['IG']:.4f}{mark}"
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
