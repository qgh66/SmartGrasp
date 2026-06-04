from __future__ import annotations

import networkx as nx

from ..schemas import PerceptionOutput, GraspDecision, Branch
from .prior import compute_semantic_prior
from .geometry import compute_geometric_prior
from .scoring import (
    compute_cost,
    compute_score,
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

    P_s = compute_semantic_prior(candidate_mids, target_mid, perception)
    P_g = compute_geometric_prior(candidate_mids, target_mid, perception)
    P = tbm_fusion(P_s, P_g)

    details: dict[int, dict] = {}
    for mid in candidate_mids:
        ig_value = information_gain(mid, perception, belief=P)
        cost = compute_cost(mid, perception)
        score = compute_score(ig_value, cost, belief=P[mid])
        details[mid] = {
            "P_s": P_s[mid],
            "P_g": P_g[mid],
            "P": P[mid],
            "IG": ig_value,
            "cost": cost,
            "score": score,
        }

    best_mid = max(
        details,
        key=lambda m: (details[m]["score"], details[m]["IG"], details[m]["P"], -m),
    )
    best_node = perception.molmo_to_node[best_mid]
    best_label = perception.node_info[best_node]["label"]

    # Build a compact debug string for downstream inspection.
    lines = [
        f"selected mid={best_mid} ({best_label})",
        "candidates:",
    ]
    for mid in candidate_mids:
        d = details[mid]
        mark = " <-- selected" if mid == best_mid else ""
        lines.append(
            f"  mid={mid}: P_s={d['P_s']:.3f} P_g={d['P_g']:.4f} "
            f"P={d['P']:.4f} IG={d['IG']:.4f} "
            f"cost={d['cost']} score={d['score']:.4f}{mark}"
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
