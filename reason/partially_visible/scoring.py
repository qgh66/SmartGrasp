"""TBM fusion, Shannon entropy, and entropy-based information gain."""
from __future__ import annotations

import math
from collections import deque

from .geometry import compute_geometric_prior, top_level_ancestors_after


def tbm_fusion(
    P_s: dict[int, float],
    P_g: dict[int, float],
) -> dict[int, float]:
    """Fuse semantic and geometric priors into a normalized belief."""
    raw = {mid: P_s[mid] * P_g[mid] for mid in P_s}
    total = sum(raw.values())
    if total <= 0:
        n = len(raw)
        return {mid: 1.0 / n for mid in raw} if n > 0 else {}
    return {mid: v / total for mid, v in raw.items()}


def entropy(P: dict[int, float]) -> float:
    """Shannon entropy in bits; zero probs are skipped."""
    h = 0.0
    for p in P.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


def information_gain(
    occluder_mid: int,
    perception,
    P_prior: dict[int, float],
    P_s_cache: dict[int, float],
) -> float:
    """Shannon IG: H(P_prior) - H(P_after) on the top-level candidate set.

    VLM is NOT called here. The cached P_s scores (covering all ancestors)
    are reused to rebuild the belief after removing the candidate.
    """
    target_mid = perception.target_molmo_id
    H_prior = entropy(P_prior)

    # New top-level candidates after removing this object.
    new_candidates = top_level_ancestors_after(
        perception, target_mid, removed_mids={occluder_mid}
    )

    if not new_candidates:
        # Target became directly graspable.
        H_after = 0.0
    else:
        # Reuse cached VLM scores for the new candidates.
        P_s_new = {mid: P_s_cache.get(mid, 0.5) for mid in new_candidates}
        # Recompute geometry on the residual graph.
        P_g_new = compute_geometric_prior(
            new_candidates,
            target_mid,
            perception,
            removed_mids={occluder_mid},
        )
        P_after = tbm_fusion(P_s_new, P_g_new)
        H_after = entropy(P_after)

    return H_prior - H_after


def compute_cost(occluder_mid: int, perception) -> int:
    """Approximate removal cost as the depth of objects pressing on the candidate."""
    g = perception.occlusion_graph
    start = perception.molmo_to_node[occluder_mid]

    queue = deque([(start, 0)])
    visited = {start}
    max_depth = 0
    while queue:
        node, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        for pred in g.predecessors(node):
            if pred not in visited:
                visited.add(pred)
                queue.append((pred, depth + 1))
    return max_depth


def compute_score(
    ig_value: float,
    cost: int,
    belief: float,
    alpha: float = 0.1,
) -> float:
    """Final score: belief-weighted IG minus a small cost penalty."""
    return belief * ig_value - alpha * cost