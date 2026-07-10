from __future__ import annotations
import math
from .geometry import (
    compute_geometric_prior,
    top_level_candidates_after,
)
from .prior import compute_semantic_prior

def tbm_fusion(
    P_s: dict[int, float],
    P_g: dict[int, float],
) -> dict[int, float]:
    """Fuse semantic and geometry priors into a normalized belief."""
    raw = {mid: P_s[mid] * P_g[mid] for mid in P_s}
    total = sum(raw.values())
    if total <= 0:
        n = len(raw)
        return {mid: 1.0 / n for mid in raw} if n > 0 else {}
    return {mid: v / total for mid, v in raw.items()}


def entropy(P: dict[int, float]) -> float:
    """Return the Shannon entropy of a discrete distribution."""
    h = 0.0
    for p in P.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


__all__ = ["entropy", "tbm_fusion", "expected_information_gain"]


def _belief_after_miss(
    occluder_mid: int,
    perception,
    geom_cache: dict,
) -> dict[int, float]:
    removed = {occluder_mid}
    new_mids = top_level_candidates_after(
        perception, removed_mids=removed, exclude_target=True
    )
    if not new_mids:
        return {}
    P_s = compute_semantic_prior(new_mids, perception.target_molmo_id, perception)
    P_g = compute_geometric_prior(
        new_mids,
        perception.target_molmo_id,
        perception,
        geom_cache=geom_cache,
        # removed_mids=removed,   # ← 删掉, area 不再依赖
    )
    return tbm_fusion(P_s, P_g)


def expected_information_gain(
    occluder_mid: int,
    belief: dict[int, float],
    perception,
    geom_cache: dict,
) -> float:
    """Compute expected information gain with a miss-side scene update."""
    if occluder_mid not in belief:
        raise KeyError(f"candidate {occluder_mid} not found in belief")

    H_prior = entropy(belief)
    p_oi = belief[occluder_mid]

    if p_oi >= 1.0 - 1e-12:
        return H_prior

    P_miss = _belief_after_miss(occluder_mid, perception, geom_cache)
    H_miss = entropy(P_miss) if P_miss else 0.0

    expected_H = (1.0 - p_oi) * H_miss
    return H_prior - expected_H
