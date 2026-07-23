from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from perception.background import load_tray_interior_mask


EQUIVALENT_AREA_KEY = "equivalent_area_px"


def precompute_geometry_cache(perception) -> dict:
    """Cache per-object area and height proxies for invisible reasoning."""
    if perception.depth is None:
        raise ValueError("perception.depth is required for invisible geometry")

    depth = perception.depth

    # Estimate the tray-floor depth only from valid, uncovered pixels inside
    # the black rectangle enclosed by data/tray_border_mask.png.
    all_masks = np.zeros_like(depth, dtype=bool)
    for info in perception.node_info.values():
        all_masks |= np.asarray(info["mask"], dtype=bool)
    valid_depth = np.isfinite(depth) & (depth > 0)
    tray_interior = load_tray_interior_mask(depth.shape)
    if tray_interior is None:
        table_region = (~all_masks) & valid_depth
    else:
        table_region = tray_interior & (~all_masks) & valid_depth

    if table_region.any():
        ground_level = float(np.median(depth[table_region]))
    elif valid_depth.any():
        # Exceptional fallback for a missing/fully covered tray region.  Keep
        # the previous robust far-depth behavior rather than producing NaN.
        ground_level = float(np.percentile(depth[valid_depth], 95))
    else:
        raise ValueError("depth contains no finite positive values")

    cache: dict = {"__ground_level__": ground_level}

    # Cache visible area and a height proxy for each object.
    for node, info in perception.node_info.items():
        mid = info["molmo_id"]
        mask = np.asarray(info["mask"], dtype=bool)

        visible_area = float(mask.sum())
        if mask.any():
            d_top = float(depth[mask].mean())
            height = max(1.0, ground_level - d_top)
        else:
            d_top = 0.0
            height = 1.0

        cache[mid] = {
            "visible_area": visible_area,
            "visible_depth": d_top,
            "height": height,
        }

    return cache


def equivalent_area(
    mid: int,
    perception,
    geom_cache: dict,
) -> float:
    """Return a persistent area proxy for one currently selectable object.

    When an object becomes top-level after its sole direct occluder is removed,
    ``simulate_remove`` records the estimate that was available immediately
    before removal.  Reuse that estimate in later closed-loop steps instead of
    falling back to the object's originally visible pixels only.
    """
    g = perception.occlusion_graph
    node = perception.molmo_to_node[mid]

    recorded = perception.node_info[node].get(EQUIVALENT_AREA_KEY)
    if recorded is not None:
        return float(recorded)
    
    # All objects pressing on top of this candidate (not filtered by removed)
    above_nodes = list(g.predecessors(node))
    
    visible_area = geom_cache[mid]["visible_area"]
    if not above_nodes:
        return visible_area
    
    above_area_sum = sum(
        geom_cache[perception.node_info[n]["molmo_id"]]["visible_area"]
        for n in above_nodes
    )
    
    # New formula: average over self + all above
    return (visible_area + above_area_sum) / (len(above_nodes) + 1)


def newly_exposed_equivalent_areas(
    perception,
    removed_mid: int,
) -> dict[int, float]:
    """Record area proxies for children exposed by removing ``removed_mid``.

    A child becomes top-level only when the removed node was its sole direct
    predecessor in the current graph.  Its saved area follows the existing
    equivalent-area rule, which in this case is exactly
    ``(visible_area(child) + visible_area(removed)) / 2``.
    """
    if removed_mid not in perception.molmo_to_node:
        return {}

    g = perception.occlusion_graph
    removed_node = perception.molmo_to_node[removed_mid]
    removed_info = perception.node_info[removed_node]
    removed_area = float(np.asarray(removed_info["mask"], dtype=bool).sum())

    exposed: dict[int, float] = {}
    for child_node in g.successors(removed_node):
        predecessors = list(g.predecessors(child_node))
        remaining = [node for node in predecessors if node != removed_node]
        if remaining:
            continue

        child_info = perception.node_info[child_node]
        child_mid = int(child_info["molmo_id"])
        child_area = float(np.asarray(child_info["mask"], dtype=bool).sum())
        exposed[child_mid] = (child_area + removed_area) / 2.0

    return exposed

def equivalent_height(mid: int, geom_cache: dict) -> float:
    """Return the cached height proxy for one object."""
    return geom_cache[mid]["height"]

def compute_geometric_prior(
    occluder_mids: list[int],
    target_mid: int,
    perception,
    geom_cache: dict | None = None,
) -> dict[int, float]:
    _ = target_mid
    if geom_cache is None:
        geom_cache = precompute_geometry_cache(perception)
    
    raw_scores: dict[int, float] = {}
    for mid in occluder_mids:
        area = equivalent_area(mid, perception, geom_cache)  
        height = equivalent_height(mid, geom_cache)
        raw_scores[mid] = area * height
    
    total = sum(raw_scores.values())
    if total <= 0:
        n = len(raw_scores)
        return {mid: 1.0 / n for mid in raw_scores} if n > 0 else {}
    return {mid: v / total for mid, v in raw_scores.items()}

def _dilated_ring(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Return the outer ring after dilating a binary mask."""
    dilated = binary_dilation(mask, iterations=k)
    return dilated & (~mask)


def top_level_candidates_after(
    perception,
    removed_mids: set[int],
    exclude_target: bool = True,
) -> list[int]:
    """Return top-level candidates after removing a set of object ids."""
    g = perception.occlusion_graph
    target_mid = perception.target_molmo_id

    candidates: list[int] = []
    for n in g.nodes:
        mid_n = perception.node_info[n]["molmo_id"]
        if mid_n in removed_mids:
            continue
        if exclude_target and mid_n == target_mid:
            continue
        still_pressed = [
            p for p in g.predecessors(n)
            if perception.node_info[p]["molmo_id"] not in removed_mids
        ]
        if not still_pressed:
            candidates.append(mid_n)
    return sorted(candidates)
