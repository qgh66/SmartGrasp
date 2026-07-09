from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def precompute_geometry_cache(perception) -> dict:
    """Cache per-object area and height proxies for invisible reasoning."""
    if perception.depth is None:
        raise ValueError("perception.depth is required for invisible geometry")

    depth = perception.depth

    # Estimate the table/ground depth from pixels outside all masks.
    all_masks = np.zeros_like(depth, dtype=bool)
    for info in perception.node_info.values():
        all_masks |= np.asarray(info["mask"], dtype=bool)
    table_region = ~all_masks

    if table_region.any():
        ground_level = float(np.median(depth[table_region]))
    else:
            ground_level = float(np.percentile(depth, 95))

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
    """Fixed area estimate: average of self + all objects pressing on top."""
    g = perception.occlusion_graph
    node = perception.molmo_to_node[mid]
    
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
