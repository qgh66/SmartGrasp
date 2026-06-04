from __future__ import annotations

import networkx as nx


def compute_geometric_prior(
    occluder_mids: list[int],
    target_mid: int,
    perception,
) -> dict[int, float]:
    """Return an unnormalized geometric score for each visible occluder."""
    g = perception.occlusion_graph
    t_node = perception.molmo_to_node[target_mid]

    out = {}
    for mid in occluder_mids:
        o_node = perception.molmo_to_node[mid]
        out[mid] = _path_product_sum(g, o_node, t_node)
    return out


def _path_product_sum(g: nx.DiGraph, source: int, target: int) -> float:
    """Sum the edge-ratio products over all simple paths to the target."""
    if source == target:
        return 0.0

    total = 0.0
    for path in nx.all_simple_paths(g, source=source, target=target):
        product = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            ratio = float(g[u][v].get("ratio", 0.0))
            product *= ratio
        total += product
    return total
