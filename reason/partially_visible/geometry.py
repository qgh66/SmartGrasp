"""Geometric prior via path-product over the occlusion graph."""
from __future__ import annotations

import networkx as nx


def compute_geometric_prior(
    occluder_mids: list[int],
    target_mid: int,
    perception,
    removed_mids: set[int] | None = None,
) -> dict[int, float]:
    """Path-product score; optionally simulating the removal of some nodes."""
    g = perception.occlusion_graph
    if removed_mids:
        # Counterfactual: remove specified nodes from the graph for this call.
        nodes_to_remove = {
            perception.molmo_to_node[m]
            for m in removed_mids if m in perception.molmo_to_node
        }
        g = g.copy()
        g.remove_nodes_from(nodes_to_remove)

    t_node = perception.molmo_to_node[target_mid]
    out = {}
    for mid in occluder_mids:
        if mid not in perception.molmo_to_node:
            out[mid] = 0.0
            continue
        o_node = perception.molmo_to_node[mid]
        if o_node not in g.nodes:
            out[mid] = 0.0
            continue
        out[mid] = _path_product_sum(g, o_node, t_node)
    return out


def _path_product_sum(g: nx.DiGraph, source: int, target: int) -> float:
    """Sum of edge-ratio products over all simple paths source -> target."""
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


def top_level_ancestors_after(
    perception,
    target_mid: int,
    removed_mids: set[int],
) -> list[int]:
    """Return top-level ancestors of target after simulated removals."""
    g = perception.occlusion_graph.copy()
    nodes_to_remove = {
        perception.molmo_to_node[m]
        for m in removed_mids if m in perception.molmo_to_node
    }
    g.remove_nodes_from(nodes_to_remove)

    t_node = perception.molmo_to_node.get(target_mid)
    if t_node is None or t_node not in g.nodes:
        return []

    ancestors = nx.ancestors(g, t_node)
    candidate_nodes = [n for n in ancestors if g.in_degree(n) == 0]
    return sorted(
        perception.node_info[n]["molmo_id"] for n in candidate_nodes
    )