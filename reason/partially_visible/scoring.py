from __future__ import annotations

import math
from collections import deque

import networkx as nx


def tbm_fusion(
    P_s: dict[int, float],
    P_g: dict[int, float],
) -> dict[int, float]:
    """Fuse semantic and geometric scores into a normalized belief."""
    raw = {mid: P_s[mid] * P_g[mid] for mid in P_s}
    total = sum(raw.values())
    if total <= 0:
        n = len(raw)
        return {mid: 1.0 / n for mid in raw} if n > 0 else {}
    return {mid: v / total for mid, v in raw.items()}


def _upstream_levels(g: nx.DiGraph, target_node: int) -> dict[int, int]:
    """Assign a top-down level to the target and all of its ancestors."""
    relevant_nodes = nx.ancestors(g, target_node) | {target_node}
    subgraph = g.subgraph(relevant_nodes).copy()

    levels: dict[int, int] = {}
    for node in nx.topological_sort(subgraph):
        preds = list(subgraph.predecessors(node))
        if not preds:
            levels[node] = 0
        else:
            levels[node] = 1 + max(levels[pred] for pred in preds)
    return levels


def information_gain(
    occluder_mid: int,
    perception,
    belief: dict[int, float] | None = None,
) -> float:
    """Estimate structural gain after removing one visible occluder."""
    if occluder_mid not in perception.molmo_to_node:
        raise KeyError(f"candidate {occluder_mid} not found in occlusion graph")

    target_mid = perception.target_molmo_id
    if target_mid not in perception.molmo_to_node:
        raise KeyError(f"target {target_mid} not found in occlusion graph")

    g = perception.occlusion_graph
    target_node = perception.molmo_to_node[target_mid]
    occluder_node = perception.molmo_to_node[occluder_mid]

    current_levels = _upstream_levels(g, target_node)

    new_graph = g.copy()
    new_graph.remove_node(occluder_node)
    next_levels = _upstream_levels(new_graph, target_node)

    shared_nodes = [
        node for node in current_levels
        if node != occluder_node and node in next_levels
    ]

    level_drop = sum(
        max(0, current_levels[node] - next_levels[node])
        for node in shared_nodes
    )
    promoted_to_top = sum(
        1
        for node in shared_nodes
        if current_levels[node] > 0 and next_levels[node] == 0
    )

    target_bonus = 1.0 if next_levels.get(target_node, 0) == 0 else 0.0
    belief_bonus = 0.0 if belief is None else 0.1 * belief.get(occluder_mid, 0.0)

    return float(level_drop) + 0.5 * promoted_to_top + target_bonus + belief_bonus


def compute_cost(occluder_mid: int, perception) -> int:
    """Approximate removal cost as the number of layers above the object."""
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
    alpha: float = 0.1,
    belief: float = 1.0,
) -> float:
    """Combine structural gain and cost into a final ranking score."""
    return belief * ig_value - alpha * cost
