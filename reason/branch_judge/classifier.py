from __future__ import annotations

from ..schemas import PerceptionOutput, Branch



def classify_branch(perception: PerceptionOutput) -> tuple[Branch, str]:
    """Classify the target into one of the four reasoning branches."""
    target_mid = perception.target_molmo_id
    graph = perception.occlusion_graph

    # Case 1: target already appears in the graph.
    if target_mid in perception.molmo_to_node:
        node_id = perception.molmo_to_node[target_mid]
        in_degree = graph.in_degree(node_id)

        if in_degree == 0:
            return (
                Branch.FULLY_VISIBLE,
                f"target molmo_id={target_mid} (node_id={node_id}) "
                f"is in graph, in_degree=0 -> on top, no occluder",
            )

        obstructor_mids = [
            perception.node_info[n]["molmo_id"]
            for n in graph.predecessors(node_id)
        ]
        return (
            Branch.PARTIALLY_OCCLUDED,
            f"target molmo_id={target_mid} (node_id={node_id}) "
            f"is in graph, occluded by molmo_id={obstructor_mids}",
        )

    # Case 2: target is missing from the graph.
    preferred_occluders = set(perception.preferred_occluder_ids or ())
    visible_preferred_occluders = preferred_occluders.intersection(
        perception.molmo_to_node
    )
    if visible_preferred_occluders:
        return (
            Branch.FULLY_OCCLUDED,
            f"target molmo_id={target_mid} not in graph, but Intent identified "
            "visible occluder candidates="
            f"{sorted(visible_preferred_occluders)} -> likely fully occluded",
        )

    if graph.number_of_edges() == 0:
        return (
            Branch.FAULT,
            f"target molmo_id={target_mid} not in graph, "
            f"and graph has no occlusion edges -> target does not exist",
        )

    return (
        Branch.FULLY_OCCLUDED,
        f"target molmo_id={target_mid} not in graph, "
        f"but scene has {graph.number_of_edges()} occlusion edges "
        f"-> likely fully occluded",
    )
