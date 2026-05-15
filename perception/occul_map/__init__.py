"""Occlusion relationship graph utilities."""

__all__ = [
    "build_occlusion_graph",
    "compute_adjacency_matrix",
    "find_contact_area",
    "graph_to_jsonable",
    "visualize_occlusion_graph",
]


def __getattr__(name: str):
    if name in __all__:
        from .org import (
            build_occlusion_graph,
            compute_adjacency_matrix,
            find_contact_area,
            graph_to_jsonable,
            visualize_occlusion_graph,
        )

        namespace = {
            "build_occlusion_graph": build_occlusion_graph,
            "compute_adjacency_matrix": compute_adjacency_matrix,
            "find_contact_area": find_contact_area,
            "graph_to_jsonable": graph_to_jsonable,
            "visualize_occlusion_graph": visualize_occlusion_graph,
        }
        return namespace[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
