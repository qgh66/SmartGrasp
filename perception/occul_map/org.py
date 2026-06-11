"""Build an occlusion relationship graph (ORG) from masks and a depth map.

Depth convention for this project:
- smaller depth value means the object is closer to the camera
- in a top-down scene, closer depth means the object is higher / on top
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only in cv2-less environments
    cv2 = None

SMARTGRASP_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OcclusionEdgeInfo:
    """Metadata for an occlusion edge."""

    contact_pixels: int
    contact_ratio: float
    depth_i_median: float
    depth_j_median: float
    depth_gap: float


def _binary_dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Binary dilation with a cv2 fast path and a pure NumPy fallback."""

    mask_bool = np.asarray(mask, dtype=bool)
    if cv2 is not None:
        kernel = _make_kernel(kernel_size)
        return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0

    pad = kernel_size // 2
    padded = np.pad(mask_bool, pad_width=pad, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask_bool, dtype=bool)
    for row_offset in range(kernel_size):
        for col_offset in range(kernel_size):
            dilated |= padded[row_offset : row_offset + mask_bool.shape[0], col_offset : col_offset + mask_bool.shape[1]]
    return dilated


def _validate_inputs(masks: np.ndarray, depth_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(masks)
    depth_map = np.asarray(depth_map)

    if masks.ndim != 3:
        raise ValueError(f"`masks` must have shape (N, H, W), got {masks.shape}.")
    if depth_map.ndim != 2:
        raise ValueError(f"`depth_map` must have shape (H, W), got {depth_map.shape}.")
    if masks.shape[1:] != depth_map.shape:
        raise ValueError(
            "`masks` spatial shape must match `depth_map` shape: "
            f"got {masks.shape[1:]} vs {depth_map.shape}."
        )

    return masks.astype(bool, copy=False), depth_map.astype(np.float32, copy=False)


def _make_kernel(kernel_size: int) -> np.ndarray:
    if kernel_size < 1:
        raise ValueError(f"`kernel_size` must be >= 1, got {kernel_size}.")
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


def find_contact_area(
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Return the boolean contact area between two masks after light dilation."""

    dilated_i = _binary_dilate(mask_i, kernel_size=kernel_size)
    dilated_j = _binary_dilate(mask_j, kernel_size=kernel_size)
    return dilated_i & dilated_j


def _finite_mask(depth_values: np.ndarray) -> np.ndarray:
    return np.isfinite(depth_values) & (depth_values > 0)


def _contact_depth_median(mask: np.ndarray, contact_area: np.ndarray, depth_map: np.ndarray) -> Optional[float]:
    depth_values = depth_map[mask & contact_area]
    valid = _finite_mask(depth_values)
    if not np.any(valid):
        return None
    return float(np.median(depth_values[valid]))


def compute_adjacency_matrix(graph: nx.DiGraph, num_nodes: Optional[int] = None) -> np.ndarray:
    """Return the adjacency matrix of the directed occlusion graph."""

    if num_nodes is None:
        nodes = list(graph.nodes())
    else:
        nodes = list(range(num_nodes))
        for node in nodes:
            if node not in graph:
                graph.add_node(node)

    return nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8)


def graph_to_jsonable(
    graph: nx.DiGraph,
    adjacency_matrix: np.ndarray,
    node_records: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Convert an ORG into a JSON-serializable dictionary."""

    nodes_payload: list[dict[str, Any]]
    if node_records is None:
        nodes_payload = [{"node_id": int(node)} for node in graph.nodes()]
    else:
        nodes_payload = []
        for node in node_records:
            node_payload = dict(node)
            if "node_id" in node_payload:
                node_payload["node_id"] = int(node_payload["node_id"])
            if "object_id" in node_payload:
                node_payload["object_id"] = int(node_payload["object_id"])
            nodes_payload.append(node_payload)

    edges_payload: list[dict[str, Any]] = []
    for source, target, data in graph.edges(data=True):
        info = data.get("info")
        edge_payload = {
            "source": int(source),
            "target": int(target),
            "relation": "occludes",
            "direction": "source_occludes_target",
        }
        if info is not None:
            edge_payload.update(
                {
                    "contact_pixels": int(info.contact_pixels),
                    "contact_ratio": float(info.contact_ratio),
                    "source_depth_median": float(info.depth_i_median),
                    "target_depth_median": float(info.depth_j_median),
                    "depth_gap": float(info.depth_gap),
                }
            )
        edges_payload.append(edge_payload)

    return {
        "graph_type": "occlusion_relationship_graph",
        "adjacency_matrix": np.asarray(adjacency_matrix, dtype=np.uint8).tolist(),
        "nodes": nodes_payload,
        "edges": edges_payload,
    }


def build_occlusion_graph(
    masks: np.ndarray,
    depth_map: np.ndarray,
    epsilon: float = 0.05,
    kernel_size: int = 5,
    min_contact_pixels: int = 50,
    min_contact_ratio: float = 0.002,
) -> tuple[nx.DiGraph, np.ndarray]:
    """Build an ORG from instance masks and a depth map.

    Args:
        masks: Boolean-like array with shape (N, H, W).
        depth_map: Depth array with shape (H, W). Smaller values mean closer / higher.
        epsilon: Minimum depth gap required to consider the occlusion relation reliable.
        kernel_size: Dilation kernel size used to detect contact areas.
        min_contact_pixels: Ignore weak contacts smaller than this many pixels.
        min_contact_ratio: Ignore contacts smaller than this fraction of the smaller object mask.

    Returns:
        graph: A directed graph where edge i -> j means i occludes j.
        adjacency_matrix: NxN binary adjacency matrix.
    """

    masks, depth_map = _validate_inputs(masks, depth_map)

    if epsilon < 0:
        raise ValueError(f"`epsilon` must be >= 0, got {epsilon}.")
    if min_contact_pixels < 1:
        raise ValueError(f"`min_contact_pixels` must be >= 1, got {min_contact_pixels}.")
    if min_contact_ratio < 0:
        raise ValueError(f"`min_contact_ratio` must be >= 0, got {min_contact_ratio}.")

    num_objects = masks.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_objects))
    mask_areas = [int(np.count_nonzero(masks[index])) for index in range(num_objects)]

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            contact_area = find_contact_area(masks[i], masks[j], kernel_size=kernel_size)
            contact_pixels = int(np.count_nonzero(contact_area))
            if contact_pixels < min_contact_pixels:
                continue
            min_area = max(1, min(mask_areas[i], mask_areas[j]))
            contact_ratio = contact_pixels / min_area
            if contact_ratio < min_contact_ratio:
                continue

            depth_i = _contact_depth_median(masks[i], contact_area, depth_map)
            depth_j = _contact_depth_median(masks[j], contact_area, depth_map)
            if depth_i is None or depth_j is None:
                continue

            if depth_i < depth_j - epsilon:
                graph.add_edge(
                    i,
                    j,
                    info=OcclusionEdgeInfo(
                        contact_pixels=contact_pixels,
                        contact_ratio=contact_ratio,
                        depth_i_median=depth_i,
                        depth_j_median=depth_j,
                        depth_gap=depth_j - depth_i,
                    ),
                )
            elif depth_j < depth_i - epsilon:
                graph.add_edge(
                    j,
                    i,
                    info=OcclusionEdgeInfo(
                        contact_pixels=contact_pixels,
                        contact_ratio=contact_ratio,
                        depth_i_median=depth_j,
                        depth_j_median=depth_i,
                        depth_gap=depth_i - depth_j,
                    ),
                )

    adjacency = compute_adjacency_matrix(graph, num_nodes=num_objects)
    return graph, adjacency


def visualize_occlusion_graph(
    graph: nx.DiGraph,
    ax: Optional[plt.Axes] = None,
    title: str = "Occlusion Relationship Graph",
    with_edge_labels: bool = True,
) -> plt.Axes:
    """Visualize the directed occlusion graph."""

    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(6, 5))

    assert ax is not None

    if graph.number_of_nodes() == 0:
        ax.set_title(title)
        ax.axis("off")
        return ax

    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        node_color="#cfe8ff",
        edge_color="#1f4e79",
        node_size=1800,
        arrows=True,
        arrowsize=18,
        linewidths=1.2,
        font_size=10,
    )

    if with_edge_labels and graph.number_of_edges() > 0:
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            info = data.get("info")
            if info is None:
                continue
            edge_labels[(u, v)] = f"gap={info.depth_gap:.3f}\npx={info.contact_pixels}"
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8)

    ax.set_title(title)
    ax.axis("off")
    return ax


def _build_demo_case() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic case for quick manual verification."""

    height, width = 80, 80
    masks = np.zeros((3, height, width), dtype=np.uint8)
    depth_map = np.full((height, width), 10.0, dtype=np.float32)

    masks[0, 18:48, 16:36] = 1
    depth_map[masks[0] > 0] = 1.0

    masks[1, 28:58, 30:56] = 1
    depth_map[masks[1] > 0] = 2.0

    masks[2, 8:18, 58:72] = 1
    depth_map[masks[2] > 0] = 0.8

    return masks, depth_map


if __name__ == "__main__":
    demo_masks, demo_depth = _build_demo_case()
    demo_graph, demo_adjacency = build_occlusion_graph(
        demo_masks,
        demo_depth,
        epsilon=0.05,
        kernel_size=5,
        min_contact_pixels=3,
    )

    print("Adjacency matrix:")
    print(demo_adjacency.astype(int))

    visualize_occlusion_graph(demo_graph)
    plt.tight_layout()
    output_path = SMARTGRASP_ROOT / "perception" / "occul_map" / "org_demo_graph.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved graph visualization to: {output_path}")
