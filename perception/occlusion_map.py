"""Occlusion graph building: mask finalization, renumbering, graph construction, top-level orchestrator."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from SmartGrasp.perception._shared import (
    _draw_mask_records_label, _load_depth_map,
    _log_step, _mask_centroid_xy, _prepare_mask_output_dir,
    _safe_label, _save_mask_png, _write_json, SMARTGRASP_ROOT,
)
from SmartGrasp.perception.background import (
    generate_background_exclusion_mask_from_source,
)
from SmartGrasp.perception.langsam import LANGSAM_MIN_AREA_RATIO

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINALIZE_OVERLAP_LOSS_RATIO = 0.10  # If a mask loses >90% of its area during overlap resolution, treat as noise
FINALIZE_CULLED_AREA_RATIO = 0.003  # Absolute minimum (0.3%) for a heavily-culled mask to survive


def _save_background_exclusion_mask(
    background_exclusion_mask: np.ndarray | None,
    output_mask_dir: Path,
) -> str | None:
    if background_exclusion_mask is None or int(np.count_nonzero(background_exclusion_mask)) == 0:
        return None
    background_mask_path = output_mask_dir / "000_background_mask.png"
    _save_mask_png(np.asarray(background_exclusion_mask, dtype=bool), background_mask_path)
    return str(background_mask_path.resolve())


def _finalize_independent_scene_masks(
    mask_records: list[dict[str, Any]],
    background_exclusion_mask: np.ndarray | None,
    image_shape: tuple[int, int],
    containment_threshold: float = 0.92,
    overlap_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Make final masks non-overlapping and report finalization quality."""
    image_area = max(1, int(image_shape[0] * image_shape[1]))
    background_area = (
        int(np.count_nonzero(background_exclusion_mask))
        if background_exclusion_mask is not None
        else 0
    )
    coverage = {
        "background_mask_coverage_ratio": float(background_area / image_area),
        "background_mask_available": bool(background_area > 0),
        "overlap_pixels_after_finalize": 0,
    }
    if not mask_records:
        return mask_records, {
            "dedup_report": [],
            "overlap_report": [],
            "coverage": coverage,
        }

    kept = list(mask_records)
    dedup_report: list[dict[str, Any]] = []
    masks = [np.asarray(record["mask_array"], dtype=bool).copy() for record in kept]
    areas = [int(np.count_nonzero(mask)) for mask in masks]
    explicit_scores = [
        0 if str(record.get("segmentation_backend", "")) in ("langsam", "fusion") else 1
        for record in kept
    ]
    order = sorted(range(len(masks)), key=lambda index: (explicit_scores[index], areas[index]))
    owner = np.full(image_shape, -1, dtype=np.int32)
    overlap_report: list[dict[str, Any]] = []

    for index in order:
        mask = masks[index]
        overlap = mask & (owner >= 0)
        overlap_pixels = int(np.count_nonzero(overlap))
        if overlap_pixels > 0:
            owner_ids, counts = np.unique(owner[overlap], return_counts=True)
            overlap_report.append(
                {
                    "object_id": int(kept[index].get("object_id", index + 1)),
                    "removed_overlap_pixels": overlap_pixels,
                    "removed_overlap_fraction": float(overlap_pixels / max(1, areas[index])),
                    "overlap_with": [
                        {
                            "object_id": int(kept[int(owner_id)].get("object_id", int(owner_id) + 1)),
                            "pixels": int(count),
                        }
                        for owner_id, count in zip(owner_ids, counts)
                        if int(owner_id) >= 0
                    ],
                }
            )
            mask = mask & (owner < 0)
        masks[index] = mask
        owner[mask] = index

    finalized: list[dict[str, Any]] = []
    removed_empty: list[dict[str, Any]] = []
    image_area = max(1.0, float(image_shape[0] * image_shape[1]))
    for index, record in enumerate(kept):
        mask = masks[index]
        area = int(np.count_nonzero(mask))
        area_ratio = float(area) / image_area
        original_area = max(1, areas[index])
        overlap_loss = 1.0 - float(area) / float(original_area)
        # Heavily culled mask: lost >90% during overlap resolution and remaining is tiny → noise
        heavily_culled = overlap_loss > (1.0 - FINALIZE_OVERLAP_LOSS_RATIO) and area_ratio < FINALIZE_CULLED_AREA_RATIO
        if area == 0 or heavily_culled or area_ratio < LANGSAM_MIN_AREA_RATIO:
            removed = {key: value for key, value in record.items() if key != "mask_array"}
            if heavily_culled:
                removed["duplicate_reason"] = "removed_heavily_culled_by_overlap"
                removed["overlap_loss_ratio"] = float(overlap_loss)
                removed["original_area"] = int(original_area)
            elif area == 0:
                removed["duplicate_reason"] = "removed_after_overlap_exclusivity"
            else:
                removed["duplicate_reason"] = "removed_too_small_after_overlap"
            if area_ratio < LANGSAM_MIN_AREA_RATIO and area > 0:
                removed["removed_area_ratio"] = float(area_ratio)
                removed["min_area_ratio_threshold"] = float(LANGSAM_MIN_AREA_RATIO)
            removed_empty.append(removed)
            continue
        record["mask_array"] = mask
        record["mask_area"] = area
        cx, cy = _mask_centroid_xy(mask)
        record["point"] = {"x": int(cx), "y": int(cy)}
        old_path = Path(str(record.get("mask_path", "")))
        if old_path.exists():
            _save_mask_png(mask, old_path)
        finalized.append(record)

    return finalized, {
        "dedup_report": dedup_report + removed_empty,
        "overlap_report": overlap_report,
        "coverage": coverage,
        "containment_threshold": float(containment_threshold),
        "overlap_threshold": float(overlap_threshold),
    }



def _renumber_masks(mask_records: list[dict[str, Any]], output_mask_dir: Path) -> list[dict[str, Any]]:
    """Renumber object_id/node_id sequentially (1,2,3...), rename mask files on disk, keep labels in sync."""
    renumbered: list[dict[str, Any]] = []
    for index, record in enumerate(mask_records, start=1):
        old_path = Path(str(record.get("mask_path", "")))
        label = _safe_label(str(record.get("label", f"object_{index}")))
        source = record.get("segmentation_backend", "anchor")
        new_name = f"{index:03d}_{source}_{label}.png"
        new_path = output_mask_dir / new_name

        # Remove stale files with same index but different source/label
        for pattern in (f"{index:03d}_*.png", f"mask_{index:03d}_*.png"):
            for leftover in output_mask_dir.glob(pattern):
                if leftover != new_path and leftover != old_path:
                    leftover.unlink()

        if old_path.exists() and old_path != new_path:
            old_path.replace(new_path)

        record["node_id"] = index - 1
        record["object_id"] = index
        record["point"] = {"x": int(record.get("point", {}).get("x", 0)), "y": int(record.get("point", {}).get("y", 0))}
        record["mask_path"] = str(new_path.resolve())
        renumbered.append(record)

    # Clean up orphaned files (proposals rejected by dedup, old-format files)
    kept_names = {Path(str(r["mask_path"])).name for r in renumbered}
    kept_names.add("000_background_mask.png")
    for f in output_mask_dir.glob("*.png"):
        if f.name not in kept_names:
            f.unlink()

    return renumbered


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
    output_path = SMARTGRASP_ROOT / "perception" / "org_demo_graph.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved graph visualization to: {output_path}")


def build_org_json(
    image_path: Path,
    depth_path: Path,
    output_json_path: Path,
    output_mask_dir: Path,
    review_model_id: str = "gpt-5.5",
    review_api_key_env: str = "OPENAI_API_KEY",
    review_base_url: str | None = None,
    review_timeout: float = 120.0,
    epsilon: float = 0.05,
    kernel_size: int = 5,
    min_contact_pixels: int = 50,
    min_contact_ratio: float = 0.002,
    mask_clean_kernel: int = 3,
    proposal_min_area_ratio: float = 0.006,
    proposal_max_area_ratio: float = 0.11,
    proposal_border_fraction_threshold: float = 0.18,
    save_candidates: bool = False,
    device: str | None = None,
    sam2_points_per_side: int | None = 30,
    sam2_crop_n_layers: int | None = 0,
    sam2_pred_iou_thresh: float | None = 0.75,
    sam2_stability_score_thresh: float | None = 0.90,
    depth_sam2_points_per_side: int | None = None,
    depth_sam2_crop_n_layers: int | None = None,
    depth_sam2_pred_iou_thresh: float | None = None,
    depth_sam2_stability_score_thresh: float | None = None,
    preserve_unclaimed_sam2: int = 24,
    background_mask_source: str = "depth",
    gt_instances_objects: np.ndarray | None = None,
) -> dict[str, Any]:
    t0 = _log_step("start", None)

    _prepare_mask_output_dir(output_mask_dir, save_candidates)
    depth_map = _load_depth_map(depth_path)
    background_exclusion_mask: np.ndarray | None = None
    try:
        background_exclusion_mask = generate_background_exclusion_mask_from_source(
            mask_source=background_mask_source,
            depth_map=depth_map,
            image=Image.open(image_path).convert("RGB"),
            instances_objects=gt_instances_objects,
            mask_clean_kernel=mask_clean_kernel,
        )
    except Exception as exc:
        print(f"Background exclusion mask generation failed: {exc}", file=sys.stderr, flush=True)
    background_mask_path = _save_background_exclusion_mask(background_exclusion_mask, output_mask_dir)
    t1 = _log_step("① background_mask", t0)

    from SmartGrasp.perception.sam2auto import generate_masks_with_sam2_langsam_pipeline  # lazy import

    mask_records, anchor_report = generate_masks_with_sam2_langsam_pipeline(
        image_path=image_path,
        output_mask_dir=output_mask_dir,
        review_model_id=review_model_id,
        review_api_key_env=review_api_key_env,
        review_base_url=review_base_url,
        review_timeout=review_timeout,
        min_area_ratio=proposal_min_area_ratio,
        max_area_ratio=proposal_max_area_ratio,
        mask_clean_kernel=mask_clean_kernel,
        save_candidates=save_candidates,
        device=device,
        background_exclusion_mask=background_exclusion_mask,
        depth_map=depth_map,
        sam2_points_per_side=sam2_points_per_side,
        sam2_crop_n_layers=sam2_crop_n_layers,
        sam2_pred_iou_thresh=sam2_pred_iou_thresh,
        sam2_stability_score_thresh=sam2_stability_score_thresh,
        depth_sam2_points_per_side=depth_sam2_points_per_side,
        depth_sam2_crop_n_layers=depth_sam2_crop_n_layers,
        depth_sam2_pred_iou_thresh=depth_sam2_pred_iou_thresh,
        depth_sam2_stability_score_thresh=depth_sam2_stability_score_thresh,
        proposal_border_fraction_threshold=proposal_border_fraction_threshold,
        preserve_unclaimed_sam2=preserve_unclaimed_sam2,
    )
    t2 = _log_step("② sam2+vlm+langsam_pipeline", t1)

    mask_records = _renumber_masks(mask_records, output_mask_dir)
    _draw_mask_records_label(
        image_path=image_path,
        mask_records=mask_records,
        out_path=output_mask_dir.parent / "label_2_VLM_langsam.png",
    )

    final_mask_quality_report: dict[str, Any] = {}
    mask_records, final_mask_quality_report = _finalize_independent_scene_masks(
        mask_records=mask_records,
        background_exclusion_mask=background_exclusion_mask,
        image_shape=tuple(depth_map.shape),
    )
    mask_records = _renumber_masks(mask_records, output_mask_dir)
    t3 = _log_step("③ finalize_non_overlap", t2)

    _draw_mask_records_label(
        image_path=image_path,
        mask_records=mask_records,
        out_path=output_mask_dir.parent / "label_3_final.png",
    )

    masks = np.stack([record["mask_array"] for record in mask_records], axis=0)

    if masks.shape[1:] != depth_map.shape:
        raise ValueError(
            f"Mask shape {masks.shape[1:]} does not match depth map shape {depth_map.shape}."
        )

    graph, adjacency = build_occlusion_graph(
        masks=masks,
        depth_map=depth_map,
        epsilon=epsilon,
        kernel_size=kernel_size,
        min_contact_pixels=min_contact_pixels,
        min_contact_ratio=min_contact_ratio,
    )

    node_records: list[dict[str, Any]] = []
    for record in mask_records:
        node_record = dict(record)
        node_record.pop("mask_array", None)
        node_records.append(node_record)

    graph_payload = graph_to_jsonable(graph, adjacency, node_records=node_records)
    with Image.open(image_path) as img:
        width, height = img.size
    payload = {
        "image": {
            "path": str(image_path.resolve()),
            "width": int(width),
            "height": int(height),
        },
        "depth_map": {
            "path": str(depth_path.resolve()),
            "shape": [int(depth_map.shape[0]), int(depth_map.shape[1])],
        },
        "segmentation_backend": "anchor-langsam-fusion",
        "anchor_report": anchor_report,
        "final_mask_quality_report": final_mask_quality_report,
        "background_mask_path": background_mask_path,
        "background_mask_source": background_mask_source,
        "save_candidates": bool(save_candidates),
        "graph": graph_payload,
    }

    for edge in payload["graph"]["edges"]:
        source_node = node_records[edge["source"]]
        target_node = node_records[edge["target"]]
        edge["source_object_id"] = int(source_node["object_id"])
        edge["target_object_id"] = int(target_node["object_id"])
        edge["source_label"] = str(source_node["label"])
        edge["target_label"] = str(target_node["label"])

    _write_json(output_json_path, payload)

    t4 = _log_step("④ occlusion_graph", t3)
    _log_step("total", t0)
    return payload
