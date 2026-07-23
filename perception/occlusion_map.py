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
    _log_step, _prepare_mask_output_dir,
    _safe_label, _save_mask_png, _write_json, SMARTGRASP_ROOT,
)
from SmartGrasp.perception.background import generate_background_exclusion_mask

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

def _save_background_exclusion_mask(
    background_exclusion_mask: np.ndarray | None,
    output_mask_dir: Path,
) -> str | None:
    if background_exclusion_mask is None or int(np.count_nonzero(background_exclusion_mask)) == 0:
        return None
    background_mask_path = output_mask_dir / "000_background_mask.png"
    _save_mask_png(np.asarray(background_exclusion_mask, dtype=bool), background_mask_path)
    return str(background_mask_path.resolve())


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
    contact_background_pixels: int = 0
    contact_background_ratio: float = 0.0


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


def _band_depth_stats(
    mask: np.ndarray,
    other_mask: np.ndarray,
    depth_map: np.ndarray,
    band_lo: int = 2,
    band_hi: int = 9,
) -> Optional[tuple[float, float, float]]:
    """Return (p25, p50, p75) of depth in *mask* pixels within [band_lo, band_hi)
    pixels of *other_mask* (close to the boundary but avoiding the noisiest edge).
    Falls back to whole-mask stats when the band is too small.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~other_mask)
        band = mask & (dist >= band_lo) & (dist < band_hi)
        if np.count_nonzero(band) < 10:
            band = mask
    except ImportError:
        band = mask

    vals = depth_map[band]
    valid = np.isfinite(vals) & (vals > 0)
    if not np.any(valid):
        return None
    p25, p50, p75 = np.percentile(vals[valid], [25, 50, 75])
    return float(p25), float(p50), float(p75)


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
                    "contact_background_pixels": int(info.contact_background_pixels),
                    "contact_background_ratio": float(info.contact_background_ratio),
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
    kernel_size: int = 11,
    min_contact_pixels: int = 100,
    min_contact_ratio: float = 0.005,
    depth_gap_threshold: float = 0.5,
    band_lo: int = 2,
    band_hi: int = 9,
    background_mask: np.ndarray | None = None,
    max_contact_background_ratio: float = 0.4,
) -> tuple[nx.DiGraph, np.ndarray]:
    """Build an ORG from instance masks and a depth map.

    Contact detected via dilation; depth measured in a narrow band
    [band_lo, band_hi) px from the other mask's boundary, close to the
    occlusion frontier but avoiding sensor-interpolation noise at 0–3 px.
    An edge i→j is added when the band median of i is at least
    *depth_gap_threshold* shallower than that of j.

    Each mask is dilated **once** and reused for every pair.

    When *background_mask* is supplied, contact-area pixels that fall inside
    the background are counted.  If the fraction of background pixels in the
    contact zone exceeds *max_contact_background_ratio* the pair is treated
    as having **no effective contact** and no occlusion edge is created.

    Args:
        masks: Boolean-like array (N, H, W).
        depth_map: Depth array (H, W).  Smaller = closer.
        kernel_size: Dilation kernel for contact detection.
        min_contact_pixels: Minimum contact pixels.
        min_contact_ratio: Minimum contact ratio.
        depth_gap_threshold: Minimum band-median gap for occlusion.
        band_lo: Inner radius of measurement band (px from other mask).
        band_hi: Outer radius of measurement band.
        background_mask: Optional boolean background exclusion mask (H, W).
        max_contact_background_ratio: Maximum allowed background fraction
            in the contact zone; exceed → skip the pair.

    Returns:
        graph: A directed graph where edge i -> j means i occludes j.
        adjacency_matrix: NxN binary adjacency matrix.
    """

    masks, depth_map = _validate_inputs(masks, depth_map)
    if background_mask is not None:
        bg = np.asarray(background_mask, dtype=bool)
        if bg.shape != depth_map.shape:
            raise ValueError(
                f"background_mask shape {bg.shape} must match depth_map shape {depth_map.shape}."
            )
    else:
        bg = None

    if min_contact_pixels < 1:
        raise ValueError(f"`min_contact_pixels` must be >= 1, got {min_contact_pixels}.")
    if min_contact_ratio < 0:
        raise ValueError(f"`min_contact_ratio` must be >= 0, got {min_contact_ratio}.")

    num_objects = masks.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_objects))
    mask_areas = [int(np.count_nonzero(masks[index])) for index in range(num_objects)]

    # ── Pre-dilate every mask once (O(N) instead of O(N²)) ──────────────
    dilated_masks = [_binary_dilate(masks[idx], kernel_size) for idx in range(num_objects)]

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            contact_area = dilated_masks[i] & dilated_masks[j]
            contact_pixels = int(np.count_nonzero(contact_area))
            if contact_pixels < min_contact_pixels:
                continue
            min_area = max(1, min(mask_areas[i], mask_areas[j]))
            contact_ratio = contact_pixels / min_area
            if contact_ratio < min_contact_ratio:
                continue

            # ── Background-ratio gate ──────────────────────────────────
            contact_background_pixels = 0
            contact_background_ratio = 0.0
            if bg is not None and int(np.count_nonzero(bg)) > 0:
                contact_background_pixels = int(np.count_nonzero(contact_area & bg))
                contact_background_ratio = contact_background_pixels / contact_pixels
                if contact_background_ratio > max_contact_background_ratio:
                    continue  # no effective contact — skip depth judgement

            stats_i = _band_depth_stats(masks[i], masks[j], depth_map, band_lo, band_hi)
            stats_j = _band_depth_stats(masks[j], masks[i], depth_map, band_lo, band_hi)
            if stats_i is None or stats_j is None:
                continue

            p25_i, p50_i, p75_i = stats_i
            p25_j, p50_j, p75_j = stats_j

            # Occlusion when far-side median gap is large enough
            if p50_i + depth_gap_threshold < p50_j:
                graph.add_edge(
                    i, j,
                    info=OcclusionEdgeInfo(
                        contact_pixels=contact_pixels,
                        contact_ratio=contact_ratio,
                        depth_i_median=p50_i,
                        depth_j_median=p50_j,
                        depth_gap=p50_j - p50_i,
                        contact_background_pixels=contact_background_pixels,
                        contact_background_ratio=contact_background_ratio,
                    ),
                )
            elif p50_j + depth_gap_threshold < p50_i:
                graph.add_edge(
                    j, i,
                    info=OcclusionEdgeInfo(
                        contact_pixels=contact_pixels,
                        contact_ratio=contact_ratio,
                        depth_i_median=p50_j,
                        depth_j_median=p50_i,
                        depth_gap=p50_i - p50_j,
                        contact_background_pixels=contact_background_pixels,
                        contact_background_ratio=contact_background_ratio,
                    ),
                )

    adjacency = compute_adjacency_matrix(graph, num_nodes=num_objects)
    return graph, adjacency


def visualize_occlusion_graph(
    graph: nx.DiGraph,
    ax: Optional[plt.Axes] = None,
    title: str = "Occlusion Relationship Graph",
    positions: dict[int, tuple[float, float]] | None = None,
    node_labels: dict[int, str] | None = None,
    background_image: np.ndarray | None = None,
) -> plt.Axes:
    """Visualize the directed occlusion graph.

    Args:
        positions: Optional dict node_id → (x, y) in data coordinates.
        node_labels: Optional dict node_id → display label string.
        background_image: Optional RGB image array (H, W, 3) to render as background.
    """

    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(10, 8))

    assert ax is not None

    if graph.number_of_nodes() == 0:
        ax.set_title(title)
        ax.axis("off")
        return ax

    # --- background image ---
    if background_image is not None:
        ax.imshow(background_image, origin="upper")

    # --- layout ---
    if positions is not None:
        pos = {n: positions[n] for n in graph.nodes()}
        if background_image is not None:
            # Let the background image define the coordinate system
            h, w = background_image.shape[:2]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        else:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            x_margin = max(20, (max(xs) - min(xs)) * 0.06) if len(xs) > 1 else 20
            y_margin = max(20, (max(ys) - min(ys)) * 0.06) if len(ys) > 1 else 20
            ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
            ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
        ax.set_aspect("equal")
    else:
        pos = nx.spring_layout(graph, seed=42)

    # --- node labels ---
    labels = node_labels if node_labels is not None else {n: str(n) for n in graph.nodes()}

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        edge_color="#FFD700",
        arrows=True,
        arrowsize=22,
        width=2.5,
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos,
        ax=ax,
        labels=labels,
        font_size=11,
        font_weight="bold",
        font_color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", edgecolor="none", alpha=0.65),
    )

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
    kernel_size: int = 11,
    min_contact_pixels: int = 100,
    min_contact_ratio: float = 0.005,
    depth_gap_threshold: float = 0.5,
    band_lo: int = 2,
    band_hi: int = 9,
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
    max_contact_background_ratio: float = 0.4,
) -> dict[str, Any]:
    t0 = _log_step("start", None)

    _prepare_mask_output_dir(output_mask_dir, save_candidates)
    depth_map = _load_depth_map(depth_path)
    background_exclusion_mask: np.ndarray | None = None
    try:
        background_exclusion_mask = generate_background_exclusion_mask(
            depth_map=depth_map,
            image=Image.open(image_path).convert("RGB"),
            mask_clean_kernel=mask_clean_kernel,
        )
    except Exception as exc:
        print(f"Background exclusion mask generation failed: {exc}", file=sys.stderr, flush=True)
    background_mask_path = _save_background_exclusion_mask(background_exclusion_mask, output_mask_dir)
    t1 = _log_step("① background_mask", t0)

    from SmartGrasp.perception.sam2auto import generate_masks_with_sam2_vlm_pipeline  # lazy import

    mask_records, anchor_report = generate_masks_with_sam2_vlm_pipeline(
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
    )
    t2 = _log_step("② sam2+vlm_pipeline", t1)

    mask_records = _renumber_masks(mask_records, output_mask_dir)
    _draw_mask_records_label(
        image_path=image_path,
        mask_records=mask_records,
        out_path=output_mask_dir.parent / "label_2_vlm.png",
    )

    masks = np.stack([record["mask_array"] for record in mask_records], axis=0)

    if masks.shape[1:] != depth_map.shape:
        raise ValueError(
            f"Mask shape {masks.shape[1:]} does not match depth map shape {depth_map.shape}."
        )

    graph, adjacency = build_occlusion_graph(
        masks=masks,
        depth_map=depth_map,
        kernel_size=kernel_size,
        min_contact_pixels=min_contact_pixels,
        min_contact_ratio=min_contact_ratio,
        band_lo=band_lo,
        band_hi=band_hi,
        depth_gap_threshold=depth_gap_threshold,
        background_mask=background_exclusion_mask,
        max_contact_background_ratio=max_contact_background_ratio,
    )

    node_records: list[dict[str, Any]] = []
    for record in mask_records:
        node_record = dict(record)
        node_record.pop("mask_array", None)
        node_records.append(node_record)

    graph_payload = graph_to_jsonable(graph, adjacency, node_records=node_records)
    with Image.open(image_path) as img:
        width, height = img.size

    # --- visualize occlusion graph with spatial positions + layer colors ---
    viz_positions: dict[int, tuple[float, float]] = {}
    viz_labels: dict[int, str] = {}
    for rec in mask_records:
        nid = rec["node_id"]
        px, py = rec["point"]["x"], rec["point"]["y"]
        # Image coords: (0,0) top-left, imshow(origin='upper') matches this
        viz_positions[nid] = (float(px), float(py))
        viz_labels[nid] = str(rec["object_id"])
    # --- visualize with scene background ---
    fig, viz_ax = plt.subplots(figsize=(12, 9))
    # Load scene image as background
    bg_img: np.ndarray | None = None
    if image_path.exists():
        bg_img = np.array(Image.open(image_path).convert("RGB"))
    visualize_occlusion_graph(
        graph,
        ax=viz_ax,
        title=f"Occlusion Graph — {image_path.parent.name}",
        positions=viz_positions,
        node_labels=viz_labels,
        background_image=bg_img,
    )
    viz_path = output_mask_dir.parent / "occlusion_graph.png"
    fig.tight_layout()
    fig.savefig(viz_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _log_step("③ viz_graph", t2)

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
        "segmentation_backend": "anchor",
        "anchor_report": anchor_report,
        "background_mask_path": background_mask_path,
        "background_mask_source": "depth",
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

    _log_step("④ occlusion_graph", t2)
    _log_step("total", t0)
    return payload
