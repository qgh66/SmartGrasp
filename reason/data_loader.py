"""Build ``PerceptionOutput`` from a scene ``summary.json``.

The loader rebuilds the occlusion graph, then attaches optional artifacts such
as depth, labeled RGB, and per-object masks if they exist on disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from PIL import Image

from .schemas import PerceptionOutput


def load_sample(
    summary_path: str | Path,
    task_type: str = "pick",
    occlusion_threshold: float = 0.0,
) -> PerceptionOutput:
    """Load one scene summary and return the reasoning input object."""
    summary_path = Path(summary_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    target_mid = summary.get("query_obj_id")
    molmo_points = summary.get("molmo_points", [])
    matrix = summary.get("occlusion_matrix", [])
    matrix_labels = summary.get("matrix_labels", [])

    graph, node_info, molmo_to_node = _build_graph(
        molmo_points, matrix, matrix_labels, threshold=occlusion_threshold
    )
    _attach_scene_artifacts(summary_path.parent, node_info)

    depth = _load_depth(summary.get("depth_path"), summary_path.parent)
    labeled_rgb = _load_labeled_rgb(summary_path.parent)

    return PerceptionOutput(
        target_molmo_id=target_mid,
        task_type=task_type,
        occlusion_graph=graph,
        node_info=node_info,
        molmo_to_node=molmo_to_node,
        depth=depth,
        labeled_rgb=labeled_rgb,
        scene_id=summary.get("scene_id"),
        annotation=summary.get("annotation"),
        point_source=summary.get("point_source"),
        output_dir=summary_path.parent,
    )


def _parse_id_to_index(matrix_labels) -> dict[int, int]:
    """Parse ``matrix_labels`` into ``molmo_id -> matrix index``."""
    mapping = {}
    for idx, label in enumerate(matrix_labels):
        prefix = str(label).split(":")[0].strip()
        try:
            molmo_id = int(prefix)
        except ValueError:
            molmo_id = idx + 1
        mapping[molmo_id] = idx
    return mapping


def _build_graph(
    molmo_points,
    matrix,
    matrix_labels,
    threshold: float = 0.0,
) -> tuple[nx.DiGraph, dict[int, dict], dict[int, int]]:
    """Build a directed occlusion graph from the summary matrix."""
    id_to_index = _parse_id_to_index(matrix_labels)
    index_to_id = {v: k for k, v in id_to_index.items()}
    id_to_label = {p["molmo_id"]: p.get("label", "") for p in molmo_points}

    g = nx.DiGraph()
    node_info: dict[int, dict] = {}
    molmo_to_node: dict[int, int] = {}

    n = len(matrix)
    for idx in range(n):
        mid = index_to_id.get(idx, idx + 1)
        node_id = idx
        g.add_node(node_id)
        node_info[node_id] = {
            "molmo_id": mid,
            "label": id_to_label.get(mid, f"object_{mid}"),
        }
        molmo_to_node[mid] = node_id

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = matrix[i][j]
            if val > threshold:
                g.add_edge(i, j, ratio=float(val))

    return g, node_info, molmo_to_node


def _load_depth(depth_path: str | None, scene_dir: Path) -> np.ndarray | None:
    if depth_path:
        candidate = Path(depth_path)
        if candidate.exists():
            return np.load(candidate)

    fallback = scene_dir / "depth.npy"
    if fallback.exists():
        return np.load(fallback)
    return None


def _load_labeled_rgb(scene_dir: Path) -> np.ndarray | None:
    """Load a labeled RGB overlay from common scene locations."""
    candidates: list[Path] = []

    # 1) Files next to the summary.
    candidates += [
        scene_dir / "label_3_final.png",
        scene_dir / "scene_labeled.png",
        scene_dir / "perception_label.png",
        scene_dir / "molmo_label.png",
    ]

    # 2) Files in the sibling ``perception/`` directory.
    sibling_perception = scene_dir.parent / "perception"
    if sibling_perception.exists() and sibling_perception != scene_dir:
        candidates += [
            sibling_perception / "label_3_final.png",
            sibling_perception / "scene_labeled.png",
            sibling_perception / "perception_label.png",
            sibling_perception / "molmo_label.png",
        ]

    # 3) Fallback to the sibling ``gt/`` directory.
    sibling_gt = scene_dir.parent / "gt"
    if sibling_gt.exists() and sibling_gt != scene_dir:
        candidates += [
            sibling_gt / "scene_labeled.png",
            sibling_gt / "perception_label.png",
        ]

    for path in candidates:
        if path.exists():
            return np.array(Image.open(path).convert("RGB"))

    return None


def _attach_scene_artifacts(scene_dir: Path, node_info: dict[int, dict]) -> None:
    """Attach per-object masks, bbox, and area if a mask directory exists."""
    mask_dir = scene_dir / "mask"
    if not mask_dir.exists():
        return

    for info in node_info.values():
        mid = int(info["molmo_id"])
        mask_path = _find_mask_path(mask_dir, mid)
        if mask_path is None:
            continue

        mask = _load_binary_mask(mask_path)
        bbox = _mask_bbox(mask)
        info["mask_path"] = str(mask_path.resolve())
        info["mask"] = mask
        info["area_px"] = int(mask.sum())
        info["bbox"] = bbox


def _find_mask_path(mask_dir: Path, molmo_id: int) -> Path | None:
    matches = sorted(mask_dir.glob(f"mask_{molmo_id:03d}_*.png"))
    if matches:
        return matches[0]
    exact = mask_dir / f"mask_{molmo_id:03d}.png"
    return exact if exact.exists() else None


def _load_binary_mask(mask_path: Path, threshold: int = 127) -> np.ndarray:
    """Load a binary object mask from a grayscale image."""
    image = Image.open(mask_path).convert("L")
    return np.array(image) > threshold


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(x, y, w, h)`` for the nonzero region of a binary mask."""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
