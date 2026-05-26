from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SMARTGRASP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SMARTGRASP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

from SmartGrasp.perception.data_loader import DATA_DIR, PARQUET_GLOB, iter_npz_sources, load_npz
from SmartGrasp.perception.molmo.molmo_annotator.draw import draw_labeled_image_matplotlib
from SmartGrasp.perception.occul_map.org import build_occlusion_graph, graph_to_jsonable


OUT_ROOT = SMARTGRASP_ROOT / "data"
DEFAULT_MOLMO_PROMPT = (
    "Point out all objects in the green tray. "
    "Return one point for each visible object instance, including repeated objects that look similar or have the same category. "
    "Do not merge adjacent objects, even if they touch or have similar colors. "
    "Ignore the green tray/green box itself and any other background support surface. "
    "Do separate different instances even if they are very close or nearly touching."
    "Use one point near the center of the visible region of each object. "
    "Use short labels with a likely noun plus visible attributes such as color, shape, material, size, brand text, or pose. "
    "If the exact category is unclear, describe visible attributes, for example red round lid, yellow rectangular packet, blue cylindrical can, or small white plastic piece. "
    "Before finishing, check the image again for any missed partially visible object and for any accidentally marked background support surface."
)
MOLMO_SCAN_PROMPT_SUFFIXES = [
    "",
    (
        "Run a second independent full-scene scan. Divide the image into upper-left, upper-center, upper-right, "
        "middle-left, center, middle-right, lower-left, lower-center, and lower-right regions. "
        "Point out every physically separate visible object instance in those regions, including repeated objects. "
        "Ignore only the green tray/green box and other background support surfaces. Do not use any expected object count."
    ),
    (
        "Run a careful missed-object scan. Focus on small, partly hidden, overlapping, low-contrast, or similarly colored foreground objects. "
        "Mark separate physical instances separately even when they touch or overlap, including repeated objects. "
        "Ignore only the green tray/green box and other background support surfaces. Do not invent objects and do not use any expected object count."
    ),
]


def read_dataset() -> pd.DataFrame:
    parquet_files = sorted(Path(DATA_DIR).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found by {PARQUET_GLOB}")
    return pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)


def select_sample(df: pd.DataFrame, scene_id: int | None, query_obj_id: int | None) -> pd.Series:
    candidates = df
    if scene_id is not None:
        candidates = candidates[candidates["sceneId"].astype(int) == int(scene_id)]
    if query_obj_id is not None:
        candidates = candidates[candidates["queryObjId"].astype(int) == int(query_obj_id)]
    if candidates.empty:
        raise ValueError(f"No sample found for scene_id={scene_id}, query_obj_id={query_obj_id}")
    return candidates.iloc[0]


def find_npz_source(scene_id: int) -> tuple[Path, str | None]:
    for name, source_path, zip_member in iter_npz_sources():
        if Path(name).stem == str(scene_id):
            return Path(source_path), zip_member
    raise FileNotFoundError(f"No npz source found for sceneId={scene_id}")


def save_sample_image(row: pd.Series, out_dir: Path) -> Path:
    image_obj = row["image"]
    if isinstance(image_obj, dict) and image_obj.get("bytes"):
        image = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
    elif isinstance(image_obj, dict) and image_obj.get("path"):
        image = Image.open(image_obj["path"]).convert("RGB")
    else:
        raise ValueError("Unsupported image field in parquet row")

    image_path = out_dir / "scene_image.png"
    image.save(image_path)
    return image_path


def object_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot compute centroid for an empty object mask")
    return int(np.round(xs.mean())), int(np.round(ys.mean()))


def count_gt_objects(instances_objects: np.ndarray) -> int:
    return int(sum(1 for value in np.unique(instances_objects) if int(value) > 0))


def build_gt_points(instances_objects: np.ndarray, annotation: str, query_obj_id: int | None) -> list[dict[str, Any]]:
    object_ids = sorted(int(value) for value in np.unique(instances_objects) if int(value) > 0)
    points: list[dict[str, Any]] = []
    for object_id in object_ids:
        x, y = object_centroid(instances_objects == object_id)
        label = annotation if query_obj_id == object_id else f"object_{object_id}"
        points.append({"molmo_id": object_id, "x": x, "y": y, "label": label})
    return points


def write_points_json(
    out_dir: Path,
    image_path: Path,
    width: int,
    height: int,
    prompt: str,
    points: list[dict[str, Any]],
    mode: str,
) -> Path:
    payload = {
        "model_id": mode,
        "prompt": prompt,
        "image": {"path": str(image_path.resolve()), "width": int(width), "height": int(height)},
        "parse_mode": mode,
        "raw_model_output": "",
        "points": points,
    }
    path = out_dir / "molmo_points.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_depth(depth: np.ndarray, out_dir: Path) -> Path:
    path = out_dir / "depth.npy"
    np.save(path, depth.astype(np.float32, copy=False))
    return path


def copy_or_save_sample_image(row: pd.Series, source_image_path: Path | None, out_dir: Path) -> Path:
    if source_image_path is not None and source_image_path.exists():
        target_path = out_dir / "scene_image.png"
        shutil.copy2(source_image_path, target_path)
        return target_path
    return save_sample_image(row, out_dir)


def save_gt_masks(instances_objects: np.ndarray, out_dir: Path) -> list[dict[str, Any]]:
    mask_dir = out_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    object_ids = sorted(int(value) for value in np.unique(instances_objects) if int(value) > 0)
    for node_id, object_id in enumerate(object_ids):
        mask = instances_objects == object_id
        mask_path = mask_dir / f"mask_{object_id:03d}_gt.png"
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
        records.append(
            {
                "node_id": node_id,
                "molmo_id": object_id,
                "label": f"object_{object_id}",
                "point": {"x": int(object_centroid(mask)[0]), "y": int(object_centroid(mask)[1])},
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(mask)),
            }
        )
    return records


def visualize_graph(graph: nx.DiGraph, graph_json: dict[str, Any], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if graph.number_of_nodes() > 0:
        pos = directed_graph_layout(graph)
        labels = {}
        for node in graph.nodes():
            node_payload = graph_json["nodes"][int(node)]
            labels[node] = str(node_payload.get("molmo_id", node))
        nx.draw_networkx_nodes(graph, pos, node_color="#e7f0ff", edgecolors="#194b7a", linewidths=1.2, node_size=1500, ax=ax)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=24,
            width=2.4,
            edge_color="#194b7a",
            connectionstyle="arc3,rad=0.08",
            min_source_margin=18,
            min_target_margin=22,
            ax=ax,
        )
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            info = data.get("info")
            if info is not None:
                edge_labels[(u, v)] = f"gap={info.depth_gap:.3f}"
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title(f"{title}\narrow direction: occluder -> occluded")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize_graph_payload(graph_payload: dict[str, Any], out_path: Path, title: str) -> None:
    graph = nx.DiGraph()
    for node in graph_payload["nodes"]:
        graph.add_node(int(node["node_id"]))
    for edge in graph_payload["edges"]:
        graph.add_edge(int(edge["source"]), int(edge["target"]), payload=edge)

    fig, ax = plt.subplots(figsize=(8, 6))
    if graph.number_of_nodes() > 0:
        pos = directed_graph_layout(graph)
        labels = {int(node["node_id"]): str(node.get("molmo_id", node["node_id"])) for node in graph_payload["nodes"]}
        nx.draw_networkx_nodes(graph, pos, node_color="#e7f0ff", edgecolors="#194b7a", linewidths=1.2, node_size=1500, ax=ax)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=24,
            width=2.4,
            edge_color="#194b7a",
            connectionstyle="arc3,rad=0.08",
            min_source_margin=18,
            min_target_margin=22,
            ax=ax,
        )
        edge_labels = {}
        for source, target, data in graph.edges(data=True):
            edge = data["payload"]
            if "depth_gap" in edge:
                edge_labels[(source, target)] = f"gap={edge['depth_gap']:.3f}"
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title(f"{title}\narrow direction: occluder -> occluded")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def directed_graph_layout(graph: nx.DiGraph) -> dict[int, tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}

    try:
        generations = list(nx.topological_generations(graph))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(graph, seed=42)

    if not generations:
        return nx.spring_layout(graph, seed=42)

    pos: dict[int, tuple[float, float]] = {}
    max_width = max(len(generation) for generation in generations)
    for level, generation in enumerate(generations):
        nodes = sorted(int(node) for node in generation)
        if len(nodes) == 1:
            xs = [0.0]
        else:
            xs = np.linspace(-max_width / 2.0, max_width / 2.0, len(nodes)).tolist()
        y = -float(level)
        for node, x in zip(nodes, xs):
            pos[node] = (float(x), y)

    return pos


def build_graph_from_gt_masks(
    instances_objects: np.ndarray,
    depth: np.ndarray,
    out_dir: Path,
    epsilon: float,
    kernel_size: int,
    min_contact_pixels: int,
    min_contact_ratio: float,
) -> dict[str, Any]:
    node_records = save_gt_masks(instances_objects, out_dir)
    object_ids = [record["molmo_id"] for record in node_records]
    masks = np.stack([(instances_objects == object_id) for object_id in object_ids], axis=0)
    graph, adjacency = build_occlusion_graph(
        masks=masks,
        depth_map=depth,
        epsilon=epsilon,
        kernel_size=kernel_size,
        min_contact_pixels=min_contact_pixels,
        min_contact_ratio=min_contact_ratio,
    )
    graph_payload = graph_to_jsonable(graph, adjacency, node_records=node_records)
    for edge in graph_payload["edges"]:
        source_node = node_records[edge["source"]]
        target_node = node_records[edge["target"]]
        edge["source_molmo_id"] = int(source_node["molmo_id"])
        edge["target_molmo_id"] = int(target_node["molmo_id"])
        edge["source_label"] = str(source_node["label"])
        edge["target_label"] = str(target_node["label"])

    payload = {"graph": graph_payload, "mask_source": "instances_objects", "depth_source": "npz.depth"}
    out_json = out_dir / "occlusion_graph.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    visualize_graph(graph, graph_payload, out_dir / "occlusion_graph.png", "GT Mask Occlusion Graph")
    return payload


def merge_point_records(points_by_attempt: list[list[dict[str, Any]]], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_size
    merged: list[dict[str, Any]] = []
    same_label_thresh = max(18, int(min(width, height) * 0.03))
    any_label_thresh = max(8, int(min(width, height) * 0.015))
    for points in points_by_attempt:
        for point in points:
            x = int(point["x"])
            y = int(point["y"])
            label = sanitize_point_label(str(point.get("label", "")))
            norm_label = label.lower()
            duplicate = False
            for kept in merged:
                kept_label = str(kept.get("label", "")).lower()
                dist_sq = (x - int(kept["x"])) ** 2 + (y - int(kept["y"])) ** 2
                threshold = same_label_thresh if norm_label and norm_label == kept_label else any_label_thresh
                if dist_sq <= threshold**2:
                    duplicate = True
                    break
            if not duplicate:
                merged.append({"molmo_id": len(merged) + 1, "x": x, "y": y, "label": label})
    return merged


def write_merged_molmo_outputs(
    image_path: Path,
    prompt: str,
    out_dir: Path,
    model_id: str,
    points: list[dict[str, Any]],
) -> Path:
    with Image.open(image_path) as image:
        width, height = image.size
        draw_labeled_image_matplotlib(
            image=image,
            points_with_ids=[(int(point["molmo_id"]), int(point["x"]), int(point["y"])) for point in points],
            out_png_path=str(out_dir / "1_molmo_label_raw.png"),
        )
    payload = {
        "model_id": model_id,
        "prompt": prompt,
        "image": {"path": str(image_path), "width": int(width), "height": int(height)},
        "parse_mode": "merged_multi_scan",
        "raw_model_output": "",
        "points": points,
    }
    final_json_path = out_dir / "molmo_points.json"
    final_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_json_path


def write_final_perception_label(
    image_path: Path,
    graph_payload: dict[str, Any],
    out_dir: Path,
) -> Path:
    points_with_ids: list[tuple[int, int, int]] = []
    for node in graph_payload.get("graph", {}).get("nodes", []):
        point = node.get("point", {})
        if "x" not in point or "y" not in point:
            continue
        points_with_ids.append((int(node.get("molmo_id", node["node_id"])), int(point["x"]), int(point["y"])))

    out_path = out_dir / "molmo_label.png"
    with Image.open(image_path) as image:
        draw_labeled_image_matplotlib(
            image=image,
            points_with_ids=points_with_ids,
            out_png_path=str(out_path),
        )
        draw_labeled_image_matplotlib(
            image=image,
            points_with_ids=points_with_ids,
            out_png_path=str(out_dir / "perception_label.png"),
        )
    return out_path


def reset_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def maybe_run_molmo(
    image_path: Path,
    prompt: str,
    out_dir: Path,
    model_id: str,
    max_attempts: int = 3,
) -> Path:
    from SmartGrasp.perception.molmo.molmo_annotator import MolmoAnnotator

    annotator = MolmoAnnotator(model_id=model_id)
    points_by_attempt: list[list[dict[str, Any]]] = []
    attempts = max(1, min(max_attempts, len(MOLMO_SCAN_PROMPT_SUFFIXES)))

    for attempt_index in range(attempts):
        suffix = MOLMO_SCAN_PROMPT_SUFFIXES[attempt_index]
        attempt_prompt = prompt if not suffix else f"{prompt}\n\n{suffix}"
        result = annotator.annotate_to_folder(
            str(image_path),
            attempt_prompt,
            str(out_dir),
            labeled_png_name=f"molmo_label_attempt_{attempt_index + 1}.png",
            json_name=f"molmo_points_attempt_{attempt_index + 1}.json",
            return_base64=False,
        )
        points = result.get("json_data", {}).get("points", [])
        points_by_attempt.append(points)

    if not points_by_attempt:
        raise RuntimeError("Molmo did not return a usable result.")

    with Image.open(image_path) as image:
        image_size = image.size
    merged_points = merge_point_records(points_by_attempt, image_size)
    return write_merged_molmo_outputs(image_path, prompt, out_dir, model_id, merged_points)


def sanitize_point_label(label: str) -> str:
    parts = [part for part in str(label).replace("-", " ").replace("_", " ").split() if part]
    vague = {"unknown", "unknownproduct", "object", "item", "product", "thing", "container"}
    filtered = [part for part in parts if part.lower() not in vague]
    return " ".join(filtered or parts).strip() or "unlabeled visible object"


def sanitize_points_json(points_path: Path) -> None:
    payload = json.loads(points_path.read_text(encoding="utf-8"))
    for point in payload.get("points", []):
        if "label" in point:
            point["label"] = sanitize_point_label(str(point["label"]))
    points_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def display_label(label: Any) -> str:
    return str(label).replace("_", " ")


def edge_strength_label(contact_ratio: float) -> str:
    if contact_ratio >= 0.03:
        return "strong"
    if contact_ratio >= 0.01:
        return "medium"
    return "weak"


def build_summary_scene_graph(points_path: Path, graph_payload: dict[str, Any]) -> dict[str, Any]:
    points_payload = load_json_file(points_path)
    graph = graph_payload["graph"]
    nodes = sorted(graph.get("nodes", []), key=lambda node: int(node["node_id"]))
    edges = graph.get("edges", [])
    molmo_points = points_payload.get("points", [])

    object_order: list[dict[str, Any]] = []
    for matrix_index, node in enumerate(nodes):
        molmo_id = int(node.get("molmo_id", node["node_id"]))
        object_order.append(
            {
                "matrix_index": matrix_index,
                "node_id": int(node["node_id"]),
                "molmo_id": molmo_id,
                "label": display_label(node.get("label", f"object_{molmo_id}")),
            }
        )

    node_id_to_index = {item["node_id"]: item["matrix_index"] for item in object_order}
    size = len(object_order)
    occlusion_matrix = [[0.0 for _ in range(size)] for _ in range(size)]

    for edge in edges:
        source_index = node_id_to_index.get(int(edge["source"]))
        target_index = node_id_to_index.get(int(edge["target"]))
        if source_index is None or target_index is None:
            continue
        contact_ratio = float(edge.get("contact_ratio", 0.0))
        occlusion_matrix[source_index][target_index] = safe_float(contact_ratio) or 0.0

    matrix_labels = [f"{item['molmo_id']}: {item['label']}" for item in object_order]
    return {
        "molmo_points": molmo_points,
        "matrix_labels": matrix_labels,
        "occlusion_matrix_direction": "row object occludes column object",
        "occlusion_matrix_metric": "contact_ratio",
        "occlusion_matrix": occlusion_matrix,
    }


def build_gt_reference_outputs(
    row: pd.Series,
    scene_id: int,
    query_obj_id: int,
    annotation: str,
    instances_objects: np.ndarray,
    depth: np.ndarray,
    prompt: str,
    args: argparse.Namespace,
    source_image_path: Path | None = None,
) -> dict[str, Any]:
    gt_dir = OUT_ROOT / f"scene_{scene_id}" / "gt"
    reset_output_dir(gt_dir)
    image_path = copy_or_save_sample_image(row, source_image_path, gt_dir)
    with Image.open(image_path) as image:
        width, height = image.size
    depth_path = save_depth(depth, gt_dir)
    points = build_gt_points(instances_objects, annotation, query_obj_id)
    points_path = write_points_json(gt_dir, image_path, width, height, prompt, points, "gt_centers")
    graph_payload = build_graph_from_gt_masks(
        instances_objects=instances_objects,
        depth=depth,
        out_dir=gt_dir,
        epsilon=args.epsilon,
        kernel_size=args.kernel_size,
        min_contact_pixels=args.min_contact_pixels,
        min_contact_ratio=args.min_contact_ratio,
    )
    scene_graph_summary = build_summary_scene_graph(points_path, graph_payload)
    summary = {
        "scene_id": scene_id,
        "query_obj_id": query_obj_id,
        "annotation": annotation,
        "point_source": "gt-centers",
        "output_dir": str(gt_dir.resolve()),
        "image_path": str(image_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "points_json": str(points_path.resolve()),
        "graph_json": str((gt_dir / "occlusion_graph.json").resolve()),
        "graph_png": str((gt_dir / "occlusion_graph.png").resolve()),
        "num_nodes": len(graph_payload["graph"]["nodes"]),
        "num_edges": len(graph_payload["graph"]["edges"]),
        "gt_object_count": count_gt_objects(instances_objects),
        "image": {"path": str(image_path.resolve()), "width": int(width), "height": int(height)},
        **scene_graph_summary,
    }
    (gt_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_pipeline(args: argparse.Namespace, df: pd.DataFrame | None = None) -> dict[str, Any]:
    if df is None:
        df = read_dataset()
    row = select_sample(df, args.scene_id, args.query_obj_id)
    scene_id = int(row["sceneId"])
    query_obj_id = int(row["queryObjId"])

    scene_dir = OUT_ROOT / f"scene_{scene_id}"
    out_dir = scene_dir / ("perception" if args.point_source == "molmo" else "gt")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = save_sample_image(row, out_dir)
    with Image.open(image_path) as image:
        width, height = image.size

    npz_source, zip_member = find_npz_source(scene_id)
    with load_npz(npz_source, zip_member) as npz:
        depth = np.asarray(npz["depth"], dtype=np.float32)
        instances_objects = np.asarray(npz["instances_objects"])

    depth_path = save_depth(depth, out_dir)
    prompt = args.prompt or DEFAULT_MOLMO_PROMPT.format(annotation=row["annotation"])
    gt_summary = None
    if args.point_source == "molmo":
        gt_summary = build_gt_reference_outputs(
            row=row,
            scene_id=scene_id,
            query_obj_id=query_obj_id,
            annotation=str(row["annotation"]),
            instances_objects=instances_objects,
            depth=depth,
            prompt=prompt,
            args=args,
            source_image_path=image_path,
        )

    if args.point_source == "molmo":
        from SmartGrasp.perception.occul_map.molmo_sam_org import build_org_json

        points_path = maybe_run_molmo(
            image_path,
            prompt,
            out_dir,
            args.molmo_model_id,
            args.molmo_max_attempts,
        )
        sanitize_points_json(points_path)
        graph_payload = build_org_json(
            points_json_path=points_path.resolve(),
            depth_path=depth_path.resolve(),
            output_json_path=(out_dir / "occlusion_graph.json").resolve(),
            output_mask_dir=(out_dir / "mask").resolve(),
            segmentation_backend=args.segmentation_backend,
            sam_model_id=args.sam_model_id,
            epsilon=args.epsilon,
            kernel_size=args.kernel_size,
            min_contact_pixels=args.min_contact_pixels,
            min_contact_ratio=args.min_contact_ratio,
            sam_point_grid_radius=args.sam_point_grid_radius,
            sam_prompt_mode=args.sam_prompt_mode,
            sam_negative_points=args.sam_negative_points,
            mask_clean_kernel=args.mask_clean_kernel,
            proposal_backend=args.proposal_backend,
            proposal_min_area_ratio=args.proposal_min_area_ratio,
            proposal_max_area_ratio=args.proposal_max_area_ratio,
            proposal_iou_threshold=args.proposal_iou_threshold,
            proposal_containment_threshold=args.proposal_containment_threshold,
            proposal_border_fraction_threshold=args.proposal_border_fraction_threshold,
            max_proposal_masks=args.max_proposal_masks,
            save_candidates=args.save_candidates,
            device=args.device,
        )
        visualize_graph_payload(graph_payload["graph"], out_dir / "occlusion_graph.png", "Molmo/SAM Occlusion Graph")
    else:
        points = build_gt_points(instances_objects, str(row["annotation"]), query_obj_id)
        points_path = write_points_json(out_dir, image_path, width, height, prompt, points, "gt_centers")
        graph_payload = build_graph_from_gt_masks(
            instances_objects=instances_objects,
            depth=depth,
            out_dir=out_dir,
            epsilon=args.epsilon,
            kernel_size=args.kernel_size,
            min_contact_pixels=args.min_contact_pixels,
            min_contact_ratio=args.min_contact_ratio,
        )

    scene_graph_summary = build_summary_scene_graph(points_path, graph_payload)
    summary = {
        "scene_id": scene_id,
        "query_obj_id": query_obj_id,
        "annotation": str(row["annotation"]),
        "point_source": args.point_source,
        "output_dir": str(out_dir.resolve()),
        "image_path": str(image_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "points_json": str(points_path.resolve()),
        "graph_json": str((out_dir / "occlusion_graph.json").resolve()),
        "graph_png": str((out_dir / "occlusion_graph.png").resolve()),
        "raw_molmo_label_png": str((out_dir / "1_molmo_label_raw.png").resolve()) if args.point_source == "molmo" else None,
        "perception_label_png": str((out_dir / "3_molmo_label_proposed.png").resolve()),
        "num_nodes": len(graph_payload["graph"]["nodes"]),
        "num_edges": len(graph_payload["graph"]["edges"]),
        "gt_summary_json": str((scene_dir / "gt" / "summary.json").resolve()) if gt_summary else None,
        **scene_graph_summary,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SmartGrasp data -> Molmo points -> occlusion graph pipeline.")
    parser.add_argument("--scene-id", type=int, default=None, help="Scene id from the parquet/npz data.")
    parser.add_argument("--scene-ids", type=int, nargs="+", default=None, help="Run multiple scene ids in one process.")
    parser.add_argument("--serve", action="store_true", help="Keep models loaded and read scene ids from stdin.")
    parser.add_argument("--query-obj-id", type=int, default=None, help="Optional target object id.")
    parser.add_argument("--point-source", choices=["gt-centers", "molmo"], default="gt-centers")
    parser.add_argument("--prompt", default=None, help="Prompt used when running Molmo or saved in points JSON.")
    parser.add_argument("--molmo-model-id", default="allenai/Molmo-7B-D-0924")
    parser.add_argument("--molmo-max-attempts", type=int, default=1, help="Number of independent full-scene Molmo scans to merge.")
    parser.add_argument("--segmentation-backend", choices=["sam", "langsam", "auto"], default="sam")
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--min-contact-pixels", type=int, default=50)
    parser.add_argument("--min-contact-ratio", type=float, default=0.002)
    parser.add_argument("--sam-point-grid-radius", type=int, default=0)
    parser.add_argument("--sam-prompt-mode", choices=["cross", "grid", "ring", "auto"], default="cross")
    parser.add_argument("--sam-negative-points", type=int, default=0)
    parser.add_argument("--mask-clean-kernel", type=int, default=3)
    parser.add_argument("--proposal-backend", choices=["none", "sam2-auto"], default="sam2-auto")
    parser.add_argument("--proposal-min-area-ratio", type=float, default=0.0015)
    parser.add_argument("--proposal-max-area-ratio", type=float, default=0.11)
    parser.add_argument("--proposal-iou-threshold", type=float, default=0.35)
    parser.add_argument("--proposal-containment-threshold", type=float, default=0.6)
    parser.add_argument("--proposal-border-fraction-threshold", type=float, default=0.18)
    parser.add_argument("--max-proposal-masks", type=int, default=3)
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.serve:
        df = read_dataset()
        print("SmartGrasp pipeline worker is ready. Enter scene ids, one line at a time. Enter q to quit.", flush=True)
        while True:
            try:
                line = input("scene_id> ").strip()
            except EOFError:
                break
            if line.lower() in {"q", "quit", "exit"}:
                break
            if not line:
                continue
            try:
                scene_id = int(line)
            except ValueError:
                print(f"Invalid scene id: {line}", flush=True)
                continue
            item_args = argparse.Namespace(**vars(args))
            item_args.scene_id = scene_id
            item_args.query_obj_id = None
            item_args.scene_ids = None
            item_args.serve = False
            try:
                run_pipeline(item_args, df=df)
            except Exception as exc:
                print(f"Failed scene_id={scene_id}: {exc}", flush=True)
    elif args.scene_ids:
        df = read_dataset()
        summaries = []
        for scene_id in args.scene_ids:
            item_args = argparse.Namespace(**vars(args))
            item_args.scene_id = scene_id
            item_args.query_obj_id = None
            summaries.append(run_pipeline(item_args, df=df))
        print(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2))
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
