from __future__ import annotations

import argparse
import io
import json
import os
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
from PIL import Image, ImageDraw, ImageFont

from SmartGrasp.perception._shared import _save_mask_png
from SmartGrasp.perception.background import generate_background_exclusion_mask
from SmartGrasp.perception.data_loader import DATA_DIR, PARQUET_GLOB, iter_npz_sources, load_npz
from SmartGrasp.perception.occlusion_map import build_occlusion_graph, graph_to_jsonable


OUT_ROOT = SMARTGRASP_ROOT / "data_realworld"
INPUT_ROOT = SMARTGRASP_ROOT / "input"
DATA_REALWORLD_ROOT = SMARTGRASP_ROOT / "data_realworld"


def _convert_depth_raw_to_npy(depth_raw_path: Path, out_dir: Path) -> Path:
    """Convert RealSense Z16 depth to the centimetres used by Perception."""
    depth_npy_path = out_dir / "depth.npy"
    meta_path = depth_raw_path.parent / "camera_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Cannot convert {depth_raw_path}: missing camera metadata {meta_path}"
        )

    metadata = load_json_file(meta_path)
    try:
        width = int(metadata["width"])
        height = int(metadata["height"])
        depth_scale_m = float(metadata["depth_scale_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid camera metadata in {meta_path}: width, height and "
            "depth_scale_m are required"
        ) from exc
    if width < 1 or height < 1:
        raise ValueError(f"Invalid camera resolution in {meta_path}: {width}x{height}")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError(
            f"Invalid depth_scale_m in {meta_path}: {depth_scale_m!r}"
        )

    raw_bytes = depth_raw_path.read_bytes()
    expected_bytes = width * height * np.dtype(np.uint16).itemsize
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Unexpected depth.raw size: {len(raw_bytes)} bytes, "
            f"expected {expected_bytes} for {width}x{height} uint16"
        )

    depth_raw = np.frombuffer(raw_bytes, dtype="<u2").reshape(height, width)
    # Perception's absolute depth thresholds are expressed in centimetres
    # (for example, background=79.752 and occlusion gap=0.5).
    depth_cm = depth_raw.astype(np.float32) * np.float32(depth_scale_m * 100.0)
    np.save(depth_npy_path, depth_cm)
    return depth_npy_path

def read_dataset() -> pd.DataFrame:
    parquet_files = sorted(Path(DATA_DIR).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found by {PARQUET_GLOB}")
    return pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)


def optional_int(value: Any, fallback: int | None = None) -> int | None:
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def scene_input_instruction(scene_dir: Path) -> tuple[str | None, Path | None]:
    for name in ("input.txt", "instruction.txt"):
        path = scene_dir / name
        if path.exists():
            text = read_text_file(path)
            if text:
                return text, path
    return None, None


def summary_instruction(summary: dict[str, Any]) -> str:
    return str(summary.get("instruction") or summary.get("annotation") or "")


def _data_realworld_scene_key(scene_dir: Path) -> str:
    """Return a stable relative key such as ``timestamp/2`` for a scene."""
    try:
        return scene_dir.relative_to(DATA_REALWORLD_ROOT).as_posix()
    except ValueError:
        return scene_dir.name


def load_priority_scene_inputs(scene_id: int | str | None) -> dict[str, Any] | None:
    if scene_id is None:
        return None

    # Try legacy FreeGrasp-style integer scene_id first
    try:
        int_scene_id = int(scene_id)
        scene_name = f"scene_{int_scene_id}"
        candidates = [
            (INPUT_ROOT / scene_name, INPUT_ROOT / scene_name),
            (OUT_ROOT / scene_name / "perception", OUT_ROOT / scene_name / "perception"),
        ]
        for input_dir, summary_dir in candidates:
            summary_path = summary_dir / "summary.json"
            depth_path = input_dir / "depth.npy"
            scene_image_path = input_dir / "scene_image.png"
            if not scene_image_path.exists():
                scene_image_path = input_dir / "rgb.png"
            if not (scene_image_path.exists() and depth_path.exists()):
                continue
            summary = load_json_file(summary_path) if summary_path.exists() else {}
            instruction, instruction_path = scene_input_instruction(input_dir)
            annotation = instruction if instruction is not None else summary_instruction(summary)
            summary = {
                **summary,
                "scene_id": optional_int(summary.get("scene_id"), int_scene_id),
                "annotation": annotation,
                "instruction": annotation,
            }
            return {
                "summary": summary,
                "input_dir": input_dir,
                "summary_path": summary_path if summary_path.exists() else None,
                "instruction_path": instruction_path,
                "image_path": scene_image_path,
                "depth_path": depth_path,
            }
    except (ValueError, TypeError):
        pass  # not an integer scene_id, try data_realworld timestamp

    # --- data_realworld direct scene dirs (e.g. data_realworld/20260724_143052/) ---
    if scene_id is not None:
        # Filter by specific scene_id (timestamp string)
        target_dir = DATA_REALWORLD_ROOT / str(scene_id)
        entries = [target_dir] if target_dir.is_dir() else []
    else:
        entries = sorted(DATA_REALWORLD_ROOT.iterdir(), reverse=True)

    for scene_dir in entries:
        if not scene_dir.is_dir():
            continue
        input_dir = scene_dir / "input"
        if not input_dir.is_dir():
            input_dir = scene_dir  # legacy layout fallback
        rgb_path = input_dir / "rgb.png"
        depth_path = input_dir / "depth.npy"
        if not rgb_path.exists():
            continue
        depth_raw = input_dir / "depth.raw"
        if depth_raw.exists():
            # depth.raw is authoritative for camera scenes. Always regenerate
            # depth.npy so files produced by the old unscaled conversion are
            # not silently reused.
            depth_path = _convert_depth_raw_to_npy(depth_raw, input_dir)
        elif not depth_path.exists():
            continue
        instruction, instruction_path = scene_input_instruction(input_dir)
        if instruction is None:
            instruction, instruction_path = scene_input_instruction(scene_dir)  # legacy fallback
        scene_key = _data_realworld_scene_key(scene_dir)
        summary = {
            "scene_id": scene_key,
            "annotation": instruction or "",
            "instruction": instruction or "",
            "depth_unit": "centimeter",
        }
        print(f"[perception] using data_realworld scene: {scene_key}", flush=True)
        return {
            "summary": summary,
            "input_dir": input_dir,
            "summary_path": input_dir / "summary.json" if (input_dir / "summary.json").exists() else None,
            "instruction_path": instruction_path,
            "image_path": rgb_path,
            "depth_path": depth_path,
            "camera_meta_path": (
                input_dir / "camera_meta.json"
                if (input_dir / "camera_meta.json").is_file()
                else None
            ),
        }

    # --- legacy direct scene dirs (rgb.png/depth.npy at scene root, no input/) ---
    for entry in sorted(DATA_REALWORLD_ROOT.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        scene_dir = entry
        rgb_path = scene_dir / "rgb.png"
        depth_path = scene_dir / "depth.npy"
        if not rgb_path.exists():
            continue
        depth_raw = scene_dir / "depth.raw"
        if depth_raw.exists():
            depth_path = _convert_depth_raw_to_npy(depth_raw, scene_dir)
        elif not depth_path.exists():
            continue
        instruction, instruction_path = scene_input_instruction(scene_dir)
        scene_key = _data_realworld_scene_key(scene_dir)
        summary = {
            "scene_id": scene_key,
            "annotation": instruction or "",
            "instruction": instruction or "",
            "depth_unit": "centimeter",
        }
        print(f"[perception] using data_realworld scene (legacy layout): {scene_key}", flush=True)
        return {
            "summary": summary,
            "input_dir": scene_dir,
            "summary_path": scene_dir / "summary.json" if (scene_dir / "summary.json").exists() else None,
            "instruction_path": instruction_path,
            "image_path": rgb_path,
            "depth_path": depth_path,
            "camera_meta_path": (
                scene_dir / "camera_meta.json"
                if (scene_dir / "camera_meta.json").is_file()
                else None
            ),
        }


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
        points.append({"object_id": object_id, "x": x, "y": y, "label": label})
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
    path = out_dir / "points.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_depth(depth: np.ndarray, out_dir: Path) -> Path:
    path = out_dir / "depth.npy"
    np.save(path, depth.astype(np.float32, copy=False))
    return path


def save_depth_image(depth: np.ndarray, out_dir: Path) -> Path:
    path = out_dir / "scene_depth.png"
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        Image.new("L", depth.shape[::-1], 0).save(path)
        return path

    near, far = np.percentile(depth[valid], [2, 98])
    if far <= near:
        gray = np.zeros(depth.shape, dtype=np.uint8)
    else:
        normalized = (far - np.clip(depth, near, far)) / max(far - near, 1e-6)
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = np.clip(normalized[valid] * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(path)
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
                "object_id": object_id,
                "label": f"object_{object_id}",
                "point": {"x": int(object_centroid(mask)[0]), "y": int(object_centroid(mask)[1])},
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(mask)),
            }
        )
    return records


def save_background_exclusion_mask(background: np.ndarray | None, mask_dir: Path) -> str | None:
    if background is None or int(np.count_nonzero(background)) == 0:
        return None
    background_path = mask_dir / "000_background_mask.png"
    _save_mask_png(np.asarray(background, dtype=bool), background_path)
    return str(background_path.resolve())


def visualize_graph(graph: nx.DiGraph, graph_json: dict[str, Any], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if graph.number_of_nodes() > 0:
        pos = directed_graph_layout(graph)
        labels = {}
        for node in graph.nodes():
            node_payload = graph_json["nodes"][int(node)]
            labels[node] = str(node_payload.get("object_id", node))
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
    kernel_size: int,
    min_contact_pixels: int,
    min_contact_ratio: float,
    background_mask: np.ndarray | None = None,
    max_contact_background_ratio: float = 0.4,
) -> dict[str, Any]:
    node_records = save_gt_masks(instances_objects, out_dir)
    object_ids = [record["object_id"] for record in node_records]
    masks = np.stack([(instances_objects == object_id) for object_id in object_ids], axis=0)
    graph, adjacency = build_occlusion_graph(
        masks=masks,
        depth_map=depth,
        kernel_size=kernel_size,
        min_contact_pixels=min_contact_pixels,
        min_contact_ratio=min_contact_ratio,
        background_mask=background_mask,
        max_contact_background_ratio=max_contact_background_ratio,
    )
    graph_payload = graph_to_jsonable(graph, adjacency, node_records=node_records)
    for edge in graph_payload["edges"]:
        source_node = node_records[edge["source"]]
        target_node = node_records[edge["target"]]
        edge["source_object_id"] = int(source_node["object_id"])
        edge["target_object_id"] = int(target_node["object_id"])
        edge["source_label"] = str(source_node["label"])
        edge["target_label"] = str(target_node["label"])

    payload = {"graph": graph_payload, "mask_source": "instances_objects", "depth_source": "npz.depth"}
    out_json = out_dir / "occlusion_graph.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    visualize_graph(graph, graph_payload, out_dir / "occlusion_graph.png", "GT Mask Occlusion Graph")
    return payload


def reset_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def display_label(label: Any) -> str:
    return str(label).replace("_", " ")


def save_final_objects_sheet(
    image_path: Path,
    graph_payload: dict[str, Any],
    output_path: Path,
    columns: int = 5,
) -> Path:
    """Save white-background crops of final assembled objects with object ids."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    items: list[tuple[int, str, Image.Image]] = []
    nodes = sorted(
        graph_payload.get("graph", {}).get("nodes", []),
        key=lambda node: int(node.get("object_id", node.get("node_id", 0))),
    )
    for node in nodes:
        mask_path_value = node.get("mask_path") or node.get("mask_file")
        if not mask_path_value:
            continue
        mask_path = Path(str(mask_path_value))
        if not mask_path.exists() or not mask_path.is_absolute():
            candidate = output_path.parent / mask_path
            if candidate.exists():
                mask_path = candidate
        if not mask_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != image_np.shape[:2] or not np.any(mask):
            continue
        ys, xs = np.nonzero(mask)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        pad = max(8, int(round(max(width, height) * 0.08)))
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(image_np.shape[1], int(xs.max()) + pad + 1)
        y1 = min(image_np.shape[0], int(ys.max()) + pad + 1)
        crop = image_np[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        visible = np.where(crop_mask[..., None], crop, 255).astype(np.uint8)
        object_id = int(node.get("object_id", node.get("node_id", 0)))
        label = str(node.get("label") or node.get("description") or f"object_{object_id}")
        items.append((object_id, label, Image.fromarray(visible, mode="RGB")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        Image.new("RGB", (256, 256), "white").save(output_path)
        return output_path

    try:
        font = ImageFont.truetype("Arial.ttf", 42)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 42)
        except Exception:
            font = ImageFont.load_default(size=42)
    label_height = 60
    rows = int(np.ceil(len(items) / columns))
    cell_width = max(crop.width for _object_id, _label, crop in items)
    cell_height = max(crop.height for _object_id, _label, crop in items)
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (object_id, _label, crop) in enumerate(items):
        row, col = divmod(index, columns)
        cell_x = col * cell_width
        cell_y = row * (cell_height + label_height)
        x = cell_x + (cell_width - crop.width) // 2
        y = cell_y + label_height + (cell_height - crop.height) // 2
        text = str(object_id)
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (cell_x + (cell_width - text_width) // 2, cell_y + 4),
            text,
            fill="black",
            font=font,
        )
        sheet.paste(crop, (x, y))
    sheet.save(output_path)
    return output_path


def build_summary_scene_graph(points_path: Path, graph_payload: dict[str, Any], scene_dir: Path) -> dict[str, Any]:
    """Build summary fields while preserving the original matrix interface."""
    graph = graph_payload["graph"]
    nodes = sorted(graph.get("nodes", []), key=lambda node: int(node["node_id"]))
    edges = graph.get("edges", [])
    out_dir = scene_dir / "perception"
    vlm_path = out_dir / "vlm.json"

    points_payload = load_json_file(points_path) if points_path.exists() else {}
    points_by_id: dict[int, dict[str, Any]] = {}
    for point in points_payload.get("points", []) or []:
        object_id = optional_int(point.get("object_id"), optional_int(point.get("molmo_id")))
        if object_id is not None:
            points_by_id[int(object_id)] = point

    object_points: list[dict[str, Any]] = []
    node_object_ids: list[int] = []
    for index, node in enumerate(nodes):
        object_id = int(node.get("object_id", index + 1))
        node_object_ids.append(object_id)
        point = points_by_id.get(object_id, {})
        node_point = node.get("point", {}) or {}
        label = str(point.get("label") or node.get("label") or node.get("description") or f"object_{object_id}")
        object_points.append({
            "object_id": object_id,
            "x": int(point.get("x", node_point.get("x", 0))),
            "y": int(point.get("y", node_point.get("y", 0))),
            "label": label,
        })

    matrix_labels = [
        f"{point['object_id']}: {display_label(point.get('label', ''))}"
        for point in object_points
    ]
    object_id_to_index = {object_id: index for index, object_id in enumerate(node_object_ids)}
    occlusion_matrix = [[0.0 for _ in node_object_ids] for _ in node_object_ids]
    for edge in edges:
        source_object_id = int(edge.get("source_object_id", node_object_ids[int(edge["source"])]))
        target_object_id = int(edge.get("target_object_id", node_object_ids[int(edge["target"])]))
        source_index = object_id_to_index.get(source_object_id)
        target_index = object_id_to_index.get(target_object_id)
        if source_index is None or target_index is None:
            continue
        occlusion_matrix[source_index][target_index] = safe_float(edge.get("contact_ratio"), 6) or 0.0

    # Build sam2_id → mask_path mapping
    sam2_to_mask: dict[int, str] = {}
    for node in nodes:
        for sid in node.get("sam2_ids", []):
            sam2_to_mask[int(sid)] = node.get("mask_path", "")

    # Build object info from VLM
    objects = []
    if vlm_path.exists():
        with open(vlm_path) as f:
            vlm = json.load(f)
        for obj in vlm.get("objects", []):
            oid = obj["id"]
            parts = []
            for part in obj.get("visible_parts", []):
                mask_paths = []
                for sid in part.get("sam2_ids", []):
                    mp = sam2_to_mask.get(int(sid))
                    if mp:
                        try:
                            mask_paths.append(str(Path(mp).relative_to(out_dir)))
                        except ValueError:
                            mask_paths.append(mp)
                parts.append({
                    "description": part.get("description", ""),
                    "sam2_ids": part.get("sam2_ids", []),
                    "mask_paths": mask_paths,
                })
            centroid = {"x": 0, "y": 0}
            for node in nodes:
                if node.get("object_id") == oid:
                    centroid = {"x": node["point"]["x"], "y": node["point"]["y"]}
                    break
            mask_rel = ""
            for sid in obj.get("sam2_ids", []):
                mp = sam2_to_mask.get(int(sid))
                if mp:
                    try:
                        mask_rel = str(Path(mp).relative_to(out_dir))
                    except ValueError:
                        mask_rel = mp
                    break
            objects.append({
                "object_id": oid,
                "label": obj.get("description", ""),
                "relative_position": obj.get("relative_position", ""),
                "centroid": centroid,
                "mask_path": mask_rel,
                "sam2_ids": obj.get("sam2_ids", []),
                "parts": parts,
            })

    object_id_to_sam2_part_ids: dict[str, list[int]] = {}
    object_id_to_sam2_part_files: dict[str, list[str]] = {}
    sam2_part_id_to_object_id: dict[str, int] = {}
    sam2_part_file_to_object_id: dict[str, int] = {}
    for obj in objects:
        object_id = int(obj["object_id"])
        part_ids: set[int] = set()
        for sid in obj.get("sam2_ids", []) or []:
            part_ids.add(int(sid))
        for part in obj.get("parts", []) or []:
            for sid in part.get("sam2_ids", []) or []:
                part_ids.add(int(sid))

        sorted_part_ids = sorted(part_ids)
        object_id_to_sam2_part_ids[str(object_id)] = sorted_part_ids
        part_files: list[str] = []
        for sid in sorted_part_ids:
            part_file = f"sam2_rgb_parts/part_{sid:03d}.png"
            if (out_dir / part_file).exists():
                part_files.append(part_file)
                sam2_part_file_to_object_id[part_file] = object_id
            sam2_part_id_to_object_id[str(sid)] = object_id
        object_id_to_sam2_part_files[str(object_id)] = part_files

    return {
        "object_points": object_points,
        "matrix_labels": matrix_labels,
        "occlusion_matrix_direction": "row object occludes column object",
        "occlusion_matrix_metric": "contact_ratio",
        "occlusion_matrix": occlusion_matrix,
        "object_id_to_sam2_part_ids": object_id_to_sam2_part_ids,
        "object_id_to_sam2_part_files": object_id_to_sam2_part_files,
        "sam2_part_id_to_object_id": sam2_part_id_to_object_id,
        "sam2_part_file_to_object_id": sam2_part_file_to_object_id,
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
    scene_dir = OUT_ROOT / f"scene_{scene_id}"
    gt_dir = scene_dir / "gt"
    reset_output_dir(gt_dir)
    image_path = copy_or_save_sample_image(row, source_image_path, gt_dir)
    with Image.open(image_path) as image:
        width, height = image.size
    depth_path = save_depth(depth, gt_dir)
    depth_image_path = save_depth_image(depth, gt_dir)
    points = build_gt_points(instances_objects, annotation, query_obj_id)
    points_path = write_points_json(gt_dir, image_path, width, height, prompt, points, "gt_centers")

    # GT background mask: everything that is not a foreground object
    from SmartGrasp.perception.background import generate_gt_background_exclusion_mask
    gt_background_mask = generate_gt_background_exclusion_mask(instances_objects)

    graph_payload = build_graph_from_gt_masks(
        instances_objects=instances_objects,
        depth=depth,
        out_dir=gt_dir,
        kernel_size=args.kernel_size,
        min_contact_pixels=args.min_contact_pixels,
        min_contact_ratio=args.min_contact_ratio,
        background_mask=gt_background_mask,
        max_contact_background_ratio=args.max_contact_background_ratio,
    )
    scene_graph_summary = build_summary_scene_graph(points_path, graph_payload, scene_dir)
    summary = {
        "scene_id": scene_id,
        "query_obj_id": query_obj_id,
        "annotation": annotation,
        "point_source": "gt-centers",
        "output_dir": str(gt_dir.resolve()),
        "image_path": str(image_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "depth_image_path": str(depth_image_path.resolve()),
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
    priority_inputs = load_priority_scene_inputs(args.scene_id)
    row = None
    source_image: Image.Image | None = None
    instances_objects: np.ndarray | None = None
    is_data_realworld = False  # True if using data_realworld/<timestamp>/ dir
    camera_intrinsics: dict[str, Any] | None = None

    if priority_inputs is not None:
        priority_summary = priority_inputs["summary"]
        input_dir = Path(priority_inputs["input_dir"])
        # Detect data_realworld scene dir (e.g. data_realworld/20260724_143052/)
        # input_dir may be the scene root or input/ subdirectory
        if DATA_REALWORLD_ROOT in input_dir.parents or input_dir.parent == DATA_REALWORLD_ROOT:
            is_data_realworld = True
            if input_dir.name == "input":
                # input/ subdirectory layout: scene_dir is the parent
                scene_dir = input_dir.parent
                scene_id = _data_realworld_scene_key(scene_dir)
            else:
                # legacy layout: input_dir IS the scene dir
                scene_dir = input_dir
                scene_id = _data_realworld_scene_key(scene_dir)
        else:
            scene_id = optional_int(priority_summary.get("scene_id"), args.scene_id)
            if scene_id is None:
                raise ValueError("Priority perception inputs require a scene id.")
            scene_dir = OUT_ROOT / f"scene_{scene_id}"

        query_obj_id = optional_int(priority_summary.get("query_obj_id"), args.query_obj_id)
        if query_obj_id is None:
            query_obj_id = -1
        annotation = str(priority_summary.get("instruction") or priority_summary.get("annotation") or "")
        source_image = Image.open(priority_inputs["image_path"]).convert("RGB")
        depth = np.asarray(np.load(priority_inputs["depth_path"]), dtype=np.float32)
        camera_meta_path = priority_inputs.get("camera_meta_path")
        if camera_meta_path is not None:
            camera_metadata = load_json_file(Path(camera_meta_path))
            raw_intrinsics = camera_metadata.get("intrinsics")
            if isinstance(raw_intrinsics, dict):
                camera_intrinsics = raw_intrinsics
        instruction_path = priority_inputs.get("instruction_path")
        input_note = f", instruction={instruction_path}" if instruction_path else ""
        print(
            f"[{scene_id}] using priority perception inputs: "
            f"{input_dir}{input_note}",
            flush=True,
        )
    else:
        if df is None:
            df = read_dataset()
        row = select_sample(df, args.scene_id, args.query_obj_id)
        scene_id = int(row["sceneId"])
        query_obj_id = int(row["queryObjId"])
        annotation = str(row["annotation"])
        scene_dir = OUT_ROOT / f"scene_{scene_id}"
    from SmartGrasp.perception._shared import set_log_scene_id
    try:
        set_log_scene_id(int(scene_id))
    except (TypeError, ValueError):
        pass

    out_dir = scene_dir / "perception"
    # Clean output directory by removing contents individually
    # (avoids iCloud Drive conflict copies caused by rmtree+recreate race)
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_image is not None:
        image_path = out_dir / "scene_image.png"
        source_image.save(image_path)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    **priority_inputs["summary"],
                    "scene_id": scene_id,
                    "query_obj_id": query_obj_id,
                    "annotation": annotation,
                    "instruction": annotation,
                    "input_dir": str(Path(priority_inputs["input_dir"]).resolve()),
                    "input_instruction_path": (
                        str(Path(priority_inputs["instruction_path"]).resolve())
                        if priority_inputs.get("instruction_path")
                        else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        image_path = save_sample_image(row, out_dir)
    with Image.open(image_path) as image:
        width, height = image.size

    if priority_inputs is None:
        npz_source, zip_member = find_npz_source(scene_id)
        with load_npz(npz_source, zip_member) as npz:
            depth = np.asarray(npz["depth"], dtype=np.float32)
            instances_objects = np.asarray(npz["instances_objects"])

    depth_path = save_depth(depth, out_dir)
    depth_image_path = save_depth_image(depth, out_dir)
    prompt = args.prompt or ""

    gt_summary = None
    if instances_objects is not None:
        gt_summary = build_gt_reference_outputs(
            row=row,
            scene_id=scene_id,
            query_obj_id=query_obj_id,
            annotation=annotation,
            instances_objects=instances_objects,
            depth=depth,
            prompt=prompt,
            args=args,
            source_image_path=image_path,
        )

    if args.mode == "vlm":
        from SmartGrasp.perception.occlusion_map import build_org_json

        # ---- Debug: sam2 only ----
        if args.debug == "sam2":
            from SmartGrasp.perception.sam2auto import (
                _sam2_auto_candidate_pool,
                _draw_sam2_auto_label_image,
                _save_sam2_rgb_parts_sheet,
            )
            bg_mask = None
            background_mask_path = None
            try:
                bg_mask = generate_background_exclusion_mask(
                    depth_map=depth,
                    image=Image.open(image_path).convert("RGB"),
                    mask_clean_kernel=args.mask_clean_kernel,
                    camera_intrinsics=camera_intrinsics,
                )
                background_mask_path = save_background_exclusion_mask(bg_mask, out_dir / "mask")
            except Exception as exc:
                print(f"bg_mask failed: {exc}", file=sys.stderr)
            candidates, report, _, _ = _sam2_auto_candidate_pool(
                image_path=image_path, output_mask_dir=out_dir / "mask",
                min_area_ratio=args.proposal_min_area_ratio,
                max_area_ratio=args.proposal_max_area_ratio,
                mask_clean_kernel=args.mask_clean_kernel,
                save_candidates=True, device=args.device,
                background_exclusion_mask=bg_mask,
                points_per_side=args.sam2_points_per_side,
                crop_n_layers=args.sam2_crop_n_layers,
                pred_iou_thresh=args.sam2_pred_iou_thresh,
                stability_score_thresh=args.sam2_stability_score_thresh,
                depth_points_per_side=args.depth_sam2_points_per_side,
                depth_crop_n_layers=args.depth_sam2_crop_n_layers,
                depth_pred_iou_thresh=args.depth_sam2_pred_iou_thresh,
                depth_stability_score_thresh=args.depth_sam2_stability_score_thresh,
                border_fraction_threshold=args.proposal_border_fraction_threshold,
                depth_map=depth,
            )
            # Generate visualization images
            label_path = out_dir / "label_1_sam2auto.png"
            _draw_sam2_auto_label_image(image_path, candidates, label_path)
            _save_sam2_rgb_parts_sheet(image_path, candidates, out_dir)
            debug_out = {
                "debug": "sam2",
                "scene_id": scene_id,
                "num_candidates": len(candidates),
                "label_png": str(label_path.resolve()),
                "parts_sheet_png": str((out_dir / "sam2_rgb_parts_sheet.png").resolve()),
                "background_mask_path": background_mask_path,
                "background_mask_source": "depth",
                "candidates": [{k: v for k, v in c.items() if k != "mask"} for c in candidates],
                "report": report,
            }
            debug_path = out_dir / "debug_sam2.json"
            debug_path.write_text(json.dumps(debug_out, ensure_ascii=False, indent=2), encoding="utf-8")
            return debug_out

        graph_payload = build_org_json(
            image_path=image_path,
            depth_path=depth_path.resolve(),
            output_json_path=(out_dir / "occlusion_graph.json").resolve(),
            output_mask_dir=(out_dir / "mask").resolve(),
            review_model_id=args.review_model_id,
            review_api_key_env=args.review_api_key_env,
            review_base_url=args.review_base_url,
            review_timeout=args.review_timeout,
            kernel_size=args.kernel_size,
            min_contact_pixels=args.min_contact_pixels,
            min_contact_ratio=args.min_contact_ratio,
            mask_clean_kernel=args.mask_clean_kernel,
            proposal_min_area_ratio=args.proposal_min_area_ratio,
            proposal_max_area_ratio=args.proposal_max_area_ratio,
            save_candidates=args.save_candidates,
            device=args.device,
            camera_intrinsics=camera_intrinsics,
            sam2_points_per_side=args.sam2_points_per_side,
            sam2_crop_n_layers=args.sam2_crop_n_layers,
            sam2_pred_iou_thresh=args.sam2_pred_iou_thresh,
            sam2_stability_score_thresh=args.sam2_stability_score_thresh,
            depth_sam2_points_per_side=args.depth_sam2_points_per_side,
            depth_sam2_crop_n_layers=args.depth_sam2_crop_n_layers,
            depth_sam2_pred_iou_thresh=args.depth_sam2_pred_iou_thresh,
            depth_sam2_stability_score_thresh=args.depth_sam2_stability_score_thresh,
            proposal_border_fraction_threshold=args.proposal_border_fraction_threshold,
            max_contact_background_ratio=args.max_contact_background_ratio,
        )
        # Graph PNG already saved by build_org_json with scene-image background
        # Write a minimal points.json for summary generation
        points_payload = {
            "points": [
                {
                    "object_id": int(node.get("object_id", node.get("node_id", 0))),
                    "x": int(node.get("point", {}).get("x", 0)),
                    "y": int(node.get("point", {}).get("y", 0)),
                    "label": str(node.get("label", "")),
                }
                for node in graph_payload["graph"].get("nodes", [])
            ]
        }
        points_path = out_dir / "points.json"
        points_path.write_text(json.dumps(points_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        scene_graph_summary = build_summary_scene_graph(points_path, graph_payload, scene_dir)
        final_objects_sheet_path = save_final_objects_sheet(
            image_path,
            graph_payload,
            out_dir / "final_objects_sheet.png",
        )
        summary = {
            "scene_id": scene_id,
            "query_obj_id": query_obj_id,
            "annotation": annotation,
            "depth_unit": (
                priority_summary.get("depth_unit")
                if priority_inputs is not None
                else "centimeter"
            ),
            "point_source": "sam2-vlm-anchor",
            "output_dir": str(out_dir.resolve()),
            "image_path": str(image_path.resolve()),
            "depth_path": str(depth_path.resolve()),
            "depth_image_path": str(depth_image_path.resolve()),
            "graph_json": str((out_dir / "occlusion_graph.json").resolve()),
            "graph_png": str((out_dir / "occlusion_graph.png").resolve()),
            "background_mask_path": graph_payload.get("background_mask_path"),
            "background_mask_source": graph_payload.get("background_mask_source"),
            "sam2_auto_label_png": str((out_dir / "label_1_sam2auto.png").resolve()),
            "sam2_rgb_parts_sheet_png": str((out_dir / "sam2_rgb_parts_sheet.png").resolve()),
            "final_objects_sheet_png": str(final_objects_sheet_path.resolve()),
            "vlm_review_json": str((out_dir / "vlm.json").resolve()),
            "perception_label_png": str((out_dir / "label_2_vlm.png").resolve()),
            "num_nodes": len(graph_payload["graph"]["nodes"]),
            "num_edges": len(graph_payload["graph"]["edges"]),
            "gt_summary_json": str((scene_dir / "gt" / "summary.json").resolve()),
            **scene_graph_summary,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    if gt_summary is None:
        raise ValueError("mode='gt' requires parquet/npz inputs with instances_objects.")
    return gt_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SmartGrasp perception pipeline: SAM2 auto -> VLM review -> occlusion graph.")
    parser.add_argument("--scene-id", default=None, help="Scene id (int for FreeGrasp, timestamp string for data_realworld). Auto-detects latest data_realworld scene if omitted.")
    parser.add_argument("--scene-ids", nargs="+", default=None, help="Run multiple scene ids in one process.")
    parser.add_argument("--mode", choices=["gt", "vlm"], default="vlm",
                        help="gt: ground-truth occlusion graph only; vlm: full SAM2+VLM pipeline (default: vlm)")
    parser.add_argument("--serve", action="store_true", help="Keep models loaded and read scene ids from stdin.")
    parser.add_argument("--query-obj-id", type=int, default=None, help="Optional target object id.")
    parser.add_argument("--prompt", default=None, help="Prompt saved in output JSON.")
    parser.add_argument("--review-model-id", default="gpt-5.5")
    parser.add_argument("--review-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--review-base-url", default=None)
    parser.add_argument("--review-timeout", type=float, default=120.0)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--min-contact-pixels", type=int, default=50)
    parser.add_argument("--min-contact-ratio", type=float, default=0.002)
    parser.add_argument(
        "--max-contact-background-ratio",
        type=float,
        default=float(os.environ.get("MAX_CONTACT_BACKGROUND_RATIO", "0.4")),
        help="Maximum allowed background fraction in contact area; exceed → skip occlusion edge. "
             "Env: MAX_CONTACT_BACKGROUND_RATIO. Default: 0.4.",
    )
    parser.add_argument("--mask-clean-kernel", type=int, default=3)
    parser.add_argument("--proposal-min-area-ratio", type=float, default=0.006)
    parser.add_argument("--proposal-max-area-ratio", type=float, default=0.11)
    parser.add_argument("--proposal-border-fraction-threshold", type=float, default=0.18)
    parser.add_argument("--sam2-points-per-side", type=int, default=24)
    parser.add_argument("--sam2-crop-n-layers", type=int, default=0)
    parser.add_argument("--sam2-pred-iou-thresh", type=float, default=0.85)
    parser.add_argument("--sam2-stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--depth-sam2-points-per-side", type=int, default=None,
                        help="Depth SAM2 points_per_side; default: use --sam2-points-per-side")
    parser.add_argument("--depth-sam2-crop-n-layers", type=int, default=None,
                        help="Depth SAM2 crop_n_layers; default: use --sam2-crop-n-layers")
    parser.add_argument("--depth-sam2-pred-iou-thresh", type=float, default=None,
                        help="Depth SAM2 pred_iou_thresh; default: use --sam2-pred-iou-thresh")
    parser.add_argument("--depth-sam2-stability-score-thresh", type=float, default=None,
                        help="Depth SAM2 stability_score_thresh; default: use --sam2-stability-score-thresh")
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--debug", choices=["sam2"], default=None,
                        help="sam2: stop after SAM2 auto candidate generation")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.serve:
        df = None
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
                scene_id = line
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
                from SmartGrasp.perception.sam2auto import clear_sam2_image_state
                clear_sam2_image_state()
            except Exception as exc:
                print(f"Failed scene_id={scene_id}: {exc}", flush=True)
    elif args.scene_ids:
        df = None
        summaries = []
        for scene_id in args.scene_ids:
            item_args = argparse.Namespace(**vars(args))
            item_args.scene_id = scene_id
            item_args.query_obj_id = None
            summaries.append(run_pipeline(item_args, df=df))
            from SmartGrasp.perception.sam2auto import clear_sam2_image_state
            clear_sam2_image_state()
        print(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2))
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
