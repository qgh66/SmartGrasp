"""Export one live PyBullet camera frame for the Perception pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import fcntl
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def generate_capture_scene_id(
    input_root: str | os.PathLike[str] | None = None,
) -> int:
    """Reserve the next persistent capture ID: 1, 2, 3, ..."""
    if input_root is None:
        root = REPO_ROOT / "input"
    else:
        raw_root = Path(input_root).expanduser()
        root = (
            raw_root.resolve()
            if raw_root.is_absolute()
            else (REPO_ROOT / raw_root).resolve()
        )
    root.mkdir(parents=True, exist_ok=True)

    counter_path = root / ".capture_scene_id"
    with counter_path.open("a+", encoding="utf-8") as counter_file:
        fcntl.flock(counter_file.fileno(), fcntl.LOCK_EX)
        counter_file.seek(0)
        stored_value = counter_file.read().strip()
        next_id = int(stored_value) + 1 if stored_value else 1
        while (root / f"scene_{next_id}").exists():
            next_id += 1

        counter_file.seek(0)
        counter_file.truncate()
        counter_file.write(f"{next_id}\n")
        counter_file.flush()
        os.fsync(counter_file.fileno())
        fcntl.flock(counter_file.fileno(), fcntl.LOCK_UN)
    return next_id


def export_perception_input(
    *,
    scene_id: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    segmentation: np.ndarray,
    instruction: str,
    input_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Save one synchronized RGB-D-segmentation frame under input/scene_<id>."""
    scene_id = int(scene_id)
    if scene_id < 0:
        raise ValueError(f"scene_id must be non-negative, got {scene_id}")

    instruction = str(instruction).strip()
    if not instruction:
        raise ValueError("instruction must be a non-empty string")

    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth, dtype=np.float32)
    segmentation_array = np.asarray(segmentation, dtype=np.int32)
    if rgb_array.ndim != 3 or rgb_array.shape[2] not in (3, 4):
        raise ValueError(
            "rgb must have shape (H, W, 3) or (H, W, 4), "
            f"got {rgb_array.shape}"
        )
    if depth_array.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth_array.shape}")
    if segmentation_array.ndim != 2:
        raise ValueError(
            "segmentation must have shape (H, W), "
            f"got {segmentation_array.shape}"
        )
    spatial_shape = tuple(rgb_array.shape[:2])
    if depth_array.shape != spatial_shape or segmentation_array.shape != spatial_shape:
        raise ValueError(
            "rgb, depth and segmentation must be from the same camera frame: "
            f"rgb={spatial_shape}, depth={depth_array.shape}, "
            f"segmentation={segmentation_array.shape}"
        )

    if input_root is None:
        root = REPO_ROOT / "input"
    else:
        raw_root = Path(input_root).expanduser()
        root = (
            raw_root.resolve()
            if raw_root.is_absolute()
            else (REPO_ROOT / raw_root).resolve()
        )
    scene_dir = root / f"scene_{scene_id}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = scene_dir / "scene_image.png"
    depth_path = scene_dir / "depth.npy"
    segmentation_path = scene_dir / "segmentation.npy"
    instruction_path = scene_dir / "input.txt"
    summary_path = scene_dir / "summary.json"

    Image.fromarray(rgb_array[..., :3].astype(np.uint8), mode="RGB").save(rgb_path)
    np.save(depth_path, depth_array)
    np.save(segmentation_path, segmentation_array)
    instruction_path.write_text(instruction + "\n", encoding="utf-8")

    summary = {
        "scene_id": scene_id,
        "annotation": instruction,
        "instruction": instruction,
        "source": "grasp_dev_pybullet",
        "same_frame_rgb_depth_segmentation": True,
        "image_path": str(rgb_path),
        "depth_path": str(depth_path),
        "segmentation_path": str(segmentation_path),
        "image_shape": [int(value) for value in rgb_array.shape],
        "depth_shape": [int(value) for value in depth_array.shape],
        "segmentation_shape": [int(value) for value in segmentation_array.shape],
        "depth_dtype": str(depth_array.dtype),
        "depth_unit": "meter",
        "segmentation_dtype": str(segmentation_array.dtype),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "scene_id": scene_id,
        "input_dir": str(scene_dir),
        "scene_image": str(rgb_path),
        "depth": str(depth_path),
        "segmentation": str(segmentation_path),
        "instruction": str(instruction_path),
        "summary": str(summary_path),
    }


def run_perception_for_scene(
    scene_id: int,
    *,
    run_reason: bool = False,
) -> Path:
    """Pass one exported scene ID to the existing Perception shell entrypoint."""
    scene_id = int(scene_id)
    script_path = REPO_ROOT / "perception" / "run_perception.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Perception entrypoint not found: {script_path}")

    environment = os.environ.copy()
    environment["RUN_REASON_AFTER_PERCEPTION"] = "1" if run_reason else "0"
    subprocess.run(
        ["bash", str(script_path), str(scene_id)],
        cwd=str(REPO_ROOT),
        env=environment,
        check=True,
    )
    output_dir = REPO_ROOT / "data" / f"scene_{scene_id}" / "perception"
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"Perception completed without producing summary.json: {summary_path}"
        )
    return output_dir


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_graph_mask_path(
    *,
    graph_path: Path,
    object_id: int,
) -> Path:
    graph_data = _load_json(graph_path)
    nodes = (graph_data.get("graph") or {}).get("nodes") or []
    matching_node = next(
        (
            node
            for node in nodes
            if int(node.get("object_id", node.get("node_id", -1)))
            == int(object_id)
        ),
        None,
    )
    if matching_node is None:
        raise ValueError(
            f"Reason object_id={object_id} is absent from {graph_path}"
        )

    mask_value = (
        matching_node.get("mask_path")
        or matching_node.get("mask_file")
    )
    if not mask_value:
        raise ValueError(
            f"Perception node object_id={object_id} has no whole-object mask"
        )

    raw_mask = Path(os.path.expanduser(str(mask_value)))
    candidates = (
        [raw_mask]
        if raw_mask.is_absolute()
        else [
            graph_path.parent / raw_mask,
            graph_path.parent.parent / raw_mask,
            REPO_ROOT / raw_mask,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Whole-object mask for Reason object_id={object_id} was not found; "
        f"graph={graph_path}, mask_path={mask_value!r}"
    )


def _resolve_reason_part_mask_path(
    *,
    reason_summary_path: Path,
    part_mask: Any,
) -> Path | None:
    """Resolve Reason's optional part-mask path without requiring it to exist."""
    if not isinstance(part_mask, dict):
        return None
    path_value = part_mask.get("path")
    if not path_value:
        return None

    raw_path = Path(os.path.expanduser(str(path_value)))
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates = [
        reason_summary_path.parent / raw_path,
        REPO_ROOT / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def run_pipeline_for_scene(
    scene_id: int,
    *,
    allow_unselected_object: bool = False,
) -> dict[str, Any]:
    """Run the pipeline and resolve the selected object's mask when present.

    A fully hidden target can legitimately produce no ``grasp_object``. Task
    closed-loop callers with a configured occlusion relation may opt into that
    result and select the known physical occluder themselves. Other callers
    remain strict so an unexpected missing selection is still reported.
    """
    scene_id = int(scene_id)
    script_path = REPO_ROOT / "run_pipeline.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline entrypoint not found: {script_path}")

    environment = os.environ.copy()
    # VirtualCamera exports metric depth. Keep the dataset-wide 0.5 default
    # untouched, but use a centimeter-scale frontier gap for tabletop objects.
    environment["DEPTH_GAP_THRESHOLD"] = os.environ.get(
        "SIMULATION_DEPTH_GAP_THRESHOLD",
        "0.01",
    )
    subprocess.run(
        [
            "bash",
            str(script_path),
            str(scene_id),
            "--instruction=input",
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        check=True,
    )

    scene_output_dir = REPO_ROOT / "data" / f"scene_{scene_id}"
    perception_output_dir = scene_output_dir / "perception"
    reason_output_dir = scene_output_dir / "reason"
    perception_summary_path = perception_output_dir / "summary.json"
    graph_path = perception_output_dir / "occlusion_graph.json"
    reason_summary_path = reason_output_dir / "summary.json"
    for required_path in (
        perception_summary_path,
        graph_path,
        reason_summary_path,
    ):
        if not required_path.exists():
            raise RuntimeError(
                "SmartGrasp pipeline completed without producing required "
                f"output: {required_path}"
            )

    reason_summary = _load_json(reason_summary_path)
    grasp_object = reason_summary.get("grasp_object") or {}
    target_object = reason_summary.get("target_object") or {}
    target_object_id = target_object.get("id")
    target_object_id = (
        int(target_object_id) if target_object_id is not None else None
    )
    target_object_mask_path = (
        _resolve_graph_mask_path(
            graph_path=graph_path,
            object_id=target_object_id,
        )
        if target_object_id is not None
        else None
    )
    object_id = grasp_object.get("id")
    if object_id is None and not allow_unselected_object:
        raise RuntimeError(
            "Reason did not select grasp_object.id: "
            f"{reason_summary_path}"
        )
    object_id = int(object_id) if object_id is not None else None
    object_mask_path = (
        _resolve_graph_mask_path(
            graph_path=graph_path,
            object_id=object_id,
        )
        if object_id is not None
        else None
    )
    grasp_part_mask = reason_summary.get("grasp_part_mask")
    grasp_part_mask_path = _resolve_reason_part_mask_path(
        reason_summary_path=reason_summary_path,
        part_mask=grasp_part_mask,
    )

    return {
        "scene_id": scene_id,
        "branch": reason_summary.get("branch"),
        "status": reason_summary.get("status"),
        "instruction": reason_summary.get("instruction"),
        "target_object_id": target_object_id,
        "target_object_label": target_object.get("label"),
        "target_object_mask_path": (
            str(target_object_mask_path)
            if target_object_mask_path is not None
            else None
        ),
        "object_id": object_id,
        "object_label": grasp_object.get("label"),
        "object_mask_path": (
            str(object_mask_path) if object_mask_path is not None else None
        ),
        "perception_output_dir": str(perception_output_dir.resolve()),
        "perception_summary_path": str(perception_summary_path.resolve()),
        "occlusion_graph_path": str(graph_path.resolve()),
        "reason_output_dir": str(reason_output_dir.resolve()),
        "reason_summary_path": str(reason_summary_path.resolve()),
        "grasp_part_mask": grasp_part_mask,
        "grasp_part_mask_path": (
            str(grasp_part_mask_path)
            if grasp_part_mask_path is not None
            else None
        ),
        "graspability": reason_summary.get("graspability"),
    }
