#!/usr/bin/env python
"""Scripted PyBullet grasp test without GraspNet inference.

This script is for testing the grasp execution layer only. It loads the same
industrial scene, uses PyBullet segmentation to get the target object's visible
point cloud, creates several deterministic top-down grasp poses, and evaluates
them with the JAKA Zu3 + Robotiq-85 gripper.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "graspnet_api"))

from graspnetAPI import GraspGroup
from simulation.camera import VirtualCamera
from simulation.evaluator import GraspEvaluator
from simulation.robot_gripper import JakaZu3Robotiq85Gripper
from simulation.scene import SimulationScene


def resolve_path(path: str | Path, *, config_dir: Path | None = None) -> Path:
    raw = Path(os.path.expanduser(str(path)))
    if raw.is_absolute():
        return raw
    candidates = []
    if config_dir is not None:
        candidates.append(config_dir / raw)
    candidates.extend([ROOT / raw, REPO_ROOT / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_scene_config(config_path: str | Path) -> dict:
    config_path = resolve_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    object_specs = config.get("objects", [])
    if not object_specs:
        raise ValueError(f"Scene config has no objects: {config_path}")
    resolved_specs = []
    for spec in object_specs:
        item = dict(spec)
        item["path"] = str(resolve_path(item["path"], config_dir=config_path.parent))
        resolved_specs.append(item)
    config["_path"] = str(config_path)
    config["_resolved_objects"] = resolved_specs
    return config


def select_target_object(scene: SimulationScene, target_name: str | None):
    if target_name:
        body_id = scene.get_body_id_by_name(target_name)
        return body_id, scene.get_object_info(body_id)
    for body_id, obj in scene.get_object_registry().items():
        if obj.metadata.get("role") == "target":
            return body_id, obj
    body_id = scene.object_ids[0]
    return body_id, scene.get_object_info(body_id)


def make_rotation(approach_axis, opening_axis):
    x_axis = np.asarray(approach_axis, dtype=float)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)
    y_axis = np.asarray(opening_axis, dtype=float)
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-8)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)


def estimate_target_grasp_center(target_points: np.ndarray) -> tuple[np.ndarray, dict]:
    xy = target_points[:, :2]
    median_xy = np.median(xy, axis=0)
    nearest_idx = int(np.linalg.norm(xy - median_xy[None, :], axis=1).argmin())
    center = target_points[nearest_idx].astype(float)
    z_low, z_high = np.percentile(target_points[:, 2], [10, 90])
    center[2] = max(0.5 * (float(z_low) + float(z_high)), 0.005)
    stats = {
        "bbox_min": target_points.min(axis=0).tolist(),
        "bbox_max": target_points.max(axis=0).tolist(),
        "median_xy": median_xy.tolist(),
        "chosen_surface_point": target_points[nearest_idx].tolist(),
        "waist_z": float(center[2]),
    }
    return center, stats


def horizontal_axes_from_points(target_points: np.ndarray) -> list[np.ndarray]:
    xy = target_points[:, :2]
    centered = xy - xy.mean(axis=0, keepdims=True)
    axes = []
    if len(centered) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axes.extend([vh[0], vh[1]])
    axes.extend([
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ])

    unique_axes = []
    for axis in axes:
        axis = np.asarray(axis, dtype=float)
        axis = axis / max(np.linalg.norm(axis), 1e-8)
        if any(abs(float(np.dot(axis, old))) > 0.95 for old in unique_axes):
            continue
        unique_axes.append(axis)
    return [np.array([axis[0], axis[1], 0.0], dtype=float) for axis in unique_axes]


def width_for_axis(target_points: np.ndarray, opening_axis: np.ndarray) -> float:
    axis = np.asarray(opening_axis[:2], dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    projection = target_points[:, :2] @ axis
    span = float(np.percentile(projection, 90) - np.percentile(projection, 10))
    return float(np.clip(span + 0.015, 0.025, 0.085))


def build_scripted_grasps(target_points: np.ndarray, target_body_id: int, max_grasps: int) -> tuple[GraspGroup, dict]:
    center, center_stats = estimate_target_grasp_center(target_points)
    approach = np.array([0.0, 0.0, -1.0], dtype=float)
    grasp_rows = []
    axes = horizontal_axes_from_points(target_points)
    for index, opening_axis in enumerate(axes[:max_grasps]):
        rotation = make_rotation(approach, opening_axis)
        width = width_for_axis(target_points, opening_axis)
        score = 1.0 - 0.05 * index
        height = 0.02
        depth = 0.035
        row = np.concatenate([
            np.array([score, width, height, depth], dtype=np.float64),
            rotation.reshape(-1),
            center.astype(np.float64),
            np.array([float(target_body_id)], dtype=np.float64),
        ])
        grasp_rows.append(row)
    gg = GraspGroup(np.vstack(grasp_rows))
    return gg, {"center": center.tolist(), "center_stats": center_stats, "num_scripted_grasps": len(gg)}


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Scripted PyBullet grasp execution test")
    parser.add_argument("--scene-config", default="config/industrial_scene.json")
    parser.add_argument("--target-object", default="medium_clamp")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", default="results/scripted_grasp_test.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_scene_config(args.scene_config)

    scene = SimulationScene(gui=args.gui)
    scene.connect()
    scene.load_plane()
    scene.load_objects(config["_resolved_objects"])

    target_body_id, target_object = select_target_object(scene, args.target_object)
    for _ in range(int(config.get("settle_steps", 300))):
        scene.step()

    camera_cfg = config.get("camera", {})
    camera = VirtualCamera(
        position=tuple(camera_cfg.get("position", (0.3, 0.0, 0.5))),
        target=tuple(camera_cfg.get("target", (0.3, 0.0, 0.05))),
        width=int(camera_cfg.get("width", 1280)),
        height=int(camera_cfg.get("height", 720)),
        fov=float(camera_cfg.get("fov", 60.0)),
        near=float(camera_cfg.get("near", 0.01)),
        far=float(camera_cfg.get("far", 5.0)),
    )
    rgb, depth, seg = camera.capture()
    point_cloud = camera.generate_point_cloud(depth, num_points=20000).numpy()
    object_clouds = camera.generate_object_point_clouds(depth, seg, scene.object_ids)
    target_points = object_clouds.get(int(target_body_id))
    if target_points is None or len(target_points) == 0:
        raise RuntimeError(f"Target object has no visible points: {target_object.name}")

    grasp_group, scripted_info = build_scripted_grasps(target_points, target_body_id, args.top_k)
    print(f"[Scripted] target={target_object.name} body_id={target_body_id}")
    print(f"[Scripted] target_points={len(target_points)} center={np.round(scripted_info['center'], 4)}")
    print(f"[Scripted] generated_grasps={len(grasp_group)}")

    gripper = JakaZu3Robotiq85Gripper(planner=None)
    gripper.load()
    evaluator = GraspEvaluator(
        object_id=target_body_id,
        gripper=gripper,
        point_cloud=target_points,
        gui=args.gui,
    )
    results = evaluator.evaluate(grasp_group, top_k=args.top_k)
    success_count = sum(1 for item in results if item["success"])
    for index, result in enumerate(results):
        status = "OK" if result["success"] else "FAIL"
        print(
            f"[{status}] scripted {index}: "
            f"width={result['width']:.4f}, "
            f"lift_delta={result.get('obj_lift_delta', 0.0):.4f}, "
            f"reason={result.get('failure_reason', '-')}"
        )

    out = {
        "mode": "scripted_grasp",
        "total": len(results),
        "success": success_count,
        "scene_config": config["_path"],
        "target_object_name": target_object.name,
        "target_body_id": int(target_body_id),
        "obj_path": target_object.path,
        "scripted_grasp": scripted_info,
        "objects": scene.get_object_poses(),
        "object_point_counts": {int(k): int(len(v)) for k, v in object_clouds.items()},
        "gripper": gripper.metadata(),
        "grasps": results,
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] {output_path}")

    viz_path = output_path.with_name(output_path.stem + "_viz_data.pkl")
    with viz_path.open("wb") as f:
        pickle.dump({
            "rgb": rgb,
            "depth": depth,
            "seg": seg,
            "point_cloud": point_cloud,
            "target_points": target_points,
            "scripted_grasp": scripted_info,
            "objects": scene.get_object_poses(),
            "target_body_id": int(target_body_id),
            "target_object_name": target_object.name,
        }, f)
    print(f"[Saved] {viz_path}")

    gripper.remove()
    scene.disconnect()


if __name__ == "__main__":
    main()
