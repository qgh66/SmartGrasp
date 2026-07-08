"""Reveal push execution for JAKA Zu3 + Robotiq in PyBullet."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from execution.reveal_api import execute_reveal_action
from simulation.camera import VirtualCamera
from simulation.robot_gripper import JakaZu3Robotiq85Gripper
from simulation.scene import SimulationScene


TABLE_CLEARANCE = 0.005
DEFAULT_PUSH_PENETRATION = 0.04
DEFAULT_CONTACT_MARGIN = 0.02


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def resolve_path(path: str | os.PathLike[str], *, config_dir: Path | None = None) -> Path:
    raw = Path(os.path.expanduser(str(path)))
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] == "graspnet-workspace":
        return (REPO_ROOT / raw).resolve()
    candidates = []
    if config_dir is not None:
        candidates.append(config_dir / raw)
    candidates.extend([ROOT / raw, REPO_ROOT / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_scene_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
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


def select_scene_object(scene: SimulationScene, object_name: str | None):
    if object_name:
        body_id = scene.get_body_id_by_name(object_name)
        return body_id, scene.get_object_info(body_id)
    if not scene.object_ids:
        raise RuntimeError("Scene has no loaded objects")
    body_id = scene.object_ids[0]
    return body_id, scene.get_object_info(body_id)


def make_virtual_camera(config: dict[str, Any]) -> VirtualCamera:
    camera_cfg = config.get("camera", {})
    return VirtualCamera(
        position=tuple(camera_cfg.get("position", (0.3, 0.0, 0.5))),
        target=tuple(camera_cfg.get("target", (0.3, 0.0, 0.05))),
        width=int(camera_cfg.get("width", 1280)),
        height=int(camera_cfg.get("height", 720)),
        fov=float(camera_cfg.get("fov", 60.0)),
        near=float(camera_cfg.get("near", 0.01)),
        far=float(camera_cfg.get("far", 5.0)),
    )


def object_center_from_aabb(body_id: int) -> np.ndarray:
    aabb_min, aabb_max = p.getAABB(body_id)
    return 0.5 * (np.asarray(aabb_min, dtype=float) + np.asarray(aabb_max, dtype=float))


def object_aabb_size(body_id: int) -> np.ndarray:
    aabb_min, aabb_max = p.getAABB(body_id)
    return np.asarray(aabb_max, dtype=float) - np.asarray(aabb_min, dtype=float)


def is_center_near_aabb(center: np.ndarray, body_id: int, margin: float = 0.03) -> bool:
    aabb_min, aabb_max = p.getAABB(body_id)
    lo = np.asarray(aabb_min, dtype=float) - float(margin)
    hi = np.asarray(aabb_max, dtype=float) + float(margin)
    return bool(np.all(center >= lo) and np.all(center <= hi))


def rotation_for_push(direction: np.ndarray, opening_axis: np.ndarray | None = None) -> np.ndarray:
    """Build a GraspNet-style pose: local x downward, local z along push."""
    push_axis = np.asarray(direction, dtype=float)
    push_axis = push_axis / max(np.linalg.norm(push_axis), 1e-8)
    approach_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    if opening_axis is None:
        opening_axis = np.cross(push_axis, approach_axis)
    opening_axis = np.asarray(opening_axis, dtype=float)
    opening_axis = opening_axis - np.dot(opening_axis, approach_axis) * approach_axis
    opening_axis = opening_axis / max(np.linalg.norm(opening_axis), 1e-8)
    side_axis = np.cross(approach_axis, opening_axis)
    if np.dot(side_axis, push_axis) < 0:
        opening_axis = -opening_axis
        side_axis = -side_axis
    return np.column_stack([approach_axis, opening_axis, side_axis]).astype(float)


def snapshot(body_id: int, gripper: JakaZu3Robotiq85Gripper | None = None) -> dict[str, Any]:
    obj_pos, obj_orn = p.getBasePositionAndOrientation(body_id)
    item = {
        "obj_pos": list(obj_pos),
        "obj_orn": list(obj_orn),
    }
    if gripper is not None and gripper.base_id is not None:
        tcp_pos, tcp_orn = gripper.get_tcp_pose()
        item.update({
            "gripper_pos": list(tcp_pos),
            "gripper_orn": list(tcp_orn),
            "opening": float(gripper._current_opening),
        })
        extra = gripper.snapshot_extra()
        item["robot"] = extra
    return item


class RevealPushExecutor:
    """Execute a side push using the current JAKA+Robotiq gripper interface."""

    def __init__(self, object_id: int, gripper: JakaZu3Robotiq85Gripper, gui: bool = False):
        self.object_id = int(object_id)
        self.gripper = gripper
        self.gui = gui

    def step(self, steps: int = 1):
        for _ in range(int(steps)):
            p.stepSimulation()

    def execute_push(
        self,
        *,
        center_point,
        direction=(1.0, 0.0, 0.0),
        move_distance: float = 0.05,
        approach_distance: float = 0.06,
        contact_margin: float = DEFAULT_CONTACT_MARGIN,
        penetration_distance: float = DEFAULT_PUSH_PENETRATION,
        approach_steps: int = 18,
        push_steps: int = 36,
        retreat_steps: int = 12,
        closed_width: float = 0.012,
    ) -> dict[str, Any]:
        center = np.asarray(center_point, dtype=float)
        if center.shape != (3,):
            raise ValueError("center_point must contain exactly three XYZ values")
        direction = np.asarray(direction, dtype=float)
        direction = direction / max(np.linalg.norm(direction), 1e-8)
        move_distance = float(move_distance)
        if move_distance <= 0:
            raise ValueError("move_distance must be positive")

        frame_log: list[dict[str, Any]] = []
        aabb_min, aabb_max = p.getAABB(self.object_id)
        aabb_min = np.asarray(aabb_min, dtype=float)
        aabb_max = np.asarray(aabb_max, dtype=float)

        # Pick the surface opposite to the push direction, so the gripper pushes
        # through the object instead of approaching from the far side.
        contact_face = np.where(direction >= 0, aabb_min, aabb_max)
        face_point = center.copy()
        dominant_axis = int(np.argmax(np.abs(direction)))
        face_point[dominant_axis] = contact_face[dominant_axis]
        face_point[2] = max(
            center[2],
            aabb_min[2] + 0.6 * (aabb_max[2] - aabb_min[2]),
            TABLE_CLEARANCE + 0.01,
        )

        rotation = rotation_for_push(direction)
        # TCP is not the outermost collision geometry of the Robotiq hand. Start
        # before the near face and move through the AABB so the fingers/palm
        # actually contact the object instead of stopping just outside it.
        contact_pos = face_point - direction * float(contact_margin)
        pre_push_pos = contact_pos - direction * float(approach_distance)
        push_end_pos = contact_pos + direction * (float(move_distance) + float(penetration_distance))
        retreat_pos = push_end_pos - direction * min(float(approach_distance), 0.03)

        self.gripper.release_grasp()
        self.gripper.set_opening(closed_width)
        self.gripper.set_pose(pre_push_pos, rotation)
        self.step(20)
        frame_log.append({"phase": "push_ready", "step": "ready", **snapshot(self.object_id, self.gripper)})

        for i in range(max(1, int(approach_steps))):
            frac = (i + 1) / max(1, int(approach_steps))
            pos = pre_push_pos + (contact_pos - pre_push_pos) * frac
            self.gripper.set_pose(pos, rotation)
            self.step(3)
            frame_log.append({"phase": "push_approach", "step": i, **snapshot(self.object_id, self.gripper)})

        start_pos, start_orn = p.getBasePositionAndOrientation(self.object_id)
        start_pos = np.asarray(start_pos, dtype=float)
        for i in range(max(1, int(push_steps))):
            frac = (i + 1) / max(1, int(push_steps))
            pos = contact_pos + (push_end_pos - contact_pos) * frac
            self.gripper.set_pose(pos, rotation)
            self.step(4)
            frame_log.append({"phase": "push", "step": i, **snapshot(self.object_id, self.gripper)})

        for i in range(max(1, int(retreat_steps))):
            frac = (i + 1) / max(1, int(retreat_steps))
            pos = push_end_pos + (retreat_pos - push_end_pos) * frac
            self.gripper.set_pose(pos, rotation)
            self.step(3)
            frame_log.append({"phase": "retreat", "step": i, **snapshot(self.object_id, self.gripper)})

        self.step(80)
        final_pos, final_orn = p.getBasePositionAndOrientation(self.object_id)
        final_pos = np.asarray(final_pos, dtype=float)
        displacement = final_pos - start_pos
        signed_displacement = float(np.dot(displacement, direction))
        success_threshold = min(0.01, move_distance * 0.2)
        success = signed_displacement >= success_threshold
        frame_log.append({
            "phase": "done",
            "step": "final",
            "success": success,
            **snapshot(self.object_id, self.gripper),
        })

        return {
            "action_type": "push",
            "success": success,
            "score": 1.0,
            "lift_z": float(final_pos[2]),
            "translation": center,
            "rotation": rotation,
            "width": float(closed_width),
            "depth": 0.0,
            "push_direction": direction,
            "requested_distance": move_distance,
            "actual_displacement": displacement,
            "signed_displacement": signed_displacement,
            "success_threshold": float(success_threshold),
            "start_position": start_pos,
            "start_orientation": start_orn,
            "target_position": start_pos + direction * move_distance,
            "final_position": final_pos,
            "final_orientation": final_orn,
            "contact_position": contact_pos,
            "pre_push_position": pre_push_pos,
            "push_end_position": push_end_pos,
            "contact_margin": float(contact_margin),
            "penetration_distance": float(penetration_distance),
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "frame_log": frame_log,
            "request_reloop": True,
        }


def run_reveal_push_scene(
    *,
    scene_config: str | os.PathLike[str],
    object_name: str | None,
    center_point=None,
    direction=(1.0, 0.0, 0.0),
    move_distance: float = 0.05,
    output: str | os.PathLike[str] = "results/reveal_push.json",
    gui: bool = False,
) -> dict[str, Any]:
    config = load_scene_config(scene_config)
    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene = SimulationScene(gui=gui)
    gripper = JakaZu3Robotiq85Gripper(planner=None)
    try:
        scene.connect()
        scene.load_plane()
        scene.load_objects(config["_resolved_objects"])
        target_body_id, target_object = select_scene_object(scene, object_name)
        scene.step(int(config.get("settle_steps", 300)))

        measured_center = object_center_from_aabb(target_body_id)
        if center_point is None:
            center = measured_center
            center_source = "pybullet_aabb"
        else:
            requested_center = np.asarray(center_point, dtype=float)
            if is_center_near_aabb(requested_center, target_body_id):
                center = requested_center
                center_source = "request"
            else:
                center = measured_center
                center_source = "pybullet_aabb_fallback"
        plan = execute_reveal_action(
            occluder_id=target_body_id,
            center_point=center,
            action_type="push",
            move_distance=float(move_distance),
        )
        push_vector = np.asarray(plan["push_vector"], dtype=float)
        push_direction = np.asarray(direction, dtype=float)
        if np.linalg.norm(push_direction) < 1e-8:
            push_direction = push_vector
        push_direction = push_direction / max(np.linalg.norm(push_direction), 1e-8)

        camera = make_virtual_camera(config)
        before_rgb, before_depth, before_seg = camera.capture()
        before_pc = camera.generate_point_cloud(before_depth, num_points=20000).numpy()
        before_object_clouds = camera.generate_object_point_clouds(before_depth, before_seg, scene.object_ids)

        pose_before = scene.get_object_poses()
        object_pose_before = pose_before[int(target_body_id)]
        aabb_size = object_aabb_size(target_body_id)

        gripper.load()
        executor = RevealPushExecutor(target_body_id, gripper, gui=gui)
        result = executor.execute_push(
            center_point=plan["start_translation"],
            direction=push_direction,
            move_distance=float(move_distance),
        )
        result["grasp_index"] = 0
        result["request_reloop"] = True
        result["reveal_plan"] = plan

        after_rgb, after_depth, after_seg = camera.capture()
        after_pc = camera.generate_point_cloud(after_depth, num_points=20000).numpy()
        after_object_clouds = camera.generate_object_point_clouds(after_depth, after_seg, scene.object_ids)
        pose_after = scene.get_object_poses()

        output_data = json_safe({
            "total": 1,
            "success": int(result["success"]),
            "action_mode": "reveal",
            "action_type": "push",
            "scene_config": config["_path"],
            "target_object_name": target_object.name,
            "target_body_id": int(target_body_id),
            "obj_path": target_object.path,
            "object_position": object_pose_before["position"],
            "object_orientation": object_pose_before["orientation"],
            "object_aabb_size": aabb_size,
            "center_source": center_source,
            "measured_center": measured_center,
            "objects_before": pose_before,
            "objects_after": pose_after,
            "object_point_counts_before": {int(k): int(len(v)) for k, v in before_object_clouds.items()},
            "object_point_counts_after": {int(k): int(len(v)) for k, v in after_object_clouds.items()},
            "reveal_plan": plan,
            "gripper": gripper.metadata(),
            "grasps": [result],
            "request_reloop": True,
        })
        output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

        viz_path = output_path.with_name(output_path.stem + "_viz_data.pkl")
        with viz_path.open("wb") as f:
            pickle.dump({
                "rgb": before_rgb,
                "depth": before_depth,
                "seg": before_seg,
                "point_cloud": before_pc,
                "after_rgb": after_rgb,
                "after_depth": after_depth,
                "after_seg": after_seg,
                "after_point_cloud": after_pc,
                "object_point_counts": {int(k): int(len(v)) for k, v in before_object_clouds.items()},
                "object_point_counts_after": {int(k): int(len(v)) for k, v in after_object_clouds.items()},
                "objects": pose_before,
                "objects_after": pose_after,
                "target_body_id": int(target_body_id),
                "target_object_name": target_object.name,
                "scene_config": config["_path"],
                "obj_path": target_object.path,
                "object_orientation": object_pose_before["orientation"],
                "object_aabb_size": aabb_size,
                "center_source": center_source,
                "measured_center": measured_center,
            }, f)

        return {
            "result": output_data,
            "result_json": str(output_path),
            "viz_data_pkl": str(viz_path),
        }
    finally:
        if gripper.base_id is not None:
            gripper.remove()
        scene.disconnect()
