#!/usr/bin/env python
"""Run the execution/reveal_api.py push plan in PyBullet and save GUI replay data."""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pybullet as p
import pybullet_data


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMARTGRASP_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, SMARTGRASP_ROOT)

from execution.reveal_api import execute_reveal_action
from simulation.camera import VirtualCamera
from simulation.evaluator import GraspEvaluator
from simulation.gripper import ParallelJawGripper
from simulation.scene import SimulationScene


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Reveal push in PyBullet and save a Dash replay")
    parser.add_argument(
        "--obj",
        default=None,
        help="Object .obj/.urdf path (default: PyBullet cube_small.urdf)",
    )
    parser.add_argument("--distance", type=float, default=0.05)
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "results", "reveal_push.json"),
    )
    return parser.parse_args()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main():
    args = parse_args()
    if args.distance <= 0:
        raise ValueError("--distance must be positive")

    obj_path = args.obj or os.path.join(
        pybullet_data.getDataPath(), "cube_small.urdf")
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scene = SimulationScene(gui=args.gui)
    gripper = ParallelJawGripper()
    try:
        scene.connect()
        scene.load_plane()
        obj_id = scene.load_object(
            obj_path,
            position=(0.30, 0.0, 0.04),
            mass=0.05,
            lateral_friction=0.7,
        )
        scene.step(240)

        aabb_min, aabb_max = p.getAABB(obj_id)
        center = (
            np.asarray(aabb_min, dtype=float)
            + np.asarray(aabb_max, dtype=float)
        ) / 2.0
        obj_pos, obj_orn = scene.get_object_pose(obj_id)

        camera = VirtualCamera(
            position=(0.30, -0.45, 0.45),
            target=(0.30, 0.0, 0.04),
            near=0.01,
            far=5.0,
        )
        rgb, depth, _ = camera.capture()
        point_cloud = camera.generate_point_cloud(
            depth, num_points=20000).numpy()

        plan = execute_reveal_action(
            occluder_id=obj_id,
            center_point=center,
            action_type="push",
            move_distance=args.distance,
        )

        gripper.load()
        evaluator = GraspEvaluator(
            object_id=obj_id,
            gripper=gripper,
            point_cloud=point_cloud[0],
            gui=args.gui,
        )
        result = evaluator.evaluate_push(
            center_point=plan["start_translation"],
            push_distance_x=plan["push_vector"][0],
            rotation_matrix=plan["default_rotation"],
        )
        result["grasp_index"] = 0
        result["request_reloop"] = plan["request_reloop"]

        output = _json_safe({
            "total": 1,
            "success": int(result["success"]),
            "action_mode": "reveal",
            "action_type": "push",
            "obj_path": obj_path if obj_path.endswith(".obj") else None,
            "object_position": obj_pos,
            "object_orientation": obj_orn,
            "reveal_plan": plan,
            "grasps": [result],
        })
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)

        viz_path = os.path.splitext(output_path)[0] + "_viz_data.pkl"
        with open(viz_path, "wb") as file:
            pickle.dump({
                "rgb": rgb,
                "depth": depth,
                "point_cloud": point_cloud,
                "object_orientation": list(obj_orn),
            }, file)

        displacement = result["signed_displacement"]
        print(f"[Reveal Push] success={result['success']}, "
              f"displacement_x={displacement:.4f} m")
        print(f"[Result] {output_path}")
        print(f"[Viz] {viz_path}")
    finally:
        if gripper.base_id is not None:
            gripper.remove()
        scene.disconnect()


if __name__ == "__main__":
    main()
