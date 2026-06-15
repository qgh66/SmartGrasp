#!/usr/bin/env python
"""Run the execution/reveal_api.py push plan in PyBullet and save GUI replay data."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation


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
    parser.add_argument("--rgb", help="Aligned real RGB image")
    parser.add_argument("--depth", help="Aligned real depth image or .npy")
    parser.add_argument("--mask", help="Binary mask of the object to push")
    parser.add_argument(
        "--intrinsics",
        help="JSON containing fx, fy, cx and cy for the real camera",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Raw depth units per meter; use 1 for depth already in meters",
    )
    parser.add_argument(
        "--plane-threshold",
        type=float,
        default=0.008,
        help="RANSAC table-plane distance threshold in meters",
    )
    parser.add_argument("--mass", type=float, default=0.05)
    parser.add_argument("--friction", type=float, default=0.7)
    parser.add_argument("--distance", type=float, default=0.05)
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "results", "reveal_push.json"),
    )
    return parser.parse_args()


def _load_intrinsics(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if "camera_intrinsics" in data:
        data = data["camera_intrinsics"]
    if "intrinsic_matrix" in data:
        matrix = np.asarray(data["intrinsic_matrix"], dtype=float).reshape(3, 3)
        data = {
            "fx": matrix[0, 0],
            "fy": matrix[1, 1],
            "cx": matrix[0, 2],
            "cy": matrix[1, 2],
        }
    intrinsics = {key: float(data[key]) for key in ("fx", "fy", "cx", "cy")}
    if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
        raise ValueError("Camera fx and fy must be positive")
    return intrinsics


def _load_depth(path, depth_scale):
    if depth_scale <= 0:
        raise ValueError("--depth-scale must be positive")
    depth_path = Path(path)
    if not depth_path.is_file():
        raise FileNotFoundError(
            f"Depth file not found: {depth_path}. "
            "Replace the example /path/to/... value with the aligned depth "
            "captured at the same time as the RGB image."
        )
    if depth_path.suffix.lower() == ".npy":
        raw_depth = np.load(depth_path)
    else:
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw_depth is None or raw_depth.ndim != 2:
        raise ValueError(f"Could not load a single-channel depth map: {path}")
    return raw_depth.astype(np.float32) / float(depth_scale)


def _deproject(depth, intrinsics):
    height, width = depth.shape
    xmap, ymap = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    z = depth
    x = (xmap - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (ymap - intrinsics["cy"]) * z / intrinsics["fy"]
    return np.stack([x, y, z], axis=-1)


def _build_real_rgbd_scene(args, output_path):
    required = {
        "--rgb": args.rgb,
        "--depth": args.depth,
        "--mask": args.mask,
        "--intrinsics": args.intrinsics,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Real RGB-D mode requires " + ", ".join(required.keys())
            + f"; missing {', '.join(missing)}")
    if args.obj:
        raise ValueError("--obj cannot be combined with real RGB-D mode")

    bgr = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        raise ValueError(f"Could not load RGB image: {args.rgb}")
    if mask is None:
        raise ValueError(f"Could not load mask image: {args.mask}")
    depth = _load_depth(args.depth, args.depth_scale)
    if bgr.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise ValueError(
            "RGB, depth and mask must be aligned and have identical dimensions")

    intrinsics = _load_intrinsics(args.intrinsics)
    height, width = depth.shape
    if not (0 <= intrinsics["cx"] < width and 0 <= intrinsics["cy"] < height):
        raise ValueError("Camera principal point lies outside the input image")

    points_image = _deproject(depth, intrinsics)
    valid = np.isfinite(depth) & (depth > 0)
    object_mask = valid & (mask > 0)
    table_candidates = valid & ~object_mask
    if np.count_nonzero(object_mask) < 50:
        raise ValueError("Object mask contains fewer than 50 valid depth pixels")
    if np.count_nonzero(table_candidates) < 100:
        raise ValueError("Not enough background depth pixels to fit the table")

    table_cloud = o3d.geometry.PointCloud()
    table_cloud.points = o3d.utility.Vector3dVector(
        points_image[table_candidates].astype(np.float64))
    plane, inliers = table_cloud.segment_plane(
        distance_threshold=args.plane_threshold,
        ransac_n=3,
        num_iterations=1500,
    )
    if len(inliers) < 100:
        raise ValueError("Could not find a stable table plane in the depth map")

    table_points = np.asarray(table_cloud.points)[inliers]
    table_center = np.median(table_points, axis=0)
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    if np.dot(normal, -table_center) < 0:
        normal = -normal
    camera_to_world = Rotation.align_vectors(
        np.array([[0.0, 0.0, 1.0]]),
        normal[None, :],
    )[0].as_matrix()

    camera_points = points_image[valid]
    world_points = (camera_to_world @ (camera_points - table_center).T).T
    object_points = (
        camera_to_world
        @ (points_image[object_mask] - table_center).T
    ).T
    object_center_xy = np.median(object_points[:, :2], axis=0)
    xy_shift = np.array(
        [0.30 - object_center_xy[0], -object_center_xy[1], 0.0])
    world_points += xy_shift
    object_points += xy_shift

    object_points = object_points[object_points[:, 2] > -args.plane_threshold]
    object_min = object_points.min(axis=0)
    object_max = object_points.max(axis=0)
    local_padding = 0.25
    local_scene_mask = (
        (world_points[:, 0] >= object_min[0] - local_padding)
        & (world_points[:, 0] <= object_max[0] + local_padding)
        & (world_points[:, 1] >= object_min[1] - local_padding)
        & (world_points[:, 1] <= object_max[1] + local_padding)
        & (world_points[:, 2] >= -args.plane_threshold * 2.0)
        & (world_points[:, 2] <= object_max[2] + 0.20)
    )
    world_points = world_points[local_scene_mask]
    if len(world_points) > 100000:
        sample_indices = np.linspace(
            0, len(world_points) - 1, 100000).astype(np.int64)
        world_points = world_points[sample_indices]

    object_cloud = o3d.geometry.PointCloud()
    object_cloud.points = o3d.utility.Vector3dVector(object_points)
    object_cloud = object_cloud.voxel_down_sample(voxel_size=0.002)
    if len(object_cloud.points) < 20:
        raise ValueError("Too few object points remain after depth processing")
    hull, _ = object_cloud.compute_convex_hull()
    hull.compute_vertex_normals()

    hull_path = os.path.splitext(output_path)[0] + "_observed_hull.obj"
    if not o3d.io.write_triangle_mesh(hull_path, hull, write_triangle_uvs=False):
        raise RuntimeError(f"Failed to save observed object hull: {hull_path}")

    colors = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return {
        "obj_path": hull_path,
        "load_position": (0.0, 0.0, 0.0),
        "rgb": colors,
        "depth": depth,
        "point_cloud": world_points.astype(np.float32),
        "source": {
            "mode": "real_rgbd",
            "rgb_path": os.path.abspath(args.rgb),
            "depth_path": os.path.abspath(args.depth),
            "mask_path": os.path.abspath(args.mask),
            "intrinsics_path": os.path.abspath(args.intrinsics),
            "intrinsics": intrinsics,
            "depth_scale": args.depth_scale,
            "table_plane_camera": plane,
            "camera_to_world_rotation": camera_to_world,
            "geometry": "single-view observed convex hull",
        },
    }


def _build_mesh_scene(args):
    obj_path = args.obj or os.path.join(
        pybullet_data.getDataPath(), "cube_small.urdf")
    return {
        "obj_path": obj_path,
        "load_position": (0.30, 0.0, 0.04),
        "source": {
            "mode": "mesh_with_virtual_rgbd",
            "mesh_path": os.path.abspath(obj_path),
        },
    }


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
    if args.mass <= 0:
        raise ValueError("--mass must be positive")
    if args.friction < 0:
        raise ValueError("--friction must be non-negative")

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    real_mode = any((args.rgb, args.depth, args.mask, args.intrinsics))
    scene_data = (
        _build_real_rgbd_scene(args, output_path)
        if real_mode else _build_mesh_scene(args)
    )
    obj_path = scene_data["obj_path"]

    scene = SimulationScene(gui=args.gui)
    gripper = ParallelJawGripper()
    try:
        scene.connect()
        scene.load_plane()
        obj_id = scene.load_object(
            obj_path,
            position=scene_data["load_position"],
            mass=args.mass,
            lateral_friction=args.friction,
        )
        scene.step(240)

        aabb_min, aabb_max = p.getAABB(obj_id)
        center = (
            np.asarray(aabb_min, dtype=float)
            + np.asarray(aabb_max, dtype=float)
        ) / 2.0
        object_aabb_size = (
            np.asarray(aabb_max, dtype=float)
            - np.asarray(aabb_min, dtype=float)
        )
        obj_pos, obj_orn = scene.get_object_pose(obj_id)

        if real_mode:
            rgb = scene_data["rgb"]
            depth = scene_data["depth"]
            point_cloud = scene_data["point_cloud"]
        else:
            camera = VirtualCamera(
                position=(0.30, -0.45, 0.45),
                target=(0.30, 0.0, 0.04),
                near=0.01,
                far=5.0,
            )
            rgb, depth, _ = camera.capture()
            point_cloud = camera.generate_point_cloud(
                depth, num_points=20000).numpy()
            point_cloud = point_cloud[0]

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
            point_cloud=point_cloud,
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
            "data_source": scene_data["source"],
            "obj_path": obj_path if obj_path.endswith(".obj") else None,
            "object_position": obj_pos,
            "object_orientation": obj_orn,
            "object_aabb_size": object_aabb_size,
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
                "data_source": scene_data["source"],
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
