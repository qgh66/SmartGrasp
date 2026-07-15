#!/usr/bin/env python
"""Sequentially grasp every configured object in one PyBullet scene."""

import json
import os
import pickle
import random
import sys
import time

import numpy as np
import pybullet as p
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.path.insert(0, os.path.join(ROOT, "graspnet_api"))

from demo_closed_loop import (
    crop_to_object,
    filter_collision_grasps,
    filter_grasps_to_object,
    load_scene_config,
    prefer_topdown_grasps,
)
from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode
from simulation.camera import VirtualCamera
from simulation.evaluator import GraspEvaluator
from simulation.robot_gripper import JakaZu3Robotiq85Gripper
from simulation.scene import SimulationScene


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _load_network(args, device):
    checkpoint = args.ckpt or os.path.join(ROOT, "checkpoints", "checkpoint-rs.tar")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"[Checkpoint] {checkpoint}")
    network = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint_data["model_state_dict"])
    network.to(device)
    network.eval()
    print(f'  GraspNet epoch {checkpoint_data.get("epoch", "?")}')
    return network


def _hold_initial_pose(scene, gripper, seconds, gui):
    seconds = max(0.0, float(seconds))
    print(f"  初始关节位姿保持 {seconds:.1f} 秒")
    if seconds == 0.0:
        return
    if not gui:
        time.sleep(seconds)
        return
    p.addUserDebugText(
        f"INITIAL POSE - grasp starts in {seconds:.1f}s",
        [0.05, -0.25, 0.45],
        [0.1, 0.8, 1.0],
        textSize=1.5,
        lifeTime=seconds,
    )
    frame_period = 1.0 / 60.0
    for _ in range(max(1, int(round(seconds / frame_period)))):
        gripper._hold_current_joints()
        scene.step()
        time.sleep(frame_period)


def _capture_target(scene, camera, target_body_id, scene_config):
    rgb, depth, segmentation = camera.capture()
    point_cloud = camera.generate_point_cloud(depth, num_points=20000).numpy()
    object_clouds = camera.generate_object_point_clouds(
        depth, segmentation, scene.object_ids
    )
    target_points = object_clouds.get(int(target_body_id))
    crop_config = scene_config.get("crop", {})
    cropped_cloud = crop_to_object(
        point_cloud,
        object_points=target_points,
        margin=float(crop_config.get("margin", 0.05)),
        num_points=int(crop_config.get("num_points", 20000)),
        table_z=float(crop_config.get("table_z", 0.005)),
    )
    return rgb, depth, segmentation, cropped_cloud, object_clouds, target_points


def _infer_target_grasps(network, device, camera, point_cloud, target_points, config):
    camera_points = camera.world_to_camera_points(point_cloud[0]).astype(np.float32)
    cloud_tensor = torch.from_numpy(camera_points[np.newaxis]).to(device)
    with torch.no_grad():
        predictions = pred_decode(network({"point_clouds": cloud_tensor}))
    grasps = GraspGroup(predictions[0].detach().cpu().numpy())
    grasps = camera.camera_grasps_to_world(grasps)
    grasps.sort_by_score()
    raw_count = len(grasps)
    if raw_count == 0:
        return grasps, {
            "grasp_filter": {"enabled": True, "kept": 0, "total": 0},
            "collision_filter": {"enabled": True, "kept": 0, "total": 0},
            "topdown_filter": {"enabled": True, "kept": 0, "total": 0},
        }

    filter_config = config.get("grasp_filter", {})
    grasps, grasp_stats = filter_grasps_to_object(
        grasps,
        target_points,
        max_center_dist=float(filter_config.get("max_center_dist", 0.04)),
        bbox_margin=float(filter_config.get("bbox_margin", 0.04)),
        min_inner_points=int(filter_config.get("min_inner_points", 5)),
    )
    grasps, collision_stats = filter_collision_grasps(
        grasps, point_cloud[0], config.get("collision_filter", {})
    )
    grasps, topdown_stats = prefer_topdown_grasps(
        grasps, config.get("topdown_filter", {})
    )
    print(
        f"  候选: raw={raw_count}, target={grasp_stats.get('kept')}, "
        f"collision={collision_stats.get('kept')}, topdown={len(grasps)}"
    )
    return grasps, {
        "grasp_filter": grasp_stats,
        "collision_filter": collision_stats,
        "topdown_filter": topdown_stats,
    }


def run_all_objects(args):
    if not args.scene_config:
        raise ValueError("--all-objects requires --scene-config")
    if args.max_candidates_per_object <= 0:
        raise ValueError("--max-candidates-per-object must be greater than zero")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[All Objects] seed={args.seed}, device={device}")
    network = _load_network(args, device)

    config = load_scene_config(args.scene_config)
    scene = SimulationScene(gui=args.gui)
    scene.connect()
    scene.load_plane()
    scene.load_objects(config["_resolved_objects"])
    for _ in range(int(config.get("settle_steps", 300))):
        scene.step()

    capture_pose = config.get("capture_joint_pose_deg")
    place_pose = config.get("place_target_joint_pose_deg")
    gripper = JakaZu3Robotiq85Gripper(
        planner=None,
        initial_joint_pose_deg=capture_pose,
        robot_base_yaw_deg=float(config.get("robot_base_yaw_deg", 0.0)),
        gui_motion_step_delay=(0.003 / args.gui_speed) if args.gui else 0.0,
    )
    gripper.load()
    gripper.move_to_joint_pose_deg(capture_pose)
    print(f"  场景已倒入并稳定: {len(scene.object_ids)} 个物体")
    _hold_initial_pose(
        scene, gripper, args.initial_pose_hold_seconds, args.gui
    )

    camera_config = config.get("camera", {})
    camera = VirtualCamera(
        position=tuple(camera_config.get("position", (0.3, 0.0, 0.5))),
        target=tuple(camera_config.get("target", (0.3, 0.0, 0.05))),
        near=float(camera_config.get("near", 0.01)),
        far=float(camera_config.get("far", 5.0)),
        width=int(camera_config.get("width", 1280)),
        height=int(camera_config.get("height", 720)),
        fov=float(camera_config.get("fov", 60.0)),
    )

    object_results = []
    last_visualization = None
    target_names = args.target_objects
    if target_names is None and args.target_object:
        target_names = [args.target_object]
    if target_names:
        target_names = list(dict.fromkeys(target_names))
        target_ids = [scene.get_body_id_by_name(name) for name in target_names]
        print(f"  按指定顺序处理: {' -> '.join(target_names)}")
    else:
        target_ids = list(scene.object_ids)
    for object_number, target_body_id in enumerate(target_ids, start=1):
        target = scene.get_object_info(target_body_id)
        print(f"\n[{object_number}/{len(target_ids)}] 目标物体: {target.name}")
        gripper.release_grasp()
        gripper.set_opening(gripper._max_opening)
        gripper.move_to_joint_pose_deg(capture_pose)

        rgb, depth, segmentation, point_cloud, object_clouds, target_points = (
            _capture_target(scene, camera, target_body_id, config)
        )
        target_point_count = int(len(target_points)) if target_points is not None else 0
        if target_point_count == 0:
            print("  跳过: 相机看不到该物体")
            object_results.append({
                "target_body_id": int(target_body_id),
                "target_object_name": target.name,
                "success": False,
                "failure_reason": "target_not_visible",
                "evaluated_candidates": 0,
                "grasps": [],
            })
            continue

        grasps, filter_stats = _infer_target_grasps(
            network, device, camera, point_cloud, target_points, config
        )
        if len(grasps) == 0:
            print("  跳过: 过滤后没有候选")
            object_results.append({
                "target_body_id": int(target_body_id),
                "target_object_name": target.name,
                "success": False,
                "failure_reason": "no_filtered_candidates",
                "evaluated_candidates": 0,
                "grasps": [],
                **filter_stats,
            })
            continue

        requested_count = len(grasps) if args.stop_on_success else args.top_k
        evaluation_count = min(
            requested_count,
            args.max_candidates_per_object,
            len(grasps),
        )
        evaluator = GraspEvaluator(
            object_id=target_body_id,
            gripper=gripper,
            point_cloud=target_points,
            gui=args.gui,
            assisted_grasp=args.assisted_grasp,
            validate_target_center=False,
            place_target_joint_pose_deg=place_pose,
            gui_speed=args.gui_speed,
        )
        results = evaluator.evaluate(
            grasps,
            top_k=evaluation_count,
            stop_on_success=args.stop_on_success,
            preserve_success_state=True,
        )
        successful = next((result for result in results if result["success"]), None)
        print(
            f"  结果: {'成功' if successful else '失败'}, "
            f"已测试 {len(results)}/{len(grasps)} 个候选"
        )
        object_results.append({
            "target_body_id": int(target_body_id),
            "target_object_name": target.name,
            "obj_path": target.path,
            "success": successful is not None,
            "successful_grasp_index": (
                int(successful["grasp_index"]) if successful else None
            ),
            "evaluated_candidates": len(results),
            "filtered_candidates": len(grasps),
            "target_point_count": target_point_count,
            "grasps": results,
            **filter_stats,
        })
        last_visualization = {
            "rgb": rgb,
            "depth": depth,
            "point_cloud": point_cloud,
            "seg": segmentation,
            "object_point_counts": {
                body_id: int(len(points)) for body_id, points in object_clouds.items()
            },
            "target_body_id": int(target_body_id),
            "target_object_name": target.name,
        }

    successful_objects = sum(item["success"] for item in object_results)
    output = _json_safe({
        "mode": "all_objects_sequential_pick_and_place",
        "scene_config": config["_path"],
        "seed": int(args.seed),
        "capture_joint_pose_deg": capture_pose,
        "robot_base_yaw_deg": float(config.get("robot_base_yaw_deg", 0.0)),
        "initial_pose_hold_seconds": float(args.initial_pose_hold_seconds),
        "max_candidates_per_object": int(args.max_candidates_per_object),
        "place_target_joint_pose_deg": place_pose,
        "gui_speed": float(args.gui_speed),
        "assisted_grasp": bool(args.assisted_grasp),
        "object_total": len(object_results),
        "object_success": successful_objects,
        "objects": object_results,
        "final_scene_objects": scene.get_object_poses(),
        "gripper": gripper.metadata(),
    })
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)

    visualization_path = args.output.replace(".json", "_viz_data.pkl")
    if last_visualization is not None:
        last_visualization.update({
            "objects": scene.get_object_poses(),
            "scene_config": config["_path"],
        })
        with open(visualization_path, "wb") as visualization_file:
            pickle.dump(last_visualization, visualization_file)

    print(f"\n全部物体结果: {successful_objects}/{len(object_results)} 成功")
    print(f"结果已保存: {args.output}")
    gripper.remove()
    scene.disconnect()
    return output
