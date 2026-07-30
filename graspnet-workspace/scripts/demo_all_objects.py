#!/usr/bin/env python
"""Sequential and continuous multi-object grasp workflows."""

import json
import os
import pickle
import random
import sys
import time
import atexit

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
from simulation.gripper_factory import create_gripper
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


def _ordered_target_ids(scene, args, config):
    """Resolve an explicit order without changing the scene registry order."""
    target_names = args.target_objects
    if target_names is None and args.target_object:
        target_names = [args.target_object]
    if target_names is None and args.continuous_grasp:
        target_names = (
            config.get("continuous_grasp", {}).get("target_order")
        )

    if target_names:
        unique_names = list(dict.fromkeys(target_names))
        available_names = {
            item.name for item in scene.get_object_registry().values()
        }
        unknown_names = [
            name for name in unique_names
            if name not in available_names
        ]
        if unknown_names:
            raise ValueError(
                "Unknown target object name(s): "
                + ", ".join(unknown_names)
            )
        return [
            scene.get_body_id_by_name(name) for name in unique_names
        ], unique_names

    target_ids = list(scene.object_ids)
    return target_ids, [
        scene.get_object_info(body_id).name for body_id in target_ids
    ]


def _attempt_target(
    *,
    scene,
    gripper,
    camera,
    network,
    device,
    config,
    args,
    target_body_id,
    capture_pose,
    place_pose,
    release_after_place,
    release_settle_steps,
):
    """Capture the current scene and execute one target's best candidates."""
    target = scene.get_object_info(target_body_id)
    gripper.release_grasp()
    gripper.set_opening(gripper._max_opening)
    gripper.move_to_joint_pose_deg(capture_pose)

    rgb, depth, segmentation, point_cloud, object_clouds, target_points = (
        _capture_target(scene, camera, target_body_id, config)
    )
    visualization = {
        "rgb": rgb,
        "depth": depth,
        "point_cloud": point_cloud,
        "seg": segmentation,
        "object_point_counts": {
            body_id: int(len(points))
            for body_id, points in object_clouds.items()
        },
        "target_body_id": int(target_body_id),
        "target_object_name": target.name,
    }
    target_point_count = (
        int(len(target_points)) if target_points is not None else 0
    )
    if target_point_count == 0:
        print("  跳过本轮: 相机看不到该物体")
        return {
            "target_body_id": int(target_body_id),
            "target_object_name": target.name,
            "obj_path": target.path,
            "success": False,
            "failure_reason": "target_not_visible",
            "evaluated_candidates": 0,
            "target_point_count": 0,
            "grasps": [],
        }, visualization

    grasps, filter_stats = _infer_target_grasps(
        network, device, camera, point_cloud, target_points, config
    )
    if len(grasps) == 0:
        print("  跳过本轮: 过滤后没有候选")
        return {
            "target_body_id": int(target_body_id),
            "target_object_name": target.name,
            "obj_path": target.path,
            "success": False,
            "failure_reason": "no_filtered_candidates",
            "evaluated_candidates": 0,
            "target_point_count": target_point_count,
            "grasps": [],
            **filter_stats,
        }, visualization

    requested_count = len(grasps) if args.stop_on_success else args.top_k
    evaluation_count = min(
        requested_count,
        args.max_candidates_per_object,
        len(grasps),
    )
    activated_from_staging = scene.activate_staged_object(target_body_id)
    if activated_from_staging:
        print(
            f"  目标已恢复动态质量: {target.name} "
            f"({target.mass:.3f} kg)"
        )
    evaluator = GraspEvaluator(
        object_id=target_body_id,
        gripper=gripper,
        point_cloud=target_points,
        gui=args.gui,
        assisted_grasp=args.assisted_grasp,
        validate_target_center=False,
        scene_object_ids=scene.object_ids,
        place_target_joint_pose_deg=place_pose,
        release_after_place=release_after_place,
        release_settle_steps=release_settle_steps,
        gui_speed=args.gui_speed,
    )
    results = evaluator.evaluate(
        grasps,
        top_k=evaluation_count,
        stop_on_success=args.stop_on_success,
        preserve_success_state=True,
    )
    for result in results:
        placement = result.get("placement") or {}
        status = "SUCCESS" if result["success"] else "FAIL"
        failure_reason = result.get("failure_reason") or "none"
        print(
            f"  {status} candidate={result['grasp_index']}, "
            f"score={result['score']:.3f}, "
            f"transported={placement.get('object_followed_to_place', False)}, "
            f"released={placement.get('released_after_place', False)}, "
            f"reason={failure_reason}"
        )
    successful = next(
        (result for result in results if result["success"]),
        None,
    )
    if activated_from_staging:
        if successful:
            scene.finish_staged_object(target_body_id)
        else:
            scene.restage_object(target_body_id)
    print(
        f"  结果: {'成功' if successful else '失败'}, "
        f"已测试 {len(results)}/{len(grasps)} 个候选"
    )
    return {
        "target_body_id": int(target_body_id),
        "target_object_name": target.name,
        "obj_path": target.path,
        "success": successful is not None,
        "successful_grasp_index": (
            int(successful["grasp_index"]) if successful else None
        ),
        "failure_reason": (
            None
            if successful
            else (
                results[-1].get("failure_reason")
                if results
                else "no_evaluated_candidates"
            )
        ),
        "evaluated_candidates": len(results),
        "filtered_candidates": len(grasps),
        "target_point_count": target_point_count,
        "activated_from_staging": activated_from_staging,
        "grasps": results,
        **filter_stats,
    }, visualization


def _summarize_continuous_objects(scene, target_ids, attempts):
    """Build one compact final record per requested object."""
    summaries = []
    for target_body_id in target_ids:
        target = scene.get_object_info(target_body_id)
        object_attempts = [
            attempt
            for attempt in attempts
            if attempt["target_body_id"] == int(target_body_id)
        ]
        successful_attempt = next(
            (attempt for attempt in object_attempts if attempt["success"]),
            None,
        )
        last_attempt = object_attempts[-1] if object_attempts else {}
        summaries.append({
            "target_body_id": int(target_body_id),
            "target_object_name": target.name,
            "success": successful_attempt is not None,
            "attempt_count": len(object_attempts),
            "successful_attempt_index": (
                successful_attempt.get("attempt_index")
                if successful_attempt
                else None
            ),
            "failure_reason": (
                None
                if successful_attempt
                else last_attempt.get("failure_reason", "not_attempted")
            ),
        })
    return summaries


def run_all_objects(args):
    if not args.scene_config:
        raise ValueError(
            "--all-objects/--continuous-grasp requires --scene-config"
        )
    if args.max_candidates_per_object <= 0:
        raise ValueError("--max-candidates-per-object must be greater than zero")

    continuous_mode = bool(args.continuous_grasp)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[All Objects] seed={args.seed}, device={device}")
    network = _load_network(args, device)

    config = load_scene_config(args.scene_config)
    continuous_config = config.get("continuous_grasp", {})
    release_after_place = bool(args.drop_after_grasp or continuous_mode)
    release_settle_steps = (
        args.drop_settle_steps
        if args.drop_settle_steps is not None
        else int(continuous_config.get("drop_settle_steps", 180))
    )
    max_stalled_passes = (
        args.max_stalled_passes
        if args.max_stalled_passes is not None
        else int(continuous_config.get("max_stalled_passes", 2))
    )
    if release_settle_steps < 0:
        raise ValueError("--drop-settle-steps must be non-negative")
    if max_stalled_passes <= 0:
        raise ValueError("--max-stalled-passes must be greater than zero")

    scene = SimulationScene(gui=args.gui)
    scene.connect()
    scene.load_plane()
    scene.load_objects(config["_resolved_objects"])
    staging_enabled = bool(
        config.get("object_staging", {}).get(
            "lock_initial_poses_until_grasp",
            False,
        )
    )
    if staging_enabled:
        scene.stage_objects_at_initial_poses()
        print(
            "  初始堆叠已暂时锁定；每个目标进入抓取评估前恢复动态质量"
        )
    for _ in range(int(config.get("settle_steps", 300))):
        scene.step()

    capture_pose = config.get("capture_joint_pose_deg")
    place_pose = config.get("place_target_joint_pose_deg")
    if release_after_place and place_pose is None:
        raise ValueError(
            "Drop mode requires scene config place_target_joint_pose_deg"
        )
    gripper = create_gripper(
        args.gripper_model,
        planner=None,
        initial_joint_pose_deg=capture_pose,
        robot_base_yaw_deg=float(config.get("robot_base_yaw_deg", 0.0)),
        gui_motion_step_delay=(0.003 / args.gui_speed) if args.gui else 0.0,
    )
    gripper.load()
    video_recorder = None
    if args.record_video:
        from simulation.video_recorder import PyBulletVideoRecorder
        video_path = args.video_output or args.output.replace(".json", "_pybullet.mp4")
        video_recorder = PyBulletVideoRecorder(video_path)
        video_recorder.start()
        atexit.register(video_recorder.close)
        print(f"  PyBullet GUI 录制: {video_path}")
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

    target_ids, target_names = _ordered_target_ids(
        scene, args, config
    )
    print(f"  处理顺序: {' -> '.join(target_names)}")
    if release_after_place:
        print(
            "  投放策略: 到达 place_target_joint_pose_deg 后松爪，"
            f"等待 {release_settle_steps} 个仿真步"
        )

    last_visualization = None
    attempts = []
    if continuous_mode:
        remaining_ids = list(target_ids)
        stalled_passes = 0
        pass_index = 0
        while remaining_ids and stalled_passes < max_stalled_passes:
            pass_index += 1
            pass_succeeded = False
            print(
                f"\n[Continuous pass {pass_index}] "
                f"剩余 {len(remaining_ids)}/{len(target_ids)} 个物体"
            )
            for target_body_id in list(remaining_ids):
                target = scene.get_object_info(target_body_id)
                print(
                    f"\n  尝试目标: {target.name} "
                    f"(pass={pass_index}, attempt={len(attempts) + 1})"
                )
                attempt, visualization = _attempt_target(
                    scene=scene,
                    gripper=gripper,
                    camera=camera,
                    network=network,
                    device=device,
                    config=config,
                    args=args,
                    target_body_id=target_body_id,
                    capture_pose=capture_pose,
                    place_pose=place_pose,
                    release_after_place=True,
                    release_settle_steps=release_settle_steps,
                )
                attempt["pass_index"] = pass_index
                attempt["attempt_index"] = len(attempts) + 1
                attempts.append(attempt)
                last_visualization = visualization
                if attempt["success"]:
                    remaining_ids.remove(target_body_id)
                    pass_succeeded = True
                    # The pile changed after every successful drop. Start a new
                    # pass so all following candidates use a fresh camera frame.
                    break

            if pass_succeeded:
                stalled_passes = 0
            else:
                stalled_passes += 1
                print(
                    "  本轮无物体抓取成功；"
                    f"连续停滞 {stalled_passes}/{max_stalled_passes} 轮"
                )
        object_results = _summarize_continuous_objects(
            scene, target_ids, attempts
        )
    else:
        object_results = []
        for object_number, target_body_id in enumerate(
            target_ids,
            start=1,
        ):
            target = scene.get_object_info(target_body_id)
            print(
                f"\n[{object_number}/{len(target_ids)}] "
                f"目标物体: {target.name}"
            )
            attempt, visualization = _attempt_target(
                scene=scene,
                gripper=gripper,
                camera=camera,
                network=network,
                device=device,
                config=config,
                args=args,
                target_body_id=target_body_id,
                capture_pose=capture_pose,
                place_pose=place_pose,
                release_after_place=release_after_place,
                release_settle_steps=release_settle_steps,
            )
            attempt["attempt_index"] = object_number
            object_results.append(attempt)
            last_visualization = visualization

    successful_objects = sum(item["success"] for item in object_results)
    output = _json_safe({
        "mode": (
            "continuous_grasp_and_drop"
            if continuous_mode
            else "all_objects_sequential_pick_and_place"
        ),
        "scene_config": config["_path"],
        "seed": int(args.seed),
        "capture_joint_pose_deg": capture_pose,
        "robot_base_yaw_deg": float(config.get("robot_base_yaw_deg", 0.0)),
        "initial_pose_hold_seconds": float(args.initial_pose_hold_seconds),
        "max_candidates_per_object": int(args.max_candidates_per_object),
        "place_target_joint_pose_deg": place_pose,
        "drop_after_grasp": release_after_place,
        "object_staging_enabled": staging_enabled,
        "drop_settle_steps": release_settle_steps,
        "max_stalled_passes": (
            max_stalled_passes if continuous_mode else None
        ),
        "gui_speed": float(args.gui_speed),
        "assisted_grasp": bool(args.assisted_grasp),
        "gripper_model": args.gripper_model,
        "object_total": len(object_results),
        "object_success": successful_objects,
        "objects": object_results,
        "attempts": attempts if continuous_mode else [],
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

    print(
        f"\n全部物体结果: "
        f"{successful_objects}/{len(object_results)} 成功"
    )
    print(f"结果已保存: {args.output}")
    if video_recorder is not None:
        video_recorder.close()
        atexit.unregister(video_recorder.close)
        print(f"PyBullet GUI 视频已保存: {video_recorder.output_path}")
    gripper.remove()
    scene.disconnect()
    return output
