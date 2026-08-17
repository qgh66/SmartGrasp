#!/usr/bin/env python
"""Sequential and continuous multi-object grasp workflows."""

import json
import os
import pickle
import random
import re
import sys
import time
import atexit
from pathlib import Path

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
from simulation.capture_artifacts import export_camera_frame
from simulation.evaluator import GraspEvaluator
from simulation.gripper_factory import create_gripper
from simulation.object_mapping import match_scene_object_by_mask
from simulation.scene import SimulationScene


def _pipeline_helpers():
    """Import task-loop helpers lazily to keep the legacy batch path light."""
    from demo_task_closed_loop import (
        _capture_and_reason,
        _configured_occluders,
        _execute_grasp,
        _execute_push,
        _map_reason_object,
        _map_reason_target,
        _round_instruction,
    )

    return {
        "capture_and_reason": _capture_and_reason,
        "configured_occluders": _configured_occluders,
        "execute_grasp": _execute_grasp,
        "execute_push": _execute_push,
        "map_reason_object": _map_reason_object,
        "map_reason_target": _map_reason_target,
        "round_instruction": _round_instruction,
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _capture_directory(capture_root, capture_index, target_name):
    safe_target_name = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(target_name)
    ).strip("._")
    return capture_root / f"capture_{int(capture_index):04d}_{safe_target_name}"


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


def _format_pipeline_instruction(template, target):
    """Build one unambiguous instruction for the current batch target."""
    aliases = target.metadata.get("instruction_aliases") or ()
    target_label = str(
        target.metadata.get("batch_instruction_target")
        or (aliases[0] if aliases else target.name.replace("_", " "))
    ).strip()
    try:
        instruction = str(template).format(
            target=target_label,
            target_name=target.name,
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Invalid batch Pipeline --instruction template; only "
            "{target} and {target_name} are supported"
        ) from error
    if not instruction.strip():
        raise ValueError("Batch Pipeline instruction resolved to an empty string")
    return instruction.strip()


def _evaluate_perception_target(
    scene,
    capture,
    target_body_id,
    minimum_iou,
):
    """Check whether any Perception object mask matches the simulator target."""
    graph_path = Path(capture["reason_target"]["occlusion_graph_path"])
    with graph_path.open("r", encoding="utf-8") as graph_file:
        graph_data = json.load(graph_file)
    nodes = graph_data.get("graph", {}).get("nodes", [])
    matching_nodes = []
    mask_errors = []
    for node in nodes:
        mask_path = node.get("mask_path")
        if not mask_path:
            continue
        resolved_mask_path = Path(mask_path).expanduser()
        if not resolved_mask_path.is_absolute():
            resolved_mask_path = graph_path.parent / resolved_mask_path
        try:
            body_id, scene_object, selection = match_scene_object_by_mask(
                scene,
                capture["segmentation"],
                resolved_mask_path,
                minimum_iou=minimum_iou,
            )
        except Exception as error:
            mask_errors.append({
                "object_id": node.get("object_id"),
                "label": node.get("label"),
                "error": str(error),
            })
            continue
        if body_id != target_body_id:
            continue
        matching_nodes.append({
            "object_id": node.get("object_id"),
            "label": node.get("label"),
            "mask_path": str(resolved_mask_path.resolve()),
            "selected_body_id": int(body_id),
            "selected_object_name": scene_object.name,
            "selected_iou": float(selection["selected_iou"]),
            "mask_pixels": int(selection["mask_pixels"]),
        })

    matching_nodes.sort(
        key=lambda item: item["selected_iou"],
        reverse=True,
    )
    return {
        "correct": bool(matching_nodes),
        "expected_body_id": int(target_body_id),
        "expected_object_name": scene.get_object_info(target_body_id).name,
        "minimum_iou": float(minimum_iou),
        "perception_object_count": len(nodes),
        "matching_mask_count": len(matching_nodes),
        "best_matching_mask": matching_nodes[0] if matching_nodes else None,
        "mask_error_count": len(mask_errors),
        "mask_errors": mask_errors,
    }


def _build_reason_validation(
    target_body_id,
    target_mapping,
    action_mapping,
):
    """Check both Reason's semantic target and proposed grasp object."""
    semantic_target_body_id = (
        int(target_mapping[0]) if target_mapping is not None else None
    )
    action_object_body_id = (
        int(action_mapping[0]) if action_mapping is not None else None
    )
    semantic_target_correct = semantic_target_body_id == int(target_body_id)
    action_object_correct = action_object_body_id == int(target_body_id)
    return {
        "correct": semantic_target_correct and action_object_correct,
        "expected_body_id": int(target_body_id),
        "semantic_target_correct": semantic_target_correct,
        "semantic_target_body_id": semantic_target_body_id,
        "semantic_target_object_name": (
            target_mapping[1].name if target_mapping is not None else None
        ),
        "action_object_correct": action_object_correct,
        "action_object_body_id": action_object_body_id,
        "action_object_name": (
            action_mapping[1].name if action_mapping is not None else None
        ),
    }


def _attempt_pipeline_target(
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
    release_settle_steps,
    completed_body_ids,
    batch_target_body_ids,
):
    """Run Perception/Reason before every action for one batch target.

    Explicit scene occlusion relations are used only as a physical fallback
    when Reason cannot see the requested target. The occluder is pushed, never
    silently substituted as the requested grasp target.
    """
    helpers = _pipeline_helpers()
    target = scene.get_object_info(target_body_id)
    instruction = _format_pipeline_instruction(args.instruction, target)
    configured_occluders = set(
        helpers["configured_occluders"](scene, target.name)
    )
    pending_occluders = configured_occluders.difference(completed_body_ids)
    pipeline_rounds = []
    latest_visualization = None
    latest_validation = None

    print(f"  Pipeline 指令: {instruction}")
    if pending_occluders:
        pending_names = [
            scene.get_object_info(body_id).name
            for body_id in sorted(pending_occluders)
        ]
        print(f"  配置遮挡物: {pending_names}")

    for round_index in range(1, args.max_task_rounds + 1):
        round_instruction = helpers["round_instruction"](
            instruction,
            round_index,
        )
        try:
            capture = helpers["capture_and_reason"](
                camera=camera,
                gripper=gripper,
                capture_pose=capture_pose,
                instruction=round_instruction,
                requested_scene_id=None,
                network=network,
                device=device,
                allow_unselected_object=True,
                prepare_gripper=not args.perception_reason_test,
            )
            reason_target = capture["reason_target"]

            reason_target_mapping = None
            target_mapping = None
            try:
                reason_target_mapping = helpers["map_reason_target"](
                    scene,
                    capture,
                    args.target_mask_min_iou,
                )
                if (
                    reason_target_mapping is not None
                    and reason_target_mapping[0] == target_body_id
                ):
                    target_mapping = reason_target_mapping
            except Exception as error:
                print(f"  ⚠️ Reason 目标 mask 映射失败: {error}")

            action_mapping = None
            if reason_target.get("object_id") is not None:
                try:
                    action_mapping = helpers["map_reason_object"](
                        scene,
                        capture,
                        args.target_mask_min_iou,
                        prefer_part_mask=args.use_reason_part_mask,
                    )
                except Exception as error:
                    print(f"  ⚠️ Reason 动作 mask 映射失败: {error}")

            if args.perception_reason_test:
                perception_validation = _evaluate_perception_target(
                    scene,
                    capture,
                    target_body_id,
                    args.target_mask_min_iou,
                )
                reason_validation = _build_reason_validation(
                    target_body_id,
                    reason_target_mapping,
                    action_mapping,
                )
                validation_passed = bool(
                    perception_validation["correct"]
                    and reason_validation["correct"]
                )
                removed_from_scene = False
                if validation_passed:
                    scene.remove_object(target_body_id)
                    removed_from_scene = True
                    print(
                        "  ✅ Perception/Reason 核验正确，已从场景删除: "
                        f"{target.name} (body_id={target_body_id})"
                    )
                else:
                    print(
                        "  ❌ Perception/Reason 核验未通过，不删除物体: "
                        f"perception={perception_validation['correct']}, "
                        f"reason={reason_validation['correct']}"
                    )

                latest_validation = {
                    "passed": validation_passed,
                    "perception": perception_validation,
                    "reason": reason_validation,
                    "removed_from_scene": removed_from_scene,
                }
                selected_mapping = action_mapping or reason_target_mapping
                selection = (
                    selected_mapping[2] if selected_mapping is not None else None
                )
                round_record = {
                    "round": round_index,
                    "scene_id": int(capture["scene_id"]),
                    "instruction": round_instruction,
                    "reason_target": reason_target,
                    "selected_body_id": (
                        int(selected_mapping[0])
                        if selected_mapping is not None
                        else None
                    ),
                    "selected_object_name": (
                        selected_mapping[1].name
                        if selected_mapping is not None
                        else None
                    ),
                    "selection_role": "validation_only",
                    "selection": selection,
                    "action": (
                        "delete-validated-target"
                        if validation_passed
                        else "no-action-validation-failed"
                    ),
                    "action_result": latest_validation,
                    "action_success": validation_passed,
                }
                pipeline_rounds.append(round_record)
                latest_visualization = {
                    "rgb": capture["rgb"],
                    "depth": capture["depth"],
                    "seg": capture["segmentation"],
                    "point_cloud": camera.generate_point_cloud(
                        capture["depth"],
                        num_points=20000,
                    ).numpy(),
                    "perception_input": capture["perception_input"],
                    "reason_target": reason_target,
                    "target_selection": selection,
                    "target_body_id": int(target_body_id),
                    "target_object_name": target.name,
                    "perception_reason_validation": latest_validation,
                }
                if validation_passed:
                    return {
                        "target_body_id": int(target_body_id),
                        "target_object_name": target.name,
                        "obj_path": target.path,
                        "instruction": instruction,
                        "success": True,
                        "failure_reason": None,
                        "evaluated_candidates": 0,
                        "perception_correct": True,
                        "reason_correct": True,
                        "removed_from_scene": True,
                        "pipeline_rounds": pipeline_rounds,
                    }, latest_visualization
                scene.step(args.reobserve_settle_steps)
                continue

            # Prefer Reason's action-object mapping when it resolves to the
            # requested target. This preserves the validated grasp part mask
            # for execution. The semantic target mask is only a whole-object
            # fallback when the action mask cannot be mapped reliably.
            if (
                action_mapping is not None
                and action_mapping[0] == target_body_id
            ):
                selected_body_id, selected_object, selection = action_mapping
                selection_role = "grasp_object"
            elif target_mapping is not None:
                selected_body_id, selected_object, selection = target_mapping
                selection_role = "target_object"
            elif pending_occluders:
                unfinished_target_occluders = pending_occluders.intersection(
                    batch_target_body_ids
                )
                if unfinished_target_occluders:
                    names = [
                        scene.get_object_info(body_id).name
                        for body_id in sorted(unfinished_target_occluders)
                    ]
                    raise RuntimeError(
                        "Configured occluder is also an unfinished batch "
                        f"target and must be grasped first: {names}"
                    )
                selected_body_id = (
                    action_mapping[0]
                    if (
                        action_mapping is not None
                        and action_mapping[0] in pending_occluders
                    )
                    else min(pending_occluders)
                )
                selected_object = scene.get_object_info(selected_body_id)
                selection = {
                    "source": "configured_occlusion_relation",
                    "reason_selection": (
                        action_mapping[2]
                        if action_mapping is not None
                        else None
                    ),
                }
                selection_role = "occluder"
            else:
                mapped_name = (
                    action_mapping[1].name
                    if action_mapping is not None
                    else None
                )
                raise RuntimeError(
                    "Reason did not map the requested target to its PyBullet "
                    f"body: requested={target.name!r}, mapped={mapped_name!r}"
                )

            is_target_action = selected_body_id == target_body_id
            action = "grasp-target" if is_target_action else "push"
            print(
                "  🧭 批量 Pipeline 动作: "
                f"round={round_index}, action={action}, "
                f"name={selected_object.name}, role={selection_role}"
            )

            if is_target_action:
                action_result = helpers["execute_grasp"](
                    scene=scene,
                    gripper=gripper,
                    camera=camera,
                    capture=capture,
                    network=network,
                    device=device,
                    body_id=selected_body_id,
                    reason_target=reason_target,
                    config=config,
                    args=args,
                    final_target=False,
                    use_reason_part_mask=bool(
                        args.use_reason_part_mask
                        and selection_role == "grasp_object"
                    ),
                    release_settle_steps=release_settle_steps,
                )
            else:
                push_result, activated = helpers["execute_push"](
                    scene=scene,
                    gripper=gripper,
                    body_id=selected_body_id,
                    config=config,
                    requested_direction=args.push_direction,
                    move_distance=args.push_distance,
                    gui=args.gui,
                    gui_speed=args.gui_speed,
                )
                action_result = {
                    "success": bool(push_result["success"]),
                    "activated_from_staging": activated,
                    "push": push_result,
                }

            round_record = {
                "round": round_index,
                "scene_id": int(capture["scene_id"]),
                "instruction": round_instruction,
                "reason_target": reason_target,
                "selected_body_id": int(selected_body_id),
                "selected_object_name": selected_object.name,
                "selection_role": selection_role,
                "selection": selection,
                "action": action,
                "action_result": action_result,
                "action_success": bool(action_result["success"]),
            }
            pipeline_rounds.append(round_record)
            latest_visualization = {
                "rgb": capture["rgb"],
                "depth": capture["depth"],
                "seg": capture["segmentation"],
                "point_cloud": action_result.get("point_cloud"),
                "perception_input": capture["perception_input"],
                "reason_target": reason_target,
                "target_selection": selection,
                "target_body_id": int(target_body_id),
                "target_object_name": target.name,
            }

            if is_target_action:
                return {
                    "target_body_id": int(target_body_id),
                    "target_object_name": target.name,
                    "obj_path": target.path,
                    "instruction": instruction,
                    "success": bool(action_result["success"]),
                    "failure_reason": (
                        None
                        if action_result["success"]
                        else "target_grasp_failed"
                    ),
                    "evaluated_candidates": len(
                        action_result.get("grasps") or []
                    ),
                    "pipeline_rounds": pipeline_rounds,
                }, latest_visualization

            if not action_result["success"]:
                print(
                    "  ⚠️ 遮挡物 Push 未成功；下一轮仍会重新感知后重试"
                )
            scene.step(args.reobserve_settle_steps)
        except Exception as error:
            print(f"  ❌ 批量 Pipeline 目标失败: {error}")
            return {
                "target_body_id": int(target_body_id),
                "target_object_name": target.name,
                "obj_path": target.path,
                "instruction": instruction,
                "success": False,
                "failure_reason": f"pipeline_failed: {error}",
                "evaluated_candidates": 0,
                "pipeline_rounds": pipeline_rounds,
                "perception_correct": bool(
                    latest_validation
                    and latest_validation["perception"]["correct"]
                ),
                "reason_correct": bool(
                    latest_validation
                    and latest_validation["reason"]["correct"]
                ),
                "removed_from_scene": False,
                "perception_reason_validation": latest_validation,
            }, latest_visualization

    return {
        "target_body_id": int(target_body_id),
        "target_object_name": target.name,
        "obj_path": target.path,
        "instruction": instruction,
        "success": False,
        "failure_reason": "max_task_rounds_reached",
        "evaluated_candidates": 0,
        "pipeline_rounds": pipeline_rounds,
        "perception_correct": bool(
            latest_validation
            and latest_validation["perception"]["correct"]
        ),
        "reason_correct": bool(
            latest_validation
            and latest_validation["reason"]["correct"]
        ),
        "removed_from_scene": False,
        "perception_reason_validation": latest_validation,
    }, latest_visualization


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
    capture_output_dir,
):
    """Capture the current scene and execute one target's best candidates."""
    target = scene.get_object_info(target_body_id)
    gripper.release_grasp()
    gripper.set_opening(gripper._max_opening)
    gripper.move_to_joint_pose_deg(capture_pose)

    rgb, depth, segmentation, point_cloud, object_clouds, target_points = (
        _capture_target(scene, camera, target_body_id, config)
    )
    camera_artifacts = export_camera_frame(
        output_dir=capture_output_dir,
        rgb=rgb,
        depth=depth,
        segmentation=segmentation,
        object_names_by_id={
            int(body_id): scene.get_object_info(body_id).name
            for body_id in scene.object_ids
        },
        target_body_id=target_body_id,
    )
    print(f"  相机数据已保存: {capture_output_dir}")
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
        "camera_artifacts": camera_artifacts,
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
            "camera_artifacts": camera_artifacts,
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
            "camera_artifacts": camera_artifacts,
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
        "camera_artifacts": camera_artifacts,
        **filter_stats,
    }, visualization


def _summarize_continuous_objects(target_ids, target_names, attempts):
    """Build one compact final record per requested object."""
    summaries = []
    for target_body_id, target_name in zip(target_ids, target_names):
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
        summary = {
            "target_body_id": int(target_body_id),
            "target_object_name": target_name,
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
        }
        validation_attempt = successful_attempt or last_attempt
        if "perception_correct" in validation_attempt:
            summary.update({
                "perception_correct": bool(
                    validation_attempt.get("perception_correct", False)
                ),
                "reason_correct": bool(
                    validation_attempt.get("reason_correct", False)
                ),
                "removed_from_scene": bool(
                    validation_attempt.get("removed_from_scene", False)
                ),
            })
        summaries.append(summary)
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
    validation_only = bool(args.perception_reason_test)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[All Objects] seed={args.seed}, device={device}")
    network = None if validation_only else _load_network(args, device)
    if validation_only:
        print(
            "  测试模式: 仅运行 Perception + Intent + Reason；"
            "不加载 GraspNet，不执行机械臂抓取或 Push"
        )

    config = load_scene_config(args.scene_config)
    continuous_config = config.get("continuous_grasp", {})
    release_after_place = bool(
        not validation_only
        and (args.drop_after_grasp or continuous_mode)
    )
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
    print(f"  场景已倒入并稳定: {len(scene.object_ids)} 个物体")
    if validation_only:
        print("  机械臂保持初始拍摄位姿，测试过程中不发送运动指令")
    else:
        gripper.move_to_joint_pose_deg(capture_pose)
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
    if args.run_pipeline_after_capture:
        print(
            "  感知策略: 每个目标、每次重试均重新运行 "
            "Perception + Intent + Reason"
        )
    if release_after_place:
        print(
            "  投放策略: 到达 place_target_joint_pose_deg 后松爪，"
            f"等待 {release_settle_steps} 个仿真步"
        )

    last_visualization = None
    attempts = []
    output_path = Path(args.output)
    capture_root = output_path.with_name(f"{output_path.stem}_captures")
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
                if args.run_pipeline_after_capture:
                    attempt, visualization = _attempt_pipeline_target(
                        scene=scene,
                        gripper=gripper,
                        camera=camera,
                        network=network,
                        device=device,
                        config=config,
                        args=args,
                        target_body_id=target_body_id,
                        capture_pose=capture_pose,
                        release_settle_steps=release_settle_steps,
                        completed_body_ids=set(target_ids).difference(
                            remaining_ids
                        ),
                        batch_target_body_ids=set(target_ids),
                    )
                else:
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
                        capture_output_dir=_capture_directory(
                            capture_root,
                            len(attempts) + 1,
                            target.name,
                        ),
                    )
                attempt["pass_index"] = pass_index
                attempt["attempt_index"] = len(attempts) + 1
                attempts.append(attempt)
                last_visualization = visualization
                if attempt["success"]:
                    remaining_ids.remove(target_body_id)
                    pass_succeeded = True
                    # A successful drop or validation deletion changes the
                    # scene, so following targets need a fresh camera frame.
                    break

            if pass_succeeded:
                stalled_passes = 0
            else:
                stalled_passes += 1
                print(
                    "  本轮无物体处理成功；"
                    f"连续停滞 {stalled_passes}/{max_stalled_passes} 轮"
                )
        object_results = _summarize_continuous_objects(
            target_ids, target_names, attempts
        )
    else:
        object_results = []
        completed_body_ids = set()
        for object_number, target_body_id in enumerate(
            target_ids,
            start=1,
        ):
            target = scene.get_object_info(target_body_id)
            print(
                f"\n[{object_number}/{len(target_ids)}] "
                f"目标物体: {target.name}"
            )
            if args.run_pipeline_after_capture:
                attempt, visualization = _attempt_pipeline_target(
                    scene=scene,
                    gripper=gripper,
                    camera=camera,
                    network=network,
                    device=device,
                    config=config,
                    args=args,
                    target_body_id=target_body_id,
                    capture_pose=capture_pose,
                    release_settle_steps=release_settle_steps,
                    completed_body_ids=completed_body_ids,
                    batch_target_body_ids=set(target_ids),
                )
            else:
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
                    capture_output_dir=_capture_directory(
                        capture_root,
                        object_number,
                        target.name,
                    ),
                )
            attempt["attempt_index"] = object_number
            object_results.append(attempt)
            last_visualization = visualization
            if attempt["success"]:
                completed_body_ids.add(target_body_id)

    successful_objects = sum(item["success"] for item in object_results)
    failed_objects = len(object_results) - successful_objects
    success_rate = (
        successful_objects / len(object_results)
        if object_results
        else 0.0
    )
    successful_object_names = [
        item["target_object_name"]
        for item in object_results
        if item["success"]
    ]
    failed_object_names = [
        item["target_object_name"]
        for item in object_results
        if not item["success"]
    ]
    output = _json_safe({
        "mode": (
            "perception_reason_validation_and_delete"
            if validation_only
            else (
                "pipeline_continuous_grasp_and_drop"
                if continuous_mode and args.run_pipeline_after_capture
                else (
                    "continuous_grasp_and_drop"
                    if continuous_mode
                    else (
                        "pipeline_all_objects_sequential_pick_and_place"
                        if args.run_pipeline_after_capture
                        else "all_objects_sequential_pick_and_place"
                    )
                )
            )
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
        "run_pipeline_after_capture": bool(
            args.run_pipeline_after_capture
        ),
        "perception_reason_test": validation_only,
        "graspnet_enabled": network is not None,
        "physical_actions_enabled": not validation_only,
        "instruction_template": (
            str(args.instruction)
            if args.run_pipeline_after_capture
            else None
        ),
        "gripper_model": args.gripper_model,
        "object_total": len(object_results),
        "object_success": successful_objects,
        "object_failed": failed_objects,
        "success_rate": success_rate,
        "experiment_summary": {
            "total": len(object_results),
            "success": successful_objects,
            "failed": failed_objects,
            "success_rate": success_rate,
            "success_rate_percent": success_rate * 100.0,
            "successful_objects": successful_object_names,
            "failed_objects": failed_object_names,
        },
        "objects": object_results,
        "attempts": attempts if continuous_mode else [],
        "capture_root": str(capture_root),
        "capture_count": len(attempts) if continuous_mode else len(object_results),
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

    print("\n实验结果汇总:")
    print(
        f"  成功率: {success_rate:.2%} "
        f"({successful_objects}/{len(object_results)})"
    )
    print(f"  成功: {successful_objects} 个")
    print(f"  失败: {failed_objects} 个")
    print(
        "  成功物体: "
        + (", ".join(successful_object_names) or "无")
    )
    print(
        "  失败物体: "
        + (", ".join(failed_object_names) or "无")
    )
    print(f"结果已保存: {args.output}")
    if video_recorder is not None:
        video_recorder.close()
        atexit.unregister(video_recorder.close)
        print(f"PyBullet GUI 视频已保存: {video_recorder.output_path}")
    gripper.remove()
    scene.disconnect()
    return output
