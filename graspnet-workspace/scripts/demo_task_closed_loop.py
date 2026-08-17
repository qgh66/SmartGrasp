#!/usr/bin/env python
"""Physical perception-reason-action loop for one semantic target."""

from __future__ import annotations

import atexit
import json
import os
import pickle
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch

from demo_closed_loop import (
    crop_to_object,
    filter_collision_grasps,
    filter_grasps_to_object,
    load_graspnet_model,
    load_scene_config,
    point_cloud_from_reason_part_mask,
    prefer_topdown_grasps,
)
from graspnetAPI import GraspGroup
from models.graspnet import pred_decode
from simulation.camera import VirtualCamera
from simulation.evaluator import GraspEvaluator
from simulation.gripper_factory import create_gripper
from simulation.object_mapping import match_scene_object_by_mask
from simulation.perception_input import (
    export_perception_input,
    generate_capture_scene_id,
    run_pipeline_for_scene,
)
from simulation.reveal_push import RevealPushExecutor, object_center_from_aabb
from simulation.scene import SimulationScene


FINAL_BRANCH = "fully_visible"
PARTIAL_BRANCH = "partially_occluded"
PIPELINE_HEARTBEAT_SECONDS = 30.0


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
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


def _make_camera(config):
    camera_config = config.get("camera", {})
    return VirtualCamera(
        position=tuple(camera_config.get("position", (0.3, 0.0, 0.5))),
        target=tuple(camera_config.get("target", (0.3, 0.0, 0.05))),
        near=float(camera_config.get("near", 0.01)),
        far=float(camera_config.get("far", 5.0)),
        width=int(camera_config.get("width", 1280)),
        height=int(camera_config.get("height", 720)),
        fov=float(camera_config.get("fov", 60.0)),
    )


def _move_network_off_gpu(network, device):
    if network is None or device.type != "cuda":
        return
    network.to("cpu")
    torch.cuda.empty_cache()


def _move_network_to_device(network, device):
    if network is not None:
        network.to(device)


def _report_pipeline_wait(
    stop_event,
    *,
    scene_id,
    started_at,
):
    """Print a heartbeat while the synchronous VLM pipeline is running."""
    while not stop_event.wait(PIPELINE_HEARTBEAT_SECONDS):
        elapsed = time.monotonic() - started_at
        print(
            "  ⏳ Perception + Reason 仍在运行: "
            f"scene_id={scene_id}, elapsed={elapsed:.0f}s；"
            "机械臂保持拍摄位姿",
            flush=True,
        )


def _capture_and_reason(
    *,
    camera,
    gripper,
    capture_pose,
    instruction,
    requested_scene_id,
    network,
    device,
    allow_unselected_object=False,
    prepare_gripper=True,
):
    """Capture one physical state and run the full perception/reason pipeline."""
    if prepare_gripper:
        gripper.release_grasp()
        gripper.set_opening(gripper._max_opening)
        gripper.move_to_joint_pose_deg(capture_pose)
    rgb, depth, segmentation = camera.capture()

    scene_id = (
        int(requested_scene_id)
        if requested_scene_id is not None
        else generate_capture_scene_id()
    )
    perception_input = export_perception_input(
        scene_id=scene_id,
        rgb=rgb,
        depth=depth,
        segmentation=segmentation,
        instruction=instruction,
    )
    print(
        f"  💾 Round 输入: scene_id={scene_id} "
        f"-> {perception_input['input_dir']}"
    )

    # SAM2 and GraspNet are both GPU-heavy. Keep only the model needed by the
    # current phase on CUDA.
    _move_network_off_gpu(network, device)
    pipeline_started_at = time.monotonic()
    print(
        "  ⏳ 开始运行 Perception + Intent + Reason；"
        "完成前机械臂会保持拍摄位姿",
        flush=True,
    )
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_report_pipeline_wait,
        kwargs={
            "stop_event": heartbeat_stop,
            "scene_id": scene_id,
            "started_at": pipeline_started_at,
        },
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        reason_target = run_pipeline_for_scene(
            scene_id,
            allow_unselected_object=allow_unselected_object,
        )
    except Exception:
        elapsed = time.monotonic() - pipeline_started_at
        print(
            f"  ❌ Pipeline 失败: scene_id={scene_id}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        _move_network_to_device(network, device)
    elapsed = time.monotonic() - pipeline_started_at
    print(
        "  ✅ Round 推理: "
        f"elapsed={elapsed:.1f}s, "
        f"branch={reason_target['branch']}, "
        f"target={reason_target.get('target_object_label')}, "
        f"grasp_object={reason_target['object_label']}",
        flush=True,
    )
    return {
        "scene_id": scene_id,
        "rgb": rgb,
        "depth": depth,
        "segmentation": segmentation,
        "perception_input": perception_input,
        "reason_target": reason_target,
    }


def _map_reason_object(
    scene,
    capture,
    minimum_iou,
    *,
    prefer_part_mask,
):
    reason_target = capture["reason_target"]
    whole_body_id, whole_scene_object, whole_selection = (
        match_scene_object_by_mask(
            scene,
            capture["segmentation"],
            reason_target["object_mask_path"],
            minimum_iou=minimum_iou,
        )
    )
    body_id = whole_body_id
    scene_object = whole_scene_object
    selection = whole_selection
    part_mask = reason_target.get("grasp_part_mask") or {}
    part_mask_path = reason_target.get("grasp_part_mask_path")
    if (
        prefer_part_mask
        and part_mask_path
        and bool(part_mask.get("validated"))
    ):
        try:
            body_id, scene_object, part_selection = (
                match_scene_object_by_mask(
                    scene,
                    capture["segmentation"],
                    part_mask_path,
                    minimum_iou=minimum_iou,
                )
            )
            selection = {
                **part_selection,
                "source": "reason_part_mask_iou",
                "reason_part_id": part_mask.get("part_id"),
                "whole_object_selection": whole_selection,
            }
            print(
                "  🧩 合并物体消歧: "
                f"whole_mask->{whole_scene_object.name}, "
                f"part_mask->{scene_object.name}"
            )
        except Exception as error:
            print(
                "  ⚠️ Part mask 映射失败，回退整物体 mask: "
                f"{error}"
            )
    selection.update(
        {
            "reason_scene_id": int(reason_target["scene_id"]),
            "reason_branch": reason_target["branch"],
            "reason_object_id": int(reason_target["object_id"]),
            "reason_object_label": reason_target["object_label"],
            "reason_summary_path": reason_target["reason_summary_path"],
            "occlusion_graph_path": reason_target["occlusion_graph_path"],
        }
    )
    print(
        f"  🔗 Reason Object {reason_target['object_id']} "
        f"-> body_id={body_id}, name={scene_object.name}, "
        f"IoU={selection['selected_iou']:.4f}, "
        f"source={selection['source']}"
    )
    return body_id, scene_object, selection


def _map_reason_target(scene, capture, minimum_iou):
    """Map Reason's semantic target mask independently of its action object."""
    reason_target = capture["reason_target"]
    target_mask_path = reason_target.get("target_object_mask_path")
    if not target_mask_path:
        return None
    body_id, scene_object, selection = match_scene_object_by_mask(
        scene,
        capture["segmentation"],
        target_mask_path,
        minimum_iou=minimum_iou,
    )
    selection.update(
        {
            "reason_scene_id": int(reason_target["scene_id"]),
            "reason_branch": reason_target["branch"],
            "reason_target_object_id": int(
                reason_target["target_object_id"]
            ),
            "reason_target_object_label": reason_target[
                "target_object_label"
            ],
            "reason_summary_path": reason_target["reason_summary_path"],
            "occlusion_graph_path": reason_target[
                "occlusion_graph_path"
            ],
        }
    )
    print(
        f"  🎯 Reason Target {reason_target['target_object_id']} "
        f"-> body_id={body_id}, name={scene_object.name}, "
        f"IoU={selection['selected_iou']:.4f}"
    )
    return body_id, scene_object, selection


def _configured_target(scene, instruction):
    """Resolve one stable simulation target from explicit scene aliases."""
    normalized_instruction = str(instruction).strip().lower()
    matches = []
    for body_id, scene_object in scene.get_object_registry().items():
        aliases = scene_object.metadata.get("instruction_aliases") or ()
        matching_aliases = [
            str(alias)
            for alias in aliases
            if str(alias).strip().lower() in normalized_instruction
        ]
        if matching_aliases:
            matches.append(
                (
                    max(len(alias) for alias in matching_aliases),
                    int(body_id),
                    scene_object,
                    matching_aliases,
                )
            )
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    best_length = matches[0][0]
    best_matches = [item for item in matches if item[0] == best_length]
    if len(best_matches) != 1:
        names = [item[2].name for item in best_matches]
        raise RuntimeError(
            "Instruction matches multiple configured final targets: "
            f"{names}"
        )
    _, body_id, scene_object, aliases = best_matches[0]
    print(
        f"🎯 配置目标锁定: body_id={body_id}, "
        f"name={scene_object.name}, aliases={aliases}"
    )
    return body_id, scene_object


def _configured_occluders(scene, target_name):
    """Return bodies explicitly declared to occlude the configured target."""
    occluders = []
    for body_id, scene_object in scene.get_object_registry().items():
        target_names = scene_object.metadata.get("occludes") or ()
        if target_name in target_names:
            occluders.append(int(body_id))
    return occluders


def _configured_occlusion_branch(target_object):
    occlusion_case = str(
        target_object.metadata.get("occlusion_case") or ""
    ).lower()
    if "fully" in occlusion_case:
        return "fully_occluded"
    if "partial" in occlusion_case:
        return PARTIAL_BRANCH
    return PARTIAL_BRANCH


def _configured_target_visibility(capture, body_id, target_object):
    """Return configured target visibility when the scene defines a threshold."""
    minimum_ratio = target_object.metadata.get("grasp_min_visible_ratio")
    if minimum_ratio is None:
        return None
    minimum_ratio = float(minimum_ratio)
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError(
            "grasp_min_visible_ratio must be in (0, 1], got "
            f"{minimum_ratio} for {target_object.name}"
        )

    segmentation = np.asarray(capture["segmentation"], dtype=np.int64)
    decoded_body_ids = np.where(
        segmentation >= 0,
        segmentation & ((1 << 24) - 1),
        -1,
    )
    visible_pixels = int(np.count_nonzero(decoded_body_ids == int(body_id)))
    total_pixels = int(segmentation.size)
    visible_ratio = visible_pixels / total_pixels if total_pixels else 0.0
    return {
        "source": "configured_target_visibility",
        "selected_body_id": int(body_id),
        "selected_object_name": target_object.name,
        "visible_pixels": visible_pixels,
        "total_pixels": total_pixels,
        "visible_ratio": visible_ratio,
        "minimum_visible_ratio": minimum_ratio,
        "visible_enough_for_grasp": visible_ratio >= minimum_ratio,
    }


def _round_instruction(instruction, round_index):
    """Describe the initial relation without freezing later visual state."""
    instruction = str(instruction).strip()
    if round_index <= 1:
        return instruction
    return (
        f"{instruction}\n"
        "这是执行遮挡物操作后的重新观察。指令中的遮挡描述仅指"
        "初始状态；请以当前图像为准，重新判断目标现在是否可见。"
    )


def _selection_override(
    *,
    reason_selection,
    selected_body_id,
    selected_object,
    source,
):
    return {
        "source": source,
        "selected_body_id": int(selected_body_id),
        "selected_object_name": selected_object.name,
        "reason_selection": reason_selection,
    }


def _resolve_action(
    policy,
    branch,
    body_id,
    failed_auto_pushes,
    *,
    final_target,
):
    """Choose the physical action for an intermediate occluder."""
    if final_target:
        return "grasp-target"
    if policy == "push":
        return "push"
    if policy == "grasp-away":
        return "grasp-away"
    if branch == PARTIAL_BRANCH and body_id not in failed_auto_pushes:
        return "push"
    return "grasp-away"


def _automatic_push_direction(scene, body_id, config):
    center = object_center_from_aabb(body_id)
    workspace_center = np.asarray(
        config.get("camera", {}).get("target", (0.3, 0.0, 0.05)),
        dtype=float,
    )
    direction = np.array(
        [
            center[0] - workspace_center[0],
            center[1] - workspace_center[1],
            0.0,
        ],
        dtype=float,
    )
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    return direction / np.linalg.norm(direction)


def _execute_push(
    *,
    scene,
    gripper,
    body_id,
    config,
    requested_direction,
    move_distance,
    gui,
    gui_speed,
):
    staged = scene.is_object_staged(body_id)
    activated = False

    def activate_for_contact():
        nonlocal activated
        if staged and not activated:
            activated = scene.activate_staged_object(body_id)
        return activated

    direction = (
        np.asarray(requested_direction, dtype=float)
        if requested_direction is not None
        else _automatic_push_direction(scene, body_id, config)
    )
    direction = direction / np.linalg.norm(direction)
    executor = RevealPushExecutor(
        body_id,
        gripper,
        gui=gui,
        gui_speed=gui_speed,
    )
    try:
        result = executor.execute_push(
            center_point=object_center_from_aabb(body_id),
            direction=direction,
            move_distance=move_distance,
            activate_callback=(activate_for_contact if staged else None),
        )
    except Exception:
        if activated:
            scene.restage_object(body_id)
        raise
    if activated:
        if result["success"]:
            scene.stage_object_at_current_pose(body_id)
        else:
            scene.restage_object(body_id)
    if result.get("unsafe_motion"):
        recovery_status = (
            "已恢复到上一次稳定位置"
            if activated
            else "物体保持在静态稳定位置"
        )
        print(
            f"  ⚠️ 检测到不安全 push，{recovery_status}: "
            f"{result.get('failure_reason')}"
        )
    return result, activated


def _grasp_region(capture, camera, body_id, reason_target, use_part_mask):
    segmentation = capture["segmentation"]
    depth = capture["depth"]
    object_clouds = camera.generate_object_point_clouds(
        depth,
        segmentation,
        [body_id],
    )
    target_points = object_clouds.get(int(body_id))
    if target_points is None or len(target_points) == 0:
        raise RuntimeError(
            f"Selected body_id={body_id} has no visible camera points"
        )

    if not use_part_mask:
        return target_points, "whole_object", None

    part_mask = reason_target.get("grasp_part_mask") or {}
    part_mask_path = reason_target.get("grasp_part_mask_path")
    try:
        part_object_id = int(part_mask.get("object_id"))
    except (TypeError, ValueError):
        part_object_id = None
    if (
        not part_mask_path
        or not bool(part_mask.get("validated"))
        or part_object_id != int(reason_target["object_id"])
    ):
        raise RuntimeError(
            "Reason did not produce a validated part mask for the selected "
            f"grasp object: object_id={reason_target['object_id']}"
        )

    region_points, diagnostics = point_cloud_from_reason_part_mask(
        camera,
        depth,
        segmentation,
        body_id,
        part_mask_path,
    )
    diagnostics["part_id"] = part_mask.get("part_id")
    diagnostics["object_id"] = part_object_id
    return region_points, "reason_part_mask", diagnostics


def _generate_grasps(
    *,
    network,
    device,
    camera,
    capture,
    region_points,
    region_source,
    config,
    args,
):
    point_cloud = camera.generate_point_cloud(
        capture["depth"],
        num_points=20000,
    ).numpy()
    crop_config = config.get("crop", {})
    point_cloud = crop_to_object(
        point_cloud,
        object_points=region_points,
        margin=float(crop_config.get("margin", 0.05)),
        num_points=int(crop_config.get("num_points", 20000)),
        table_z=float(crop_config.get("table_z", 0.005)),
    )
    camera_points = camera.world_to_camera_points(
        point_cloud[0]
    ).astype(np.float32)
    cloud_tensor = torch.from_numpy(camera_points[np.newaxis]).to(device)
    with torch.no_grad():
        predictions = pred_decode(network({"point_clouds": cloud_tensor}))
    grasps = GraspGroup(predictions[0].detach().cpu().numpy())
    grasps = camera.camera_grasps_to_world(grasps)
    grasps.sort_by_score()
    if len(grasps) == 0:
        raise RuntimeError("GraspNet produced no grasp candidates")

    raw_grasps = GraspGroup(grasps.grasp_group_array.copy())
    grasp_stats = {}
    collision_stats = {}
    topdown_stats = {}
    if not args.test_all_raw_candidates:
        filter_config = config.get("grasp_filter", {})
        grasps, grasp_stats = filter_grasps_to_object(
            grasps,
            region_points,
            max_center_dist=float(
                filter_config.get("max_center_dist", 0.04)
            ),
            bbox_margin=float(filter_config.get("bbox_margin", 0.04)),
            min_inner_points=int(filter_config.get("min_inner_points", 5)),
            enforce_center_distance=(
                region_source == "reason_part_mask"
            ),
        )
        grasp_stats["region_source"] = region_source
        grasps, collision_stats = filter_collision_grasps(
            grasps,
            point_cloud[0],
            config.get("collision_filter", {}),
        )
        grasps, topdown_stats = prefer_topdown_grasps(
            grasps,
            config.get("topdown_filter", {}),
        )
    else:
        grasps = raw_grasps
        grasp_stats = {
            "enabled": False,
            "reason": "test_all_raw_candidates",
        }
        collision_stats = dict(grasp_stats)
        topdown_stats = dict(grasp_stats)
    if len(grasps) == 0:
        raise RuntimeError("No grasp candidates remain after filtering")
    return grasps, point_cloud, {
        "grasp_filter": grasp_stats,
        "collision_filter": collision_stats,
        "topdown_filter": topdown_stats,
    }


def _execute_grasp(
    *,
    scene,
    gripper,
    camera,
    capture,
    network,
    device,
    body_id,
    reason_target,
    config,
    args,
    final_target,
    use_reason_part_mask,
    release_settle_steps,
):
    print(
        "  ⏳ 正在提取目标抓取区域并生成 GraspNet 候选...",
        flush=True,
    )
    region_points, region_source, part_diagnostics = _grasp_region(
        capture,
        camera,
        body_id,
        reason_target,
        use_reason_part_mask,
    )
    grasps, point_cloud, filter_stats = _generate_grasps(
        network=network,
        device=device,
        camera=camera,
        capture=capture,
        region_points=region_points,
        region_source=region_source,
        config=config,
        args=args,
    )
    print(
        f"  ✅ 抓取候选生成完成: count={len(grasps)}, "
        f"region={region_source}；开始执行机械臂动作",
        flush=True,
    )
    activated = scene.activate_staged_object(body_id)
    try:
        evaluator = GraspEvaluator(
            object_id=body_id,
            gripper=gripper,
            point_cloud=region_points,
            gui=args.gui,
            assisted_grasp=args.assisted_grasp,
            validate_target_center=False,
            scene_object_ids=scene.object_ids,
            place_target_joint_pose_deg=config.get(
                "place_target_joint_pose_deg"
            ),
            release_after_place=not final_target,
            release_settle_steps=release_settle_steps,
            gui_speed=args.gui_speed,
        )
        evaluate_all = (
            args.test_all_candidates
            or args.test_all_raw_candidates
            or args.stop_on_success
        )
        evaluation_count = len(grasps) if evaluate_all else args.top_k
        results = evaluator.evaluate(
            grasps,
            top_k=evaluation_count,
            # A physical closed loop executes exactly one successful action per
            # observation, irrespective of the legacy single-round CLI setting.
            stop_on_success=True,
            preserve_success_state=True,
        )
    except Exception:
        if activated:
            scene.restage_object(body_id)
        raise
    successful = next(
        (result for result in results if result["success"]),
        None,
    )
    if activated:
        if successful:
            scene.finish_staged_object(body_id)
        else:
            scene.restage_object(body_id)
    return {
        "success": successful is not None,
        "successful_grasp_index": (
            int(successful["grasp_index"]) if successful else None
        ),
        "grasps": results,
        "grasp_region": {
            "source": region_source,
            "point_count": int(len(region_points)),
            "part_mask": part_diagnostics,
        },
        "point_cloud": point_cloud,
        "activated_from_staging": activated,
        **filter_stats,
    }


def _write_round_checkpoint(
    *,
    output_path,
    instruction,
    policy,
    max_task_rounds,
    rounds,
    task_status,
    configured_target_body_id,
    configured_target_name,
    pending_occluder_ids,
    scene_config,
):
    """Atomically persist completed rounds so interruptions remain debuggable."""
    checkpoint = _json_safe(
        {
            "mode": "semantic_target_physical_closed_loop",
            "in_progress": True,
            "success": 0,
            "task_success": False,
            "task_status": task_status,
            "instruction": instruction,
            "occlusion_action_policy": policy,
            "max_task_rounds": int(max_task_rounds),
            "round_count": len(rounds),
            "last_completed_round": (
                int(rounds[-1]["round"]) if rounds else None
            ),
            "rounds": rounds,
            "configured_target_body_id": configured_target_body_id,
            "configured_target_name": configured_target_name,
            "remaining_occluder_ids": sorted(pending_occluder_ids),
            "scene_config": scene_config,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.round-checkpoint.tmp"
    )
    temporary_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(
        "  💾 轮次检查点已保存: "
        f"round={checkpoint['last_completed_round']} -> {output_path}",
        flush=True,
    )


def _is_final_target_action(reason_target):
    target_id = reason_target.get("target_object_id")
    grasp_id = reason_target.get("object_id")
    return bool(
        reason_target.get("branch") == FINAL_BRANCH
        or (
            target_id is not None
            and grasp_id is not None
            and int(target_id) == int(grasp_id)
        )
    )


def run_task_closed_loop(args):
    """Run until the instruction's final target is physically grasped."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    checkpoint = args.ckpt or os.path.join(
        Path(__file__).resolve().parents[1],
        "checkpoints",
        "checkpoint-rs.tar",
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = load_scene_config(args.scene_config)
    capture_pose = config.get("capture_joint_pose_deg")
    place_pose = config.get("place_target_joint_pose_deg")
    if place_pose is None:
        raise ValueError(
            "Task closed loop requires place_target_joint_pose_deg"
        )
    release_settle_steps = (
        args.drop_settle_steps
        if args.drop_settle_steps is not None
        else int(
            config.get("continuous_grasp", {}).get(
                "drop_settle_steps",
                180,
            )
        )
    )

    scene = SimulationScene(gui=args.gui)
    gripper = None
    video_recorder = None
    network = None
    rounds = []
    latest_visualization = None
    task_success = False
    task_status = "not_started"
    final_target_body_id = None
    final_target_name = None
    failed_auto_pushes = set()
    pushed_occluder_ids = set()
    configured_target_body_id = None
    configured_target_name = None
    configured_target_object = None
    pending_occluder_ids = set()
    output_path = Path(args.output)

    def save_round_checkpoint():
        _write_round_checkpoint(
            output_path=output_path,
            instruction=str(args.instruction).strip(),
            policy=args.occlusion_action,
            max_task_rounds=args.max_task_rounds,
            rounds=rounds,
            task_status=task_status,
            configured_target_body_id=configured_target_body_id,
            configured_target_name=configured_target_name,
            pending_occluder_ids=pending_occluder_ids,
            scene_config=config["_path"],
        )

    try:
        scene.connect()
        scene.load_plane()
        scene.load_objects(config["_resolved_objects"])
        configured_target = _configured_target(
            scene,
            str(args.instruction).strip(),
        )
        if configured_target is not None:
            configured_target_body_id, configured_target_object = configured_target
            configured_target_name = configured_target_object.name
            pending_occluder_ids = set(
                _configured_occluders(
                    scene,
                    configured_target_name,
                )
            )
            print(
                "  已知遮挡物 body_id: "
                f"{sorted(pending_occluder_ids) or 'none'}"
            )
        staging_enabled = bool(
            config.get("object_staging", {}).get(
                "lock_initial_poses_until_grasp",
                False,
            )
        )
        if staging_enabled:
            scene.stage_objects_at_initial_poses()
            print(
                "🔒 初始场景已锁定；每轮被操作物体会恢复动态质量"
            )
        scene.step(int(config.get("settle_steps", 300)))

        camera = _make_camera(config)
        gripper = create_gripper(
            args.gripper_model,
            planner=None,
            initial_joint_pose_deg=capture_pose,
            robot_base_yaw_deg=float(
                config.get("robot_base_yaw_deg", 0.0)
            ),
            gui_motion_step_delay=(
                0.003 / args.gui_speed if args.gui else 0.0
            ),
        )
        gripper.load()
        if args.record_video:
            from simulation.video_recorder import PyBulletVideoRecorder

            video_path = (
                args.video_output
                or args.output.replace(".json", "_pybullet.mp4")
            )
            video_recorder = PyBulletVideoRecorder(video_path)
            video_recorder.start()
            atexit.register(video_recorder.close)

        print(
            "🎯 最终目标闭环启动: "
            f"policy={args.occlusion_action}, "
            f"max_rounds={args.max_task_rounds}"
        )
        for round_index in range(1, args.max_task_rounds + 1):
            print(
                f"\n========== Task round "
                f"{round_index}/{args.max_task_rounds} =========="
            )
            requested_scene_id = (
                args.scene_id if round_index == 1 else None
            )
            try:
                capture = _capture_and_reason(
                    camera=camera,
                    gripper=gripper,
                    capture_pose=capture_pose,
                    instruction=_round_instruction(
                        args.instruction,
                        round_index,
                    ),
                    requested_scene_id=requested_scene_id,
                    network=network,
                    device=device,
                    allow_unselected_object=(
                        configured_target_body_id is not None
                    ),
                )
                reason_target = capture["reason_target"]
                reason_mapping_role = None
                target_visibility = (
                    _configured_target_visibility(
                        capture,
                        configured_target_body_id,
                        configured_target_object,
                    )
                    if (
                        configured_target_body_id is not None
                        and configured_target_object is not None
                    )
                    else None
                )
                if (
                    target_visibility is not None
                    and target_visibility["visible_enough_for_grasp"]
                ):
                    reason_body_id = configured_target_body_id
                    reason_scene_object = configured_target_object
                    reason_mapping_role = "configured_target"
                    reason_selection = target_visibility
                    print(
                        "  👁️ 配置目标已充分显露: "
                        f"pixels={target_visibility['visible_pixels']}, "
                        f"ratio={target_visibility['visible_ratio']:.6f}, "
                        "转入目标抓取"
                    )
                elif reason_target.get("object_id") is None:
                    if configured_target_body_id is None:
                        raise RuntimeError(
                            "Reason did not select grasp_object.id and no "
                            "configured target fallback is available: "
                            f"{reason_target['reason_summary_path']}"
                        )
                    reason_body_id = None
                    reason_scene_object = None
                    reason_selection = {
                        "source": "reason_no_visible_grasp_object",
                        "reason_status": reason_target.get("status"),
                        "reason_branch": reason_target.get("branch"),
                        "reason_summary_path": reason_target[
                            "reason_summary_path"
                        ],
                    }
                    print(
                        "  👁️ Reason 未发现可见抓取目标；"
                        "使用场景配置继续闭环"
                    )
                else:
                    configured_target_mapping = None
                    if configured_target_body_id is not None:
                        try:
                            target_mapping = _map_reason_target(
                                scene,
                                capture,
                                args.target_mask_min_iou,
                            )
                            if (
                                target_mapping is not None
                                and target_mapping[0]
                                == configured_target_body_id
                            ):
                                configured_target_mapping = target_mapping
                        except Exception as target_mapping_error:
                            print(
                                "  ⚠️ Reason 目标 mask 映射失败；"
                                f"继续检查动作对象: {target_mapping_error}"
                            )
                    try:
                        (
                            reason_body_id,
                            reason_scene_object,
                            reason_selection,
                        ) = _map_reason_object(
                            scene,
                            capture,
                            args.target_mask_min_iou,
                            prefer_part_mask=args.use_reason_part_mask,
                        )
                        reason_mapping_role = "grasp_object"
                    except Exception as grasp_mapping_error:
                        if configured_target_mapping is not None:
                            (
                                reason_body_id,
                                reason_scene_object,
                                target_selection,
                            ) = configured_target_mapping
                            reason_mapping_role = "target_object"
                            reason_selection = {
                                **target_selection,
                                "source": "configured_target_mask_fallback",
                                "grasp_object_mapping_error": str(
                                    grasp_mapping_error
                                ),
                            }
                            print(
                                "  🎯 Reason 动作对象无法映射；"
                                "已验证目标 mask，直接转入目标抓取"
                            )
                        elif (
                            configured_target_body_id is not None
                            and pending_occluder_ids
                        ):
                            reason_body_id = None
                            reason_scene_object = None
                            reason_mapping_role = None
                            reason_selection = {
                                "source": "configured_occluder_after_"
                                "invalid_reason_mask",
                                "grasp_object_mapping_error": str(
                                    grasp_mapping_error
                                ),
                                "reason_summary_path": reason_target[
                                    "reason_summary_path"
                                ],
                            }
                            print(
                                "  ⚠️ Reason 动作对象无法映射；"
                                "目标尚未显露，继续使用配置遮挡物"
                            )
                        else:
                            raise
            except Exception as error:
                task_status = f"pipeline_failed: {error}"
                rounds.append(
                    {
                        "round": round_index,
                        "action_success": False,
                        "task_complete": False,
                        "failure_reason": str(error),
                    }
                )
                save_round_checkpoint()
                break

            reason_branch = str(reason_target.get("branch") or "")
            body_id = reason_body_id
            scene_object = reason_scene_object
            selection = reason_selection
            branch = reason_branch

            # A Reason mask mapped to the configured target is the evidence
            # that a pushed occluder has revealed it. Prefer that evidence over
            # the still-pending configured occlusion relation.
            if (
                configured_target_body_id is not None
                and reason_body_id == configured_target_body_id
            ):
                if pending_occluder_ids:
                    print(
                        "  👀 已重新观察到配置目标；"
                        "遮挡物处理阶段结束"
                    )
                pending_occluder_ids.clear()
                body_id = configured_target_body_id
                scene_object = scene.get_object_info(body_id)
                branch = FINAL_BRANCH
                selection = _selection_override(
                    reason_selection=reason_selection,
                    selected_body_id=body_id,
                    selected_object=scene_object,
                    source="reason_mapped_configured_target",
                )
            # Otherwise keep acting on the configured occluder. A successful
            # push only proves motion, not that the target is visible yet.
            elif (
                configured_target_body_id is not None
                and pending_occluder_ids
            ):
                body_id = (
                    reason_body_id
                    if reason_body_id in pending_occluder_ids
                    else min(pending_occluder_ids)
                )
                scene_object = scene.get_object_info(body_id)
                branch = _configured_occlusion_branch(
                    scene.get_object_info(configured_target_body_id)
                )
                selection = _selection_override(
                    reason_selection=reason_selection,
                    selected_body_id=body_id,
                    selected_object=scene_object,
                    source="configured_occlusion_relation",
                )
            elif configured_target_body_id is not None:
                body_id = configured_target_body_id
                scene_object = scene.get_object_info(body_id)
                branch = FINAL_BRANCH
                selection = _selection_override(
                    reason_selection=reason_selection,
                    selected_body_id=body_id,
                    selected_object=scene_object,
                    source="configured_final_target_after_occluders",
                )

            final_action = (
                body_id == configured_target_body_id
                if configured_target_body_id is not None
                else _is_final_target_action(reason_target)
            )
            action = _resolve_action(
                args.occlusion_action,
                branch,
                body_id,
                failed_auto_pushes,
                final_target=final_action,
            )
            print(
                "  🧭 物理动作: "
                f"reason_branch={reason_branch}, "
                f"effective_branch={branch}, "
                f"action={action}, body_id={body_id}, "
                f"name={scene_object.name}, final={final_action}"
            )
            round_record = {
                "round": round_index,
                "scene_id": int(capture["scene_id"]),
                "branch": branch,
                "reason_branch": reason_branch,
                "target_object": {
                    "id": (
                        configured_target_body_id
                        if configured_target_body_id is not None
                        else reason_target.get("target_object_id")
                    ),
                    "label": (
                        configured_target_name
                        if configured_target_name is not None
                        else reason_target.get("target_object_label")
                    ),
                },
                "reason_mapped_object": {
                    "role": reason_mapping_role,
                    "reason_id": (
                        reason_target.get("target_object_id")
                        if reason_mapping_role == "target_object"
                        else reason_target.get("object_id")
                    ),
                    "reason_label": (
                        reason_target.get("target_object_label")
                        if reason_mapping_role == "target_object"
                        else reason_target["object_label"]
                    ),
                    "body_id": (
                        int(reason_body_id)
                        if reason_body_id is not None
                        else None
                    ),
                    "name": (
                        reason_scene_object.name
                        if reason_scene_object is not None
                        else None
                    ),
                },
                "grasp_object": {
                    "reason_id": reason_target.get("object_id"),
                    "reason_label": reason_target["object_label"],
                    "body_id": int(body_id),
                    "name": scene_object.name,
                },
                "action": action,
                "action_role": (
                    "final_target" if final_action else "occluder"
                ),
                "target_selection": selection,
                "reason_target": reason_target,
                "task_complete": False,
            }

            try:
                if action == "push":
                    push_result, activated = _execute_push(
                        scene=scene,
                        gripper=gripper,
                        body_id=body_id,
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
                else:
                    if network is None:
                        network, epoch = load_graspnet_model(
                            checkpoint,
                            device,
                        )
                        print(f"  ✅ GraspNet loaded, epoch={epoch}")
                    action_result = _execute_grasp(
                        scene=scene,
                        gripper=gripper,
                        camera=camera,
                        capture=capture,
                        network=network,
                        device=device,
                        body_id=body_id,
                        reason_target=reason_target,
                        config=config,
                        args=args,
                        final_target=final_action,
                        use_reason_part_mask=bool(
                            args.use_reason_part_mask
                            and body_id == reason_body_id
                            and reason_mapping_role == "grasp_object"
                        ),
                        release_settle_steps=release_settle_steps,
                    )
            except Exception as error:
                print(
                    f"  ❌ {action} 在机械臂执行前或执行中失败: {error}",
                    flush=True,
                )
                action_result = {
                    "success": False,
                    "failure_reason": str(error),
                }

            action_success = bool(action_result["success"])
            round_record["action_success"] = action_success
            round_record["action_result"] = action_result
            latest_visualization = {
                "rgb": capture["rgb"],
                "depth": capture["depth"],
                "seg": capture["segmentation"],
                "point_cloud": action_result.get("point_cloud"),
                "perception_input": capture["perception_input"],
                "reason_target": reason_target,
                "target_selection": selection,
                "target_body_id": int(body_id),
                "target_object_name": scene_object.name,
            }

            if (
                not final_action
                and action_success
                and body_id in pending_occluder_ids
            ):
                if action == "push":
                    pushed_occluder_ids.add(body_id)
                    print(
                        "  🔄 遮挡物推动成功；保留待处理状态，"
                        "下一轮重观察确认目标是否显露"
                    )
                else:
                    pending_occluder_ids.discard(body_id)
                    pushed_occluder_ids.discard(body_id)

            if final_action and action_success:
                task_success = True
                task_status = "final_target_grasped"
                final_target_body_id = int(body_id)
                final_target_name = scene_object.name
                round_record["task_complete"] = True
                rounds.append(round_record)
                save_round_checkpoint()
                print(
                    f"🏁 最终目标抓取成功: {scene_object.name} "
                    f"(round={round_index})"
                )
                break

            rounds.append(round_record)
            if not action_success:
                if final_action:
                    task_status = "final_target_grasp_failed_retry"
                    if pushed_occluder_ids:
                        retry_occluder_id = min(
                            pushed_occluder_ids
                        )
                        pending_occluder_ids.add(retry_occluder_id)
                        if args.occlusion_action == "auto":
                            failed_auto_pushes.add(
                                retry_occluder_id
                            )
                        print(
                            "  ⚠️ 最终目标抓取失败；重新处理之前推动过的"
                            f"遮挡物 body_id={retry_occluder_id}"
                        )
                elif (
                    args.occlusion_action == "auto"
                    and action == "push"
                ):
                    failed_auto_pushes.add(body_id)
                    task_status = "auto_push_failed_retry_grasp_away"
                    print(
                        "  ⚠️ 推动失败；下一轮 auto 将改为抓走该遮挡物"
                    )
                else:
                    task_status = f"{action}_failed"
            else:
                task_status = "reobserve_after_intermediate_action"

            save_round_checkpoint()
            scene.step(args.reobserve_settle_steps)
        else:
            task_status = "max_task_rounds_reached"

        output = _json_safe(
            {
                "mode": "semantic_target_physical_closed_loop",
                "in_progress": False,
                "success": int(task_success),
                "task_success": task_success,
                "task_status": task_status,
                "instruction": str(args.instruction).strip(),
                "occlusion_action_policy": args.occlusion_action,
                "max_task_rounds": int(args.max_task_rounds),
                "round_count": len(rounds),
                "rounds": rounds,
                "final_target_body_id": final_target_body_id,
                "final_target_name": final_target_name,
                "configured_target_body_id": configured_target_body_id,
                "configured_target_name": configured_target_name,
                "remaining_occluder_ids": sorted(
                    pending_occluder_ids
                ),
                "scene_config": config["_path"],
                "object_staging_enabled": staging_enabled,
                "push_distance": float(args.push_distance),
                "push_direction": args.push_direction,
                "reobserve_settle_steps": int(
                    args.reobserve_settle_steps
                ),
                "release_settle_steps": int(release_settle_steps),
                "final_scene_objects": scene.get_object_poses(),
                "gripper": gripper.metadata(),
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if latest_visualization is not None:
            visualization_path = output_path.with_name(
                output_path.stem + "_viz_data.pkl"
            )
            with visualization_path.open("wb") as visualization_file:
                pickle.dump(latest_visualization, visualization_file)
        print(
            f"\n📊 最终任务: "
            f"{'SUCCESS' if task_success else 'FAILED'} "
            f"({task_status})"
        )
        print(f"💾 结果已保存: {output_path}")
        return output
    finally:
        if video_recorder is not None:
            video_recorder.close()
            try:
                atexit.unregister(video_recorder.close)
            except Exception:
                pass
        if gripper is not None:
            gripper.remove()
        scene.disconnect()
