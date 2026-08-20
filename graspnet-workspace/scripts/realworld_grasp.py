#!/usr/bin/env python
"""Real-world SmartGrasp pipeline for eye-in-hand RGB-D capture and GraspNet.

This entrypoint intentionally does not use PyBullet. It can move the robot to a
fixed calibrated ready pose, capture one aligned RealSense RGB-D frame, build a
camera-frame point cloud, run GraspNet, export candidate visualization, and
optionally execute a minimal JAKA grasp sequence.

The camera-to-robot transform reuses the legacy chessboard calibration directory:

    T_robot_grasp = T_plate_to_robot @ inv(T_board_to_camera) @ T_camera_grasp

The legacy matrices are millimeter based. GraspNet translations are converted
from meters to millimeters before applying that chain.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SMARTGRASP_ROOT = WORKSPACE_ROOT.parent
sys.path = [
    path
    for path in sys.path
    if "graspnet-workspace/pointnet2" not in path and "graspnet-workspace/knn" not in path
]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "models"))
sys.path.insert(0, str(WORKSPACE_ROOT / "utils"))
sys.path.insert(0, str(WORKSPACE_ROOT / "graspnet_api"))

from graspnetAPI import GraspGroup  # noqa: E402
from simulation.candidate_visualizer import export_candidate_html, export_candidate_ply, export_candidate_png  # noqa: E402
from utils.camera import build_grasp_point_cloud, capture_realsense, load_captured_frame  # noqa: E402
from utils.data_loader import json_safe as _json_safe, sample_points  # noqa: E402
from utils.grasp_processing import (  # noqa: E402
    build_pca_fallback_grasp,
    compute_robot_targets,
    filter_grasp_center_outliers,
    filter_grasp_centers_in_target_mask,
    filter_grasp_collisions,
    filter_grasp_widths_by_mask_consistency,
    filter_target_tcp_z,
    grasp_to_record,
    jaka_pose_to_transform,
    load_hand_eye_calibration,
    load_legacy_plate_calibration,
    offset_transform_along_tcp_z,
    print_candidate_target_centers,
    renumber_candidate_records,
    rerank_candidates_by_topdown,
    run_graspnet,
    transform_to_jaka_pose,
)
from utils.joint import PersistentJakaWorker, read_jaka_tcp_pose, run_jaka_sequence  # noqa: E402
from utils.realworld_config import config_get, config_path, load_realworld_config  # noqa: E402


REALWORLD_CONFIG_PATH = WORKSPACE_ROOT / "config" / "realworld_config.yaml"
REALWORLD_CONFIG = load_realworld_config(REALWORLD_CONFIG_PATH)
IMAGE_WIDTH = int(config_get(REALWORLD_CONFIG, "image.width", 1280))
IMAGE_HEIGHT = int(config_get(REALWORLD_CONFIG, "image.height", 720))
DATA_REALWORLD_ROOT = SMARTGRASP_ROOT / "data_realworld"
DEFAULT_OUTPUT_DIR = config_path(REALWORLD_CONFIG, "paths.output_dir", WORKSPACE_ROOT, SMARTGRASP_ROOT / "result")
DEFAULT_TRIAL_LOG_ROOT = config_path(REALWORLD_CONFIG, "paths.trial_log_root", WORKSPACE_ROOT, WORKSPACE_ROOT / "log")
DEFAULT_TRIAL_LOG_SUBDIR = str(config_get(REALWORLD_CONFIG, "paths.trial_log_subdir", "single_object_grasp"))
DEFAULT_CHECKPOINT = config_path(REALWORLD_CONFIG, "paths.checkpoint", WORKSPACE_ROOT, WORKSPACE_ROOT / "checkpoints" / "checkpoint-rs.tar")
DEFAULT_HAND_EYE_CALIBRATION = config_path(
    REALWORLD_CONFIG,
    "paths.hand_eye_calibration",
    WORKSPACE_ROOT,
    WORKSPACE_ROOT / "calibration" / "hand_eye_tcp_camera.json",
)
DEFAULT_CAMERA_COORDINATES_DIR = config_path(
    REALWORLD_CONFIG,
    "paths.camera_coordinates_dir",
    WORKSPACE_ROOT,
    "/home/admin128/ChengyuanWang/high_low_comm/scripts/human_playdata_process/hand_object_detector/camera_coordinates/camera_coordinates - 副本",
)
DEFAULT_TCP_CAMERA_TRANSLATION_OFFSET_MM = config_get(REALWORLD_CONFIG, "calibration.tcp_camera_translation_offset_mm", [0.0, 0.0, -82.5])
DEFAULT_GRASP_CENTER_TO_TCP_OFFSET_MM = float(config_get(REALWORLD_CONFIG, "calibration.grasp_center_to_tcp_offset_mm", 165.0))
BASE_GRIPPER_OPENING_AXIS = np.array(config_get(REALWORLD_CONFIG, "calibration.gripper_opening_axis_base", [0.0, 1.0, 0.0]), dtype=float)
DEFAULT_GRIPPER_ROLL_OFFSET_DEG = float(config_get(REALWORLD_CONFIG, "calibration.gripper_roll_offset_deg", 120.0))
DEFAULT_TCP_TARGET_TRANSLATION_OFFSET_MM = np.array(
    config_get(REALWORLD_CONFIG, "calibration.tcp_target_translation_offset_mm", [-23.0, -30.0, 15.0]),
    dtype=float,
)
DEFAULT_JAKA_IP = str(config_get(REALWORLD_CONFIG, "robot.jaka_ip", "192.168.1.199"))
DEFAULT_ROBOTIQ_PORT = str(config_get(REALWORLD_CONFIG, "gripper.robotiq_port", "/dev/ttyUSB0"))
DEFAULT_GRIPPER_OPEN_FORCE = int(config_get(REALWORLD_CONFIG, "gripper.open_force", 30))
DEFAULT_GRIPPER_CLOSE_FORCE = int(config_get(REALWORLD_CONFIG, "gripper.close_force", 200))
DEFAULT_READY_POSE = config_get(REALWORLD_CONFIG, "robot.ready_pose", [300.0, 0.0, 350.0, 3.141592653589793, 0.0, 0.0])
DEFAULT_CAPTURE_JOINT_POSE_DEG = config_get(REALWORLD_CONFIG, "robot.capture_joint_pose_deg", [0.0, 90.0, 45.0, 135.0, 270.0, 72.0])
DEFAULT_PLACE_TARGET_JOINT_POSE_DEG = config_get(REALWORLD_CONFIG, "robot.place_target_joint_pose_deg", [-75.0, 90.0, 45.0, 135.0, 270.0, 72.0])
DEFAULT_PLACE_RELEASE_LOWER_MM = float(config_get(REALWORLD_CONFIG, "robot.place_release_lower_mm", 50.0))
DEFAULT_GRASP_EXTRA_DEPTH_MM = float(config_get(REALWORLD_CONFIG, "robot.grasp_extra_depth_mm", 10.0))
DEFAULT_JOINT_VELOCITY_RAD_S = float(config_get(REALWORLD_CONFIG, "robot.joint_velocity_rad_s", 0.5))
DEFAULT_CAMERA_INDEX = int(config_get(REALWORLD_CONFIG, "camera.default_index", 1))
DEFAULT_CAMERA_SERIAL_SUFFIX = str(config_get(REALWORLD_CONFIG, "camera.default_serial_suffix", "76630"))
DEFAULT_JAKA_PYTHON = os.environ.get(
    "JAKA_PYTHON",
    str(config_path(REALWORLD_CONFIG, "paths.jaka_python", WORKSPACE_ROOT, "/home/admin128/anaconda3/envs/smartgrasp310/bin/python")),
)
DEFAULT_TARGET_MASK_CENTER_TOLERANCE_PX = float(config_get(REALWORLD_CONFIG, "filters.target_mask_center_tolerance_px", 25.0))
DEFAULT_FILTER_GRASP_CENTERS_IN_MASK = bool(
    config_get(REALWORLD_CONFIG, "filters.filter_grasp_centers_in_mask", False)
)
DEFAULT_FILTER_GRASP_OUTLIERS = bool(config_get(REALWORLD_CONFIG, "filters.filter_grasp_outliers", False))
DEFAULT_FILTER_GRASP_CLOSING_POINTS = bool(
    config_get(REALWORLD_CONFIG, "filters.filter_grasp_closing_points", False)
)
DEFAULT_FILTER_GRASP_WIDTH_FROM_MASK = bool(
    config_get(REALWORLD_CONFIG, "filters.filter_grasp_width_from_mask", False)
)
DEFAULT_MIN_TARGET_TCP_Z_MM = float(config_get(REALWORLD_CONFIG, "filters.min_target_tcp_z_mm", 125.0))
DEFAULT_GEOMETRY_SCORE_WEIGHT = float(config_get(REALWORLD_CONFIG, "ranking.geometry_score_weight", 0.5))
DEFAULT_WIDTH_QUALITY_WEIGHT = float(config_get(REALWORLD_CONFIG, "ranking.width_quality_weight", 2.0))
DEFAULT_CENTERING_QUALITY_WEIGHT = float(config_get(REALWORLD_CONFIG, "ranking.centering_quality_weight", 1.0))
DEFAULT_GRASP_WIDTH_MIN_CONTACT_POINTS = int(config_get(REALWORLD_CONFIG, "filters.grasp_width_min_contact_points", 200))
DEFAULT_GRASP_CLOSING_MIN_POINTS = int(config_get(REALWORLD_CONFIG, "filters.grasp_closing_min_points", 4000))
DEFAULT_GRASP_CLOSING_MIN_INPUT_RATIO = float(
    config_get(REALWORLD_CONFIG, "filters.grasp_closing_min_input_ratio", 0.6)
)
DEFAULT_GRASP_WIDTH_PERCENTILE_LOW = float(config_get(REALWORLD_CONFIG, "filters.grasp_width_percentile_low", 2.0))
DEFAULT_GRASP_WIDTH_PERCENTILE_HIGH = float(config_get(REALWORLD_CONFIG, "filters.grasp_width_percentile_high", 98.0))
DEFAULT_GRASP_WIDTH_TOLERANCE_MM = float(config_get(REALWORLD_CONFIG, "filters.grasp_width_tolerance_mm", 20.0))
DEFAULT_GRASP_WIDTH_MAX_CENTER_OFFSET_RATIO = float(
    config_get(REALWORLD_CONFIG, "filters.grasp_width_max_center_offset_ratio", 0.5)
)
DEFAULT_GRASP_FILTER_FINGER_LENGTH_MM = float(
    config_get(REALWORLD_CONFIG, "filters.grasp_filter_finger_length_mm", 60.0)
)
DEFAULT_GRASP_FILTER_FINGER_WIDTH_MM = float(
    config_get(REALWORLD_CONFIG, "filters.grasp_filter_finger_width_mm", 30.0)
)
VENDOR_DIR = WORKSPACE_ROOT / "vendor"
JKRC_DIR = config_path(REALWORLD_CONFIG, "paths.jkrc_dir", WORKSPACE_ROOT, WORKSPACE_ROOT / "jkrc")
JAKA_WORKER = config_path(REALWORLD_CONFIG, "paths.jaka_worker", WORKSPACE_ROOT, WORKSPACE_ROOT / "scripts" / "jaka_motion_worker.py")
JAKA_WORKER_READY_PREFIX = "__JAKA_READY__ "
JAKA_WORKER_RESPONSE_PREFIX = "__JAKA_RESPONSE__ "
DEFAULT_PLATE_TO_ROBOT_MM = np.array(config_get(REALWORLD_CONFIG, "calibration.plate_to_robot_mm"), dtype=float)


def resolve_capture_tcp_pose(output_dir: Path, args: argparse.Namespace, capture_was_reused: bool) -> list[float]:
    pose_path = output_dir / "capture_tcp_pose.json"
    if args.capture_tcp_pose is not None:
        tcp_pose = [float(value) for value in args.capture_tcp_pose]
    elif capture_was_reused:
        if not pose_path.exists():
            raise FileNotFoundError(
                f"Hand-eye mode with --reuse-capture needs {pose_path} or --capture-tcp-pose. "
                "The TCP pose must match the moment when rgb/depth were captured."
            )
        tcp_pose = [float(value) for value in json.loads(pose_path.read_text(encoding="utf-8"))["tcp_pose"]]
    else:
        tcp_pose = read_jaka_tcp_pose(args)
    pose_path.write_text(
        json.dumps({"tcp_pose": tcp_pose, "timestamp": time.time()}, indent=2),
        encoding="utf-8",
    )
    return tcp_pose


def prepare_robot(args: argparse.Namespace) -> None:
    if args.skip_ready:
        return
    capture_joints_rad = np.deg2rad(np.asarray(args.capture_joint_pose_deg, dtype=float)).astype(float).tolist()
    run_jaka_sequence(
        [
            {"type": "joint_move", "joints_rad": capture_joints_rad},
            {"type": "gripper", "command": "open"},
        ],
        args,
        label="prepare_robot",
    )


def offset_pose_along_approach(target_transform: np.ndarray, offset_mm: float) -> list[float]:
    approach_axis = target_transform[:3, 2]
    transform = target_transform.copy()
    transform[:3, 3] = transform[:3, 3] - approach_axis * offset_mm
    return transform_to_jaka_pose(transform)


def lift_pose_from_target(target_transform: np.ndarray, lift_mm: float) -> list[float]:
    transform = target_transform.copy()
    transform[2, 3] = transform[2, 3] + lift_mm
    return transform_to_jaka_pose(transform)


def execute_grasp_sequence(record: dict[str, Any], args: argparse.Namespace) -> None:
    target_transform = np.asarray(
        record.get("target_robot_from_tcp", record["target_robot_from_grasp"]),
        dtype=float,
    ).reshape(4, 4)
    execution_target = record.get("execution_target_robot_from_tcp")
    if execution_target is None:
        execution_target = offset_transform_along_tcp_z(target_transform, args.grasp_extra_depth_mm)
    execution_target_transform = np.asarray(execution_target, dtype=float).reshape(4, 4)
    pre_grasp_pose = offset_pose_along_approach(target_transform, args.approach_offset_mm)
    grasp_pose = transform_to_jaka_pose(execution_target_transform)
    lift_pose = lift_pose_from_target(execution_target_transform, args.lift_mm)
    initial_joints_rad = np.deg2rad(np.asarray(args.capture_joint_pose_deg, dtype=float)).astype(float).tolist()
    place_target_joints_rad = np.deg2rad(
        np.asarray(args.place_target_joint_pose_deg, dtype=float)
    ).astype(float).tolist()

    run_jaka_sequence(
        [
            {"type": "move", "pose": pre_grasp_pose},
            {"type": "move", "pose": grasp_pose},
            {"type": "gripper", "command": "close"},
            {"type": "move", "pose": lift_pose},
            {"type": "joint_move", "joints_rad": initial_joints_rad},
            {"type": "joint_move", "joints_rad": place_target_joints_rad},
            {"type": "move_relative_base", "translation_mm": [0.0, 0.0, -args.place_release_lower_mm]},
            {"type": "gripper", "command": "open"},
            {"type": "joint_move", "joints_rad": initial_joints_rad},
        ],
        args,
        label="execute_grasp_sequence",
    )


def save_outputs(
    output_dir: Path,
    grasp_cloud: np.ndarray,
    grasp_cloud_rgb: np.ndarray | None,
    grasps: GraspGroup,
    top_k: int,
    calibration: dict[str, Any] | None,
    args: argparse.Namespace,
    point_cloud_info: dict[str, Any] | None = None,
    ply_cloud: np.ndarray | None = None,
    ply_cloud_rgb: np.ndarray | None = None,
    obstacle_cloud: np.ndarray | None = None,
    candidate_source: str = "graspnet",
    pca_fallback_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    top_count = min(top_k, len(grasps))
    top_grasps = grasps[:top_count]
    records = [grasp_to_record(top_grasps[index], index) for index in range(top_count)]
    for record in records:
        record["candidate_source"] = candidate_source
    target_mask = None if point_cloud_info is None else point_cloud_info.get("object_mask_array")
    intrinsics = None if point_cloud_info is None else point_cloud_info.get("camera_intrinsics")
    records, target_mask_filter_info = filter_grasp_centers_in_target_mask(records, target_mask, intrinsics, args)
    records = compute_robot_targets(
        records,
        calibration,
        args.grasp_center_to_tcp_offset_mm,
        args.gripper_roll_offset_deg,
        BASE_GRIPPER_OPENING_AXIS,
        DEFAULT_TCP_TARGET_TRANSLATION_OFFSET_MM,
    )
    records, target_tcp_z_filter_info = filter_target_tcp_z(records, args)
    top_grasps_for_collision = GraspGroup()
    for record in records:
        raw_index = int(record["grasp_index"])
        if 0 <= raw_index < len(top_grasps):
            grasp_for_collision = top_grasps[raw_index]
            grasp_for_collision.width = float(record["width"])
            top_grasps_for_collision.add(grasp_for_collision)
    collision_obstacle_cloud = grasp_cloud if obstacle_cloud is None else obstacle_cloud
    records, collision_filter_info = filter_grasp_collisions(
        records,
        top_grasps_for_collision,
        collision_obstacle_cloud,
        args,
    )
    object_cloud_for_width = None
    if point_cloud_info is not None and point_cloud_info.get("object_point_cloud_path") is not None:
        object_cloud_for_width = np.load(point_cloud_info["object_point_cloud_path"])
    records, grasp_width_filter_info = filter_grasp_widths_by_mask_consistency(
        records,
        object_cloud_for_width,
        len(grasp_cloud),
        args,
    )
    records, outlier_filter_info = filter_grasp_center_outliers(records, args)
    records, topdown_rerank_info = rerank_candidates_by_topdown(records, args)
    records = renumber_candidate_records(records)
    print_candidate_target_centers(records, args.candidate_index, args.approach_offset_mm)

    candidate_dicts = [
        {
            "score": item["score"],
            "width": item["width"],
            "depth": item["depth"],
            "translation": item["translation_camera_m"],
            "rotation": item["rotation_camera"],
        }
        for item in records
    ]
    export_candidate_png(
        point_cloud=grasp_cloud,
        candidates=candidate_dicts,
        results=[],
        output_path=output_dir / "grasp_candidates.png",
        max_points=args.viz_max_points,
        gripper_visual_scale=args.gripper_visual_scale,
    )
    export_candidate_html(
        point_cloud=grasp_cloud,
        candidates=candidate_dicts,
        results=[],
        output_path=output_dir / "grasp_candidates_3d.html",
        max_points=args.plotly_max_points,
        gripper_visual_scale=args.gripper_visual_scale,
    )
    export_candidate_ply(
        point_cloud=grasp_cloud if ply_cloud is None else ply_cloud,
        candidates=candidate_dicts,
        output_path=output_dir / "grasp_candidates.ply",
        point_colors=grasp_cloud_rgb if ply_cloud_rgb is None else ply_cloud_rgb,
        max_points=args.ply_max_points,
        gripper_visual_scale=args.gripper_visual_scale,
    )
    serializable_point_cloud_info = None
    if point_cloud_info is not None:
        serializable_point_cloud_info = {
            key: value
            for key, value in point_cloud_info.items()
            if key not in {"object_mask_array"}
        }
    payload = {
        "frame": "camera",
        "top_k": top_count,
        "candidate_source": candidate_source,
        "pca_fallback": pca_fallback_info,
        "num_candidates_after_filter": len(records),
        "grasp_candidates_ply": str((output_dir / "grasp_candidates.ply").resolve()),
        "target_mask_center_filter": target_mask_filter_info,
        "mask_grasp_width_filter": grasp_width_filter_info,
        "model_free_collision_filter": collision_filter_info,
        "target_tcp_z_filter": target_tcp_z_filter_info,
        "center_outlier_filter": outlier_filter_info,
        "topdown_rerank": topdown_rerank_info,
        "point_cloud_source": None if point_cloud_info is None else point_cloud_info.get("point_cloud_source"),
        "grasp_point_cloud_source": None if point_cloud_info is None else point_cloud_info.get("grasp_point_cloud_source"),
        "grasp_input_mode": None if point_cloud_info is None else point_cloud_info.get("grasp_input_mode"),
        "point_cloud_info": serializable_point_cloud_info,
        "calibration_mode": None if calibration is None else calibration["mode"],
        "camera_to_robot_chain": (
            calibration.get("runtime_chain", "T_base_tcp_capture @ T_tcp_camera @ T_camera_grasp")
            if calibration is not None and calibration["mode"] == "hand_eye"
            else "T_plate_to_robot @ inv(T_board_to_camera) @ T_camera_grasp"
        ),
        "hand_eye_runtime_transform": (
            calibration.get("runtime_transform")
            if calibration is not None and calibration["mode"] == "hand_eye"
            else None
        ),
        "hand_eye_translation_offset_mm": (
            calibration.get("translation_offset_mm")
            if calibration is not None and calibration["mode"] == "hand_eye"
            else None
        ),
        "tcp_from_camera_translation_mm": (
            calibration.get("tcp_from_camera_translation_mm")
            if calibration is not None and calibration["mode"] == "hand_eye"
            else None
        ),
        "grasp_center_to_tcp_offset_mm": float(args.grasp_center_to_tcp_offset_mm),
        "grasp_center_to_tcp_offset_axis": "-tcp_local_z",
        "grasp_extra_depth_mm": float(args.grasp_extra_depth_mm),
        "grasp_extra_depth_axis": "+tcp_local_z",
        "gripper_roll_offset_deg": float(args.gripper_roll_offset_deg),
        "base_grasp_to_tcp_rotation": (
            calibration.get("base_grasp_to_tcp_rotation")
            if calibration is not None and "base_grasp_to_tcp_rotation" in calibration
            else None
        ),
        "grasp_to_tcp_rotation": (
            calibration.get("grasp_to_tcp_rotation_with_roll_offset")
            if calibration is not None and "grasp_to_tcp_rotation_with_roll_offset" in calibration
            else None
        ),
        "grasp_to_tcp_rotation_convention": (
            calibration.get("grasp_to_tcp_rotation_convention")
            if calibration is not None and "grasp_to_tcp_rotation_convention" in calibration
            else "fallback_static: tcp_z=grasp_x,tcp_y=grasp_y,tcp_x=-grasp_z"
        ),
        "gripper_opening_axis_base": (
            calibration.get("gripper_opening_axis_base")
            if calibration is not None and "gripper_opening_axis_base" in calibration
            else None
        ),
        "capture_tcp_local_z_base": (
            calibration.get("capture_tcp_local_z_base")
            if calibration is not None and "capture_tcp_local_z_base" in calibration
            else None
        ),
        "capture_gripper_opening_axis_base": (
            calibration.get("capture_gripper_opening_axis_base")
            if calibration is not None and "capture_gripper_opening_axis_base" in calibration
            else None
        ),
        "requires_ready_pose_matching_calibration": not (calibration is not None and calibration["mode"] == "hand_eye"),
        "candidates": records,
    }
    (output_dir / "grasp_candidates.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    return records


def _run_git_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=SMARTGRASP_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def collect_git_info() -> dict[str, Any]:
    status = _run_git_command(["status", "--short"])
    return {
        "branch": _run_git_command(["branch", "--show-current"]),
        "commit": _run_git_command(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def sanitize_trial_name(name: str | None) -> str:
    if not name:
        return ""
    safe_chars = []
    for char in name.strip():
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        elif char.isspace():
            safe_chars.append("_")
    return "".join(safe_chars).strip("_")


def resolve_trial_root(args: argparse.Namespace) -> tuple[Path, str]:
    trial_subdir = sanitize_trial_name(args.trial_log_subdir) or DEFAULT_TRIAL_LOG_SUBDIR
    trial_root = Path(args.trial_log_dir).expanduser() if args.trial_log_dir else DEFAULT_TRIAL_LOG_ROOT / trial_subdir
    if not trial_root.is_absolute():
        trial_root = WORKSPACE_ROOT / trial_root
    return trial_root, trial_subdir


def write_manual_result_template(trial_dir: Path) -> None:
    manual_result_path = trial_dir / "manual_result.json"
    if manual_result_path.exists():
        return
    manual_result = {
        "status": "unreviewed",
        "success": None,
        "failure_reason": "",
        "notes": "",
        "reviewed_by": "",
        "reviewed_at": "",
    }
    manual_result_path.write_text(json.dumps(manual_result, indent=2), encoding="utf-8")


def write_trial_run_info(
    trial_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
    summary: dict[str, Any],
    copied_files: list[str] | None = None,
    missing_files: list[str] | None = None,
    stale_files: list[str] | None = None,
) -> None:
    run_info = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_at_epoch": getattr(args, "_trial_started_at_epoch", None),
        "trial_name": args.trial_name,
        "trial_log_subdir": sanitize_trial_name(args.trial_log_subdir) or DEFAULT_TRIAL_LOG_SUBDIR,
        "trial_dir": str(trial_dir),
        "argv": sys.argv,
        "output_dir": str(output_dir),
        "executed": bool(args.execute),
        "num_candidates": summary.get("num_candidates"),
        "camera_serial": args.camera_serial,
        "calibration_mode": args.calibration_mode,
        "hand_eye_calibration": str(Path(args.hand_eye_calibration).expanduser().resolve()),
        "copied_files": copied_files or [],
        "missing_files": missing_files or [],
        "stale_files": stale_files or [],
        "git": collect_git_info(),
        "summary": summary,
    }
    (trial_dir / "run_info.json").write_text(json.dumps(_json_safe(run_info), indent=2), encoding="utf-8")


def create_trial_log_dir(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.no_trial_log:
        return None

    args._trial_started_at_epoch = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_name = sanitize_trial_name(args.trial_name)
    trial_dir_name = f"{timestamp}_{trial_name}" if trial_name else timestamp
    trial_root, _ = resolve_trial_root(args)
    trial_dir = trial_root / trial_dir_name
    trial_dir.mkdir(parents=True, exist_ok=False)
    initial_summary = {
        "output_dir": str(output_dir),
        "execution_status": "initializing",
        "trial_log_dir": str(trial_dir),
    }
    write_manual_result_template(trial_dir)
    write_trial_run_info(trial_dir, args, output_dir, initial_summary)
    print(f"[trial-log] initialized lightweight trial log: {trial_dir}")
    return trial_dir


def save_trial_log_files(
    trial_dir: Path | None,
    output_dir: Path,
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> None:
    if trial_dir is None:
        return
    copied_files: list[str] = []
    missing_files: list[str] = []
    stale_files: list[str] = []
    file_map = {
        "rgb.png": "rgb.png",
        "depth.raw": "depth.raw",
        "camera_meta.json": "camera_meta.json",
        "grasp_candidates.json": "grasp_candidates.json",
        "grasp_candidates.png": "grasp_candidates.png",
        "grasp_candidates.ply": "scene_grasps.ply",
        "mask_overlay.png": "mask_overlay.png",
    }
    for source_name, target_name in file_map.items():
        source_path = output_dir / source_name
        if source_path.exists():
            started_at = getattr(args, "_trial_started_at_epoch", None)
            if started_at is not None and not args.reuse_capture and source_path.stat().st_mtime < started_at - 1.0:
                stale_files.append(source_name)
                continue
            shutil.copy2(source_path, trial_dir / target_name)
            copied_files.append(target_name)
        else:
            missing_files.append(source_name)

    write_manual_result_template(trial_dir)
    write_trial_run_info(trial_dir, args, output_dir, summary, copied_files, missing_files, stale_files)
    print(f"[trial-log] saved lightweight trial files: {trial_dir}")
    if missing_files:
        print(f"[trial-log] missing optional files: {', '.join(missing_files)}")
    if stale_files:
        print(f"[trial-log] skipped stale files from a previous run: {', '.join(stale_files)}")


def update_trial_run_info(
    trial_dir: Path | None,
    args: argparse.Namespace,
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    if trial_dir is None:
        return
    run_info_path = trial_dir / "run_info.json"
    try:
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    except Exception:
        run_info = {}
    run_info["summary"] = _json_safe(summary)
    run_info["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_info["executed"] = bool(args.execute)
    run_info["output_dir"] = str(output_dir)
    run_info_path.write_text(json.dumps(_json_safe(run_info), indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture real RGB-D, run GraspNet, and optionally move JAKA.")
    parser.add_argument(
        "--instruction",
        default=None,
        help="Natural language instruction for the grasp task (e.g. 'grasp the leftmost apple'). Also saved to instruction.txt.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for rgb.png, depth.raw, and results.")
    parser.add_argument(
        "--trial-log-dir",
        default=None,
        help=(
            "Advanced override for timestamped trial log root. "
            "By default logs are saved under graspnet-workspace/log/<trial-log-subdir>."
        ),
    )
    parser.add_argument(
        "--trial-log-subdir",
        default=DEFAULT_TRIAL_LOG_SUBDIR,
        help="Subdirectory under graspnet-workspace/log for timestamped trial logs.",
    )
    parser.add_argument("--trial-name", default="", help="Optional suffix for the timestamped trial log directory.")
    parser.add_argument("--no-trial-log", action="store_true", help="Disable timestamped lightweight trial log saving.")
    parser.add_argument("--reuse-capture", action="store_true", help="Use existing output-dir/rgb.png + depth.raw.")
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Capture RGB-D (and the hand-eye capture TCP pose) without running SAM or GraspNet.",
    )
    parser.add_argument("--warmup-frames", type=int, default=30, help="RealSense warmup frames before capture.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="Fallback RealSense device index if --camera-serial is empty.")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL_SUFFIX, help="RealSense serial number or unique suffix. Default matches the camera ending with 76630.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT), help="GraspNet checkpoint path.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num-points", type=int, default=20000, help="Point count sampled for GraspNet.")
    parser.add_argument(
        "--if-pca",
        "--if_pca",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If GraspNet returns zero raw candidates, generate one fallback grasp by applying PCA "
            "to the SAM-masked object point cloud. Disabled by default."
        ),
    )
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to save and visualize.")
    parser.add_argument("--viz-max-points", type=int, default=18000, help="Maximum points rendered in grasp_candidates.png.")
    parser.add_argument("--plotly-max-points", type=int, default=30000, help="Maximum points rendered in grasp_candidates_3d.html.")
    parser.add_argument("--ply-max-points", type=int, default=60000, help="Maximum points written to grasp_candidates.ply.")
    parser.add_argument(
        "--gripper-visual-scale",
        type=float,
        default=1.0,
        help="Only scales candidate grippers in visualizations; does not change poses.",
    )
    parser.add_argument("--min-depth", type=float, default=0.10, help="Minimum valid depth in meters.")
    parser.add_argument("--max-depth", type=float, default=1.20, help="Maximum valid depth in meters.")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=None,
        help="Optional camera-frame crop bounds in meters.",
    )
    parser.add_argument(
        "--exclude-camera-box",
        type=float,
        nargs=6,
        action="append",
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=None,
        help="Exclude a camera-frame 3D box in meters after point-cloud generation. Can be repeated.",
    )
    parser.add_argument(
        "--use-sam-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use an interactive SAM mask before GraspNet. Use --no-use-sam-mask to send the full depth cloud.",
    )
    parser.add_argument(
        "--object-min-points",
        type=int,
        default=500,
        help="Minimum masked object point count required before running GraspNet.",
    )
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base", help="SAM model id for interactive mask mode.")
    parser.add_argument("--sam-device", default=None, help="SAM device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--mask-clean-kernel", type=int, default=3, help="Morphological cleanup kernel for SAM masks; use 1 to disable.")
    parser.add_argument(
        "--grasp-input-mode",
        choices=("bbox", "mask"),
        default="mask",
        help=(
            "Point-cloud region sent to GraspNet when SAM is enabled: bbox keeps all valid depth "
            "inside the expanded mask bounding box; mask keeps only valid depth inside the SAM mask."
        ),
    )
    parser.add_argument(
        "--grasp-crop-margin-px",
        type=int,
        default=50,
        help="Fixed pixel margin added around the SAM object bbox before sending points to GraspNet.",
    )
    parser.add_argument(
        "--grasp-crop-margin-ratio",
        type=float,
        default=0.2,
        help="Relative bbox margin added around the SAM object bbox before sending points to GraspNet.",
    )
    parser.add_argument(
        "--filter-grasp-centers-in-mask",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FILTER_GRASP_CENTERS_IN_MASK,
        help="Optionally require each projected candidate center to land inside or near the SAM target mask.",
    )
    parser.add_argument(
        "--target-mask-center-tolerance-px",
        type=float,
        default=DEFAULT_TARGET_MASK_CENTER_TOLERANCE_PX,
        help=(
            "Allow candidate centers this many pixels outside the SAM target mask. "
            "This keeps near-boundary grasp centers while still rejecting clearly off-target candidates."
        ),
    )
    parser.add_argument(
        "--filter-grasp-width-from-mask",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FILTER_GRASP_WIDTH_FROM_MASK,
        help=(
            "Optionally hard-filter candidates using masked width and centering thresholds. Geometry "
            "metrics are still calculated for composite ranking when this is disabled."
        ),
    )
    parser.add_argument(
        "--grasp-width-percentile-low",
        type=float,
        default=DEFAULT_GRASP_WIDTH_PERCENTILE_LOW,
        help="Lower percentile used to measure masked point-cloud width inside each gripper.",
    )
    parser.add_argument(
        "--grasp-width-percentile-high",
        type=float,
        default=DEFAULT_GRASP_WIDTH_PERCENTILE_HIGH,
        help="Upper percentile used to measure masked point-cloud width inside each gripper.",
    )
    parser.add_argument(
        "--grasp-width-min-contact-points",
        type=int,
        default=DEFAULT_GRASP_WIDTH_MIN_CONTACT_POINTS,
        help="Minimum masked point count in the candidate contact slice.",
    )
    parser.add_argument(
        "--filter-grasp-closing-points",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FILTER_GRASP_CLOSING_POINTS,
        help=(
            "Optionally reject candidates whose closing-sweep point count satisfies neither the fixed "
            "minimum nor the GraspNet-input ratio threshold. Counts are still logged when disabled."
        ),
    )
    parser.add_argument(
        "--grasp-closing-min-points",
        type=int,
        default=DEFAULT_GRASP_CLOSING_MIN_POINTS,
        help=(
            "Minimum target-mask point count swept by the jaws while closing from predicted width to zero. "
            "Candidates below this count may still pass via --grasp-closing-min-input-ratio."
        ),
    )
    parser.add_argument(
        "--grasp-closing-min-input-ratio",
        type=float,
        default=DEFAULT_GRASP_CLOSING_MIN_INPUT_RATIO,
        help=(
            "When closing-sweep points are below the fixed minimum, require their count divided by the "
            "unsampled GraspNet input-cloud point count to exceed this ratio."
        ),
    )
    parser.add_argument(
        "--grasp-width-tolerance-mm",
        type=float,
        default=DEFAULT_GRASP_WIDTH_TOLERANCE_MM,
        help="Maximum absolute difference between GraspNet width and masked point-cloud width.",
    )
    parser.add_argument(
        "--grasp-width-max-center-offset-ratio",
        type=float,
        default=DEFAULT_GRASP_WIDTH_MAX_CENTER_OFFSET_RATIO,
        help="Maximum object-center offset divided by half its local opening-axis width.",
    )
    parser.add_argument(
        "--grasp-filter-finger-length-mm",
        type=float,
        default=DEFAULT_GRASP_FILTER_FINGER_LENGTH_MM,
        help="Effective finger length along gripper-local X used to select the contact point-cloud slice.",
    )
    parser.add_argument(
        "--grasp-filter-finger-width-mm",
        type=float,
        default=DEFAULT_GRASP_FILTER_FINGER_WIDTH_MM,
        help="Total finger width along gripper-local Z used to select the contact point-cloud slice.",
    )
    parser.add_argument(
        "--filter-grasp-outliers",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FILTER_GRASP_OUTLIERS,
        help="Optionally remove grasp candidates whose centers are outside the largest spatial cluster.",
    )
    parser.add_argument(
        "--filter-target-tcp-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove candidates whose final physical TCP target is below --min-target-tcp-z-mm.",
    )
    parser.add_argument(
        "--min-target-tcp-z-mm",
        type=float,
        default=DEFAULT_MIN_TARGET_TCP_Z_MM,
        help="Minimum allowed final TCP z in JAKA base frame, in millimeters.",
    )
    parser.add_argument(
        "--grasp-outlier-radius-mm",
        type=float,
        default=80.0,
        help="Maximum center distance for two candidates to belong to the same cluster.",
    )
    parser.add_argument(
        "--grasp-outlier-min-cluster-size",
        type=int,
        default=3,
        help="Do not filter unless the largest center cluster has at least this many candidates.",
    )
    parser.add_argument(
        "--filter-grasp-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GraspNet model-free collision detection before saving candidate grasps.",
    )
    parser.add_argument(
        "--grasp-collision-voxel-size",
        type=float,
        default=0.005,
        help="Voxel size in meters for model-free collision detection scene downsampling.",
    )
    parser.add_argument(
        "--grasp-collision-approach-dist",
        type=float,
        default=0.05,
        help="Approach-path distance in meters checked by model-free collision detection.",
    )
    parser.add_argument(
        "--grasp-collision-thresh",
        type=float,
        default=0.05,
        help="Global collision IoU threshold; larger values keep more candidates. Matches GraspNet's common model-free default.",
    )
    parser.add_argument(
        "--filter-empty-grasps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also remove candidates whose inner grasp region contains too few scene points.",
    )
    parser.add_argument(
        "--empty-grasp-thresh",
        type=float,
        default=0.01,
        help="Inner occupancy threshold used only when --filter-empty-grasps is enabled.",
    )
    parser.add_argument(
        "--camera-coordinates-dir",
        default=str(DEFAULT_CAMERA_COORDINATES_DIR),
        help="Directory containing d435_camera.json, extrinsics.npy, and pixel2world.py calibration.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["legacy_plate", "hand_eye"],
        default="legacy_plate",
        help="legacy_plate uses old board/plate calibration; hand_eye uses T_base_tcp_capture @ T_tcp_camera.",
    )
    parser.add_argument(
        "--hand-eye-calibration",
        default=str(DEFAULT_HAND_EYE_CALIBRATION),
        help="JSON containing T_tcp_camera from solve_handeye_chessboard.py.",
    )
    parser.add_argument(
        "--tcp-camera-translation-offset-mm",
        type=float,
        nargs=3,
        default=DEFAULT_TCP_CAMERA_TRANSLATION_OFFSET_MM,
        metavar=("DX", "DY", "DZ"),
        help=(
            "XYZ offset in TCP frame added to T_tcp_camera translation before converting grasps. "
            "Default is the chessboard height validated correction."
        ),
    )
    parser.add_argument(
        "--capture-tcp-pose",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="TCP pose at capture time for hand-eye mode, in mm + xyz Euler radians.",
    )
    parser.add_argument("--execute", action="store_true", help="Move JAKA to the selected candidate after capture/inference.")
    parser.add_argument("--skip-ready", action="store_true", help="Do not move to the capture joint pose/open gripper before capture.")
    parser.add_argument(
        "--ready-pose",
        type=float,
        nargs=6,
        default=DEFAULT_READY_POSE,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Legacy Cartesian ready pose, kept for compatibility; capture now defaults to --capture-joint-pose-deg.",
    )
    parser.add_argument(
        "--capture-joint-pose-deg",
        type=float,
        nargs=6,
        default=DEFAULT_CAPTURE_JOINT_POSE_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Default JAKA joint pose before every new camera capture, in degrees.",
    )
    parser.add_argument(
        "--place-target-joint-pose-deg",
        type=float,
        nargs=6,
        default=DEFAULT_PLACE_TARGET_JOINT_POSE_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Post-grasp placement target joint pose in degrees before opening the gripper.",
    )
    parser.add_argument(
        "--place-release-lower-mm",
        type=float,
        default=DEFAULT_PLACE_RELEASE_LOWER_MM,
        help="Move the TCP down along robot-base Z by this distance after reaching the place pose and before opening.",
    )
    parser.add_argument("--jaka-ip", default=DEFAULT_JAKA_IP, help="JAKA controller IP.")
    parser.add_argument("--robotiq-port", default=DEFAULT_ROBOTIQ_PORT, help="Robotiq serial port.")
    parser.add_argument("--gripper-open-force", type=int, default=DEFAULT_GRIPPER_OPEN_FORCE, help="Robotiq force used when opening.")
    parser.add_argument("--gripper-close-force", type=int, default=DEFAULT_GRIPPER_CLOSE_FORCE, help="Robotiq force used when closing.")
    parser.add_argument(
        "--jaka-executor",
        choices=["subprocess", "direct"],
        default="subprocess",
        help="Run JAKA in a separate Python process by default, so GraspNet can stay in smartgrasp.",
    )
    parser.add_argument(
        "--persistent-jaka-worker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep one JAKA subprocess alive across loop cycles instead of reconnecting for every motion sequence.",
    )
    parser.add_argument(
        "--jaka-python",
        default=DEFAULT_JAKA_PYTHON,
        help="Python executable for --jaka-executor subprocess. Defaults to JAKA_PYTHON or smartgrasp310.",
    )
    parser.add_argument(
        "--jkrc-dir",
        default=str(JKRC_DIR),
        help="Directory containing the controller-compatible jkrc.so and libjakaAPI.so for the JAKA subprocess.",
    )
    parser.add_argument("--jaka-worker", default=str(JAKA_WORKER), help="Path to jaka_motion_worker.py for subprocess execution.")
    parser.add_argument("--vendor-dir", default=str(VENDOR_DIR), help="Directory containing local gripper vendor modules.")
    parser.add_argument("--candidate-index", type=int, default=0, help="Candidate index to execute.")
    parser.add_argument(
        "--prefer-topdown-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Jointly rank all safety-filtered candidates by mask geometry quality and vertical approach.",
    )
    parser.add_argument(
        "--geometry-score-weight",
        type=float,
        default=DEFAULT_GEOMETRY_SCORE_WEIGHT,
        help="Composite-ranking weight for width/centering quality; the remaining weight is vertical approach.",
    )
    parser.add_argument(
        "--width-quality-weight",
        type=float,
        default=DEFAULT_WIDTH_QUALITY_WEIGHT,
        help="Relative weight of point-cloud width consistency inside the geometry score.",
    )
    parser.add_argument(
        "--centering-quality-weight",
        type=float,
        default=DEFAULT_CENTERING_QUALITY_WEIGHT,
        help="Relative weight of centering quality inside the geometry score.",
    )
    parser.add_argument(
        "--topdown-rerank-window",
        type=int,
        default=10,
        help="Deprecated compatibility option; composite ranking now evaluates every surviving candidate.",
    )
    parser.add_argument("--velocity", type=float, default=60.0, help="JAKA linear_move_extend velocity.")
    parser.add_argument("--acceleration", type=float, default=60.0, help="JAKA linear_move_extend acceleration.")
    parser.add_argument("--joint-velocity-rad-s", type=float, default=DEFAULT_JOINT_VELOCITY_RAD_S, help="JAKA joint_move velocity in rad/s.")
    parser.add_argument(
        "--grasp-center-to-tcp-offset-mm",
        type=float,
        default=DEFAULT_GRASP_CENTER_TO_TCP_OFFSET_MM,
        help=(
            "Distance from GraspNet grasp center back to the physical Robotiq TCP, "
            "applied along -TCP local Z after mapping GraspNet grasp frame to the JAKA TCP frame."
        ),
    )
    parser.add_argument(
        "--gripper-roll-offset-deg",
        type=float,
        default=DEFAULT_GRIPPER_ROLL_OFFSET_DEG,
        help="Fixed rotation around TCP local Z to align the physical gripper opening direction.",
    )
    parser.add_argument("--approach-offset-mm", type=float, default=80.0, help="Pre-grasp retreat along TCP local Z.")
    parser.add_argument(
        "--grasp-extra-depth-mm",
        type=float,
        default=DEFAULT_GRASP_EXTRA_DEPTH_MM,
        help=(
            "Signed offset from the planned grasp target along TCP local Z: "
            "positive moves along +Z and negative moves along -Z."
        ),
    )
    parser.add_argument("--lift-mm", type=float, default=170.0, help="Post-close vertical lift in robot base frame.")
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=1,
        help="Number of capture-grasp-place cycles to run. Use 0 for an infinite loop.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run capture-grasp-place cycles until interrupted. Equivalent to --num-cycles 0.",
    )
    parser.add_argument(
        "--data-realworld",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save captured RGB-D to data_realworld/<timestamp>/ instead of --output-dir.",
    )
    parser.add_argument(
        "--perception-mask",
        default=None,
        help="Path to a pre-computed SAM2 mask PNG (from perception pipeline). Skips interactive SAM when provided.",
    )
    return parser


def run_one_cycle(
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_path: Path,
    camera_coordinates_dir: Path,
    cycle_index: int,
) -> dict[str, Any]:
    trial_dir: Path | None = None
    try:
        # --- resolve scene directory (data_realworld or legacy output-dir) ---
        if args.data_realworld and not args.reuse_capture:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            scene_dir = DATA_REALWORLD_ROOT / timestamp
            input_dir = scene_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir = input_dir
            print(f"[data_realworld] scene_id={timestamp}", flush=True)
        else:
            output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.instruction:
            instruction_path = output_dir / "instruction.txt"
            instruction_path.write_text(args.instruction.strip(), encoding="utf-8")
            print(f"[capture] instruction={args.instruction[:60]}... -> {instruction_path}", flush=True)

        if args.calibration_mode == "hand_eye":
            calibration = load_hand_eye_calibration(
                Path(args.hand_eye_calibration).expanduser().resolve(),
                args.tcp_camera_translation_offset_mm,
            )
        else:
            calibration = load_legacy_plate_calibration(camera_coordinates_dir, DEFAULT_PLATE_TO_ROBOT_MM)

        capture_was_reused = bool(args.reuse_capture)
        if not capture_was_reused:
            prepare_robot(args)
        frame = (
            load_captured_frame(output_dir)
            if capture_was_reused
            else capture_realsense(output_dir, args.warmup_frames, args.camera_serial, args.camera_index)
        )
        if calibration["mode"] == "hand_eye":
            capture_tcp_pose = resolve_capture_tcp_pose(output_dir, args, capture_was_reused)
            calibration["capture_tcp_pose"] = capture_tcp_pose
            calibration["base_from_tcp_capture"] = jaka_pose_to_transform(capture_tcp_pose)
        if args.capture_only:
            depth_cm = (
                np.asarray(frame["depth_raw"], dtype=np.float32)
                * np.float32(frame["meta"]["depth_scale_m"] * 100.0)
            )
            depth_npy_path = output_dir / "depth.npy"
            np.save(depth_npy_path, depth_cm)
            summary = {
                "output_dir": str(output_dir),
                "rgb": str(output_dir / "rgb.png"),
                "depth_raw": str(output_dir / "depth.raw"),
                "depth_npy": str(depth_npy_path),
                "camera_meta": str(output_dir / "camera_meta.json"),
                "capture_tcp_pose": calibration.get("capture_tcp_pose"),
                "camera_index": args.camera_index,
                "camera_serial": args.camera_serial,
                "calibration_mode": args.calibration_mode,
                "capture_only": True,
                "executed": False,
            }
            print(json.dumps(summary, indent=2))
            return summary
        full_cloud, grasp_cloud, full_cloud_rgb, grasp_cloud_rgb, obstacle_cloud, point_cloud_info = build_grasp_point_cloud(
            frame,
            output_dir,
            args,
        )
        if point_cloud_info.get("mask_path") is not None:
            trial_dir = create_trial_log_dir(args, output_dir)
        cloud_sampled = sample_points(grasp_cloud, args.num_points)
        np.save(output_dir / "point_cloud_camera.npy", full_cloud.astype(np.float32, copy=False))
        np.save(output_dir / "point_cloud_camera_rgb.npy", full_cloud_rgb.astype(np.uint8, copy=False))

        grasps = run_graspnet(cloud_sampled, checkpoint_path, args.device)
        candidate_source = "graspnet"
        pca_fallback_info = None
        if len(grasps) == 0 and args.if_pca:
            object_cloud_path = point_cloud_info.get("object_point_cloud_path")
            if object_cloud_path is None:
                raise ValueError("--if-pca requires --use-sam-mask and its masked object point cloud")
            grasps, pca_fallback_info = build_pca_fallback_grasp(np.load(object_cloud_path))
            candidate_source = "pca_fallback"
        records = save_outputs(
            output_dir,
            grasp_cloud,
            grasp_cloud_rgb,
            grasps,
            args.top_k,
            calibration,
            args,
            point_cloud_info,
            ply_cloud=full_cloud,
            ply_cloud_rgb=full_cloud_rgb,
            obstacle_cloud=obstacle_cloud,
            candidate_source=candidate_source,
            pca_fallback_info=pca_fallback_info,
        )

        summary = {
            "output_dir": str(output_dir),
            "rgb": str(output_dir / "rgb.png"),
            "depth_raw": str(output_dir / "depth.raw"),
            "point_cloud": str(output_dir / "point_cloud_camera.npy"),
            "grasp_point_cloud_source": point_cloud_info.get("grasp_point_cloud_source"),
            "mask_png": point_cloud_info.get("mask_path"),
            "mask_overlay_png": point_cloud_info.get("mask_overlay_path"),
            "grasp_crop_overlay_png": point_cloud_info.get("grasp_crop_overlay_path"),
            "object_point_cloud": point_cloud_info.get("object_point_cloud_path"),
            "grasp_input_point_cloud": point_cloud_info.get("grasp_input_point_cloud_path"),
            "obstacle_point_cloud": point_cloud_info.get("obstacle_point_cloud_path"),
            "num_grasp_input_points": point_cloud_info.get("num_grasp_input_points"),
            "num_object_points": point_cloud_info.get("num_object_points"),
            "num_grasp_crop_points": point_cloud_info.get("num_grasp_crop_points"),
            "num_obstacle_points": point_cloud_info.get("num_obstacle_points"),
            "grasp_input_mode": point_cloud_info.get("grasp_input_mode"),
            "grasp_candidates_png": str(output_dir / "grasp_candidates.png"),
            "grasp_candidates_ply": str(output_dir / "grasp_candidates.ply"),
            "grasp_candidates_3d_html": str(output_dir / "grasp_candidates_3d.html"),
            "grasp_candidates_json": str(output_dir / "grasp_candidates.json"),
            "num_candidates": len(records),
            "candidate_source": candidate_source,
            "pca_fallback_used": candidate_source == "pca_fallback",
            "calibration_mode": args.calibration_mode,
            "camera_coordinates_dir": str(camera_coordinates_dir),
            "hand_eye_calibration": str(Path(args.hand_eye_calibration).expanduser().resolve()),
            "tcp_camera_translation_offset_mm": (
                calibration.get("translation_offset_mm")
                if calibration is not None and calibration["mode"] == "hand_eye"
                else None
            ),
            "capture_tcp_pose": calibration.get("capture_tcp_pose"),
            "camera_index": args.camera_index,
            "camera_serial": args.camera_serial,
            "ready_pose": args.ready_pose,
            "capture_joint_pose_deg": args.capture_joint_pose_deg,
            "place_target_joint_pose_deg": args.place_target_joint_pose_deg,
            "cycle_index": int(cycle_index),
            "executed": bool(args.execute),
            "execution_status": "not_requested",
            "trial_log_dir": str(trial_dir) if trial_dir is not None else None,
        }
        save_trial_log_files(trial_dir, output_dir, args, summary)

        if args.execute:
            if args.candidate_index < 0 or args.candidate_index >= len(records):
                summary["execution_status"] = "invalid_candidate_index"
                update_trial_run_info(trial_dir, args, output_dir, summary)
                raise ValueError(f"--candidate-index out of range: {args.candidate_index}")
            summary["execution_status"] = "started"
            summary["executed_candidate_index"] = args.candidate_index
            update_trial_run_info(trial_dir, args, output_dir, summary)
            execute_grasp_sequence(records[args.candidate_index], args)
            summary["execution_status"] = "completed"
            update_trial_run_info(trial_dir, args, output_dir, summary)

        print(json.dumps(summary, indent=2))
        return summary
    except BaseException as exc:
        if trial_dir is None:
            raise
        failure_summary = {
            "output_dir": str(output_dir),
            "trial_log_dir": str(trial_dir) if trial_dir is not None else None,
            "cycle_index": int(cycle_index),
            "executed": bool(args.execute),
            "execution_status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error": repr(exc),
        }
        update_trial_run_info(trial_dir, args, output_dir, failure_summary)
        save_trial_log_files(trial_dir, output_dir, args, failure_summary)
        raise


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.loop:
        args.num_cycles = 0
    if args.num_cycles < 0:
        raise ValueError("--num-cycles must be >= 0")
    if args.grasp_width_min_contact_points < 1:
        raise ValueError("--grasp-width-min-contact-points must be >= 1")
    if args.grasp_closing_min_points < 1:
        raise ValueError("--grasp-closing-min-points must be >= 1")
    if not 0 <= args.grasp_closing_min_input_ratio <= 1:
        raise ValueError("--grasp-closing-min-input-ratio must be between 0 and 1")
    if not 0 <= args.grasp_width_percentile_low < args.grasp_width_percentile_high <= 100:
        raise ValueError("grasp width percentiles must satisfy 0 <= low < high <= 100")
    if args.grasp_width_tolerance_mm < 0:
        raise ValueError("--grasp-width-tolerance-mm must be >= 0")
    if args.grasp_width_max_center_offset_ratio < 0:
        raise ValueError("--grasp-width-max-center-offset-ratio must be >= 0")
    if args.grasp_filter_finger_length_mm <= 0:
        raise ValueError("--grasp-filter-finger-length-mm must be > 0")
    if args.grasp_filter_finger_width_mm <= 0:
        raise ValueError("--grasp-filter-finger-width-mm must be > 0")
    if not 0 <= args.geometry_score_weight <= 1:
        raise ValueError("--geometry-score-weight must be between 0 and 1")
    if args.width_quality_weight < 0 or args.centering_quality_weight < 0:
        raise ValueError("geometry component weights must be >= 0")
    if args.width_quality_weight + args.centering_quality_weight <= 0:
        raise ValueError("at least one geometry component weight must be > 0")
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    camera_coordinates_dir = Path(args.camera_coordinates_dir).expanduser().resolve()

    needs_robot_session = args.execute or (not args.skip_ready and not args.reuse_capture)
    worker_context = (
        PersistentJakaWorker(args, Path(args.jaka_worker).expanduser(), JAKA_WORKER_READY_PREFIX, JAKA_WORKER_RESPONSE_PREFIX)
        if needs_robot_session and args.jaka_executor == "subprocess" and args.persistent_jaka_worker
        else None
    )
    if worker_context is None:
        run_loop(args, output_dir, checkpoint_path, camera_coordinates_dir)
        return

    with worker_context as worker:
        args._persistent_jaka_worker = worker
        try:
            run_loop(args, output_dir, checkpoint_path, camera_coordinates_dir)
        finally:
            if hasattr(args, "_persistent_jaka_worker"):
                delattr(args, "_persistent_jaka_worker")


def run_loop(
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_path: Path,
    camera_coordinates_dir: Path,
) -> None:
    cycle_index = 1
    while args.num_cycles == 0 or cycle_index <= args.num_cycles:
        total_text = "infinite" if args.num_cycles == 0 else str(args.num_cycles)
        print(f"[loop] starting cycle {cycle_index}/{total_text}", flush=True)
        run_one_cycle(args, output_dir, checkpoint_path, camera_coordinates_dir, cycle_index)
        if args.num_cycles != 0 and cycle_index >= args.num_cycles:
            break
        print(f"[loop] completed cycle {cycle_index}; returning to next capture cycle", flush=True)
        cycle_index += 1


if __name__ == "__main__":
    main()
