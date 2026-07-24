"""GraspNet inference, calibration, and candidate filtering helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode
from utils.camera import camera_point_to_pixel
from utils.collision_detector import ModelFreeCollisionDetector


def load_graspnet(checkpoint_path: Path, device: torch.device) -> GraspNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()
    return net


def run_graspnet(cloud_sampled: np.ndarray, checkpoint_path: Path, device_name: str) -> GraspGroup:
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    net = load_graspnet(checkpoint_path, device)
    cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        end_points = net({"point_clouds": cloud_tensor})
        grasp_preds = pred_decode(end_points)
    grasps = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    grasps.sort_by_score()
    if len(grasps) == 0:
        print("[graspnet] returned no grasp candidates.", flush=True)
    return grasps


def build_pca_fallback_grasp(object_cloud: np.ndarray) -> tuple[GraspGroup, dict[str, Any]]:
    """Build one conservative parallel-jaw grasp from a masked object cloud."""
    points = np.asarray(object_cloud, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"PCA fallback expects an Nx3 object cloud, got {points.shape}")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 30:
        raise ValueError(f"PCA fallback needs at least 30 finite object points, got {len(points)}")

    # Remove the farthest mask/depth outliers before estimating the object axes.
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)
    distance_limit = float(np.percentile(distances, 95.0))
    inlier_points = points[distances <= distance_limit]
    if len(inlier_points) < 30:
        raise ValueError(f"PCA fallback retained too few inlier points: {len(inlier_points)}")

    center = np.mean(inlier_points, axis=0)
    covariance = np.cov(inlier_points - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[1] < 1e-10:
        raise ValueError(f"PCA fallback object cloud is geometrically degenerate: {eigenvalues.tolist()}")

    # GraspNet uses local X as approach and local Y as finger-opening direction.
    # The least-variance PCA axis approximates the visible surface normal; the
    # middle axis gives a short in-plane direction suitable for closing fingers.
    approach_axis = eigenvectors[:, 2]
    if float(np.dot(approach_axis, center)) < 0.0:
        approach_axis = -approach_axis
    opening_axis = eigenvectors[:, 1]
    opening_axis -= np.dot(opening_axis, approach_axis) * approach_axis
    opening_axis = normalized(opening_axis, "PCA opening axis")
    closing_plane_axis = normalized(np.cross(approach_axis, opening_axis), "PCA closing-plane axis")
    opening_axis = normalized(np.cross(closing_plane_axis, approach_axis), "orthogonal PCA opening axis")
    rotation = np.column_stack([approach_axis, opening_axis, closing_plane_axis])

    opening_coordinates = (inlier_points - center) @ opening_axis
    lower, upper = np.percentile(opening_coordinates, [2.5, 97.5])
    object_opening_extent = float(upper - lower)
    grasp_width = float(np.clip(object_opening_extent * 1.15, 0.02, 0.10))
    grasp_height = 0.02
    grasp_depth = 0.02
    grasp_array = np.concatenate(
        [
            np.array([0.0, grasp_width, grasp_height, grasp_depth], dtype=float),
            rotation.reshape(-1),
            center,
            np.array([-1.0], dtype=float),
        ]
    ).reshape(1, 17)
    info = {
        "method": "masked_object_cloud_pca",
        "num_input_points": int(len(points)),
        "num_inlier_points": int(len(inlier_points)),
        "outlier_distance_percentile": 95.0,
        "eigenvalues_m2": eigenvalues.tolist(),
        "center_camera_m": center.tolist(),
        "approach_axis_camera": approach_axis.tolist(),
        "opening_axis_camera": opening_axis.tolist(),
        "object_opening_extent_m": object_opening_extent,
        "grasp_width_m": grasp_width,
        "grasp_height_m": grasp_height,
        "grasp_depth_m": grasp_depth,
    }
    print(
        "[pca-fallback] generated one grasp from the masked object cloud: "
        f"points={len(inlier_points)} width={grasp_width:.4f}m "
        f"center={np.round(center, 4).tolist()}",
        flush=True,
    )
    return GraspGroup(grasp_array), info


def grasp_to_record(grasp, index: int) -> dict[str, Any]:
    return {
        "grasp_index": index,
        "score": float(grasp.score),
        "width": float(grasp.width),
        "height": float(grasp.height),
        "depth": float(grasp.depth),
        "translation_camera_m": grasp.translation.astype(float),
        "rotation_camera": grasp.rotation_matrix.astype(float),
        "object_id": int(grasp.object_id),
    }


def transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def jaka_pose_to_transform(pose: list[float]) -> np.ndarray:
    if len(pose) != 6:
        raise ValueError(f"JAKA TCP pose must be 6D, got {pose!r}")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    transform[:3, 3] = np.asarray(pose[:3], dtype=float)
    return transform


def load_legacy_plate_calibration(camera_coordinates_dir: Path, plate_to_robot_mm: np.ndarray) -> dict[str, Any]:
    """Load the checked chessboard calibration used by pixel2world.py.

    `extrinsics.npy` is T_board_to_camera in millimeters. The hard-coded
    T_obj_plate2Robot from pixel2world.py is T_plate_to_robot in millimeters.
    """
    extrinsics_path = camera_coordinates_dir / "extrinsics.npy"
    intrinsics_path = camera_coordinates_dir / "d435_camera.json"
    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Missing extrinsics.npy: {extrinsics_path}")
    board_to_camera = np.load(extrinsics_path).astype(float).reshape(4, 4)
    intrinsics = None
    if intrinsics_path.exists():
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    return {
        "mode": "legacy_plate",
        "board_to_camera": board_to_camera,
        "plate_to_robot": np.asarray(plate_to_robot_mm, dtype=float).copy(),
        "intrinsics": intrinsics,
        "source_dir": str(camera_coordinates_dir),
    }


def load_hand_eye_calibration(calibration_path: Path, translation_offset_mm: list[float] | None = None) -> dict[str, Any]:
    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing hand-eye calibration: {calibration_path}")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if "T_tcp_camera" not in payload:
        raise ValueError(f"Hand-eye calibration must contain T_tcp_camera: {calibration_path}")
    tcp_from_camera = np.asarray(payload["T_tcp_camera"], dtype=float).reshape(4, 4)
    offset = np.zeros(3, dtype=float) if translation_offset_mm is None else np.asarray(translation_offset_mm, dtype=float).reshape(3)
    tcp_from_camera[:3, 3] += offset
    return {
        "mode": "hand_eye",
        "tcp_from_camera": tcp_from_camera,
        "runtime_transform": "T_tcp_camera",
        "runtime_chain": "T_base_tcp_capture @ T_tcp_camera @ T_camera_grasp",
        "source_path": str(calibration_path),
        "translation_offset_mm": offset.tolist(),
        "raw_tcp_from_camera_translation_mm": np.asarray(payload["T_tcp_camera"], dtype=float).reshape(4, 4)[:3, 3].tolist(),
        "tcp_from_camera_translation_mm": tcp_from_camera[:3, 3].tolist(),
        "payload": payload,
    }


def transform_to_jaka_pose(transform: np.ndarray) -> list[float]:
    # Calibration matrices from pixel2world.py are millimeter based, so the
    # robot-frame transform translation is already in JAKA's millimeter unit.
    position_mm = transform[:3, 3]
    euler = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=False)
    return [float(position_mm[0]), float(position_mm[1]), float(position_mm[2]), float(euler[0]), float(euler[1]), float(euler[2])]


def camera_grasp_to_robot_transform(record: dict[str, Any], calibration: dict[str, Any]) -> np.ndarray:
    # GraspNet outputs a grasp pose in the RealSense color camera frame:
    # T_camera_grasp. Hand-eye calibration outputs T_tcp_camera, so the
    # eye-in-hand chain is T_base_tcp_capture @ T_tcp_camera @ T_camera_grasp.
    camera_from_grasp_m = transform_from_rotation_translation(
        record["rotation_camera"],
        record["translation_camera_m"],
    )
    camera_from_grasp_mm = camera_from_grasp_m.copy()
    camera_from_grasp_mm[:3, 3] *= 1000.0
    if calibration["mode"] == "hand_eye":
        return calibration["base_from_tcp_capture"] @ calibration["tcp_from_camera"] @ camera_from_grasp_mm
    return calibration["plate_to_robot"] @ np.linalg.inv(calibration["board_to_camera"]) @ camera_from_grasp_mm


def normalized(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {name}")
    return np.asarray(vector, dtype=float) / norm


def build_grasp_to_tcp_rotation(calibration: dict[str, Any], base_gripper_opening_axis: np.ndarray) -> np.ndarray:
    if "grasp_to_tcp_rotation" in calibration:
        return np.asarray(calibration["grasp_to_tcp_rotation"], dtype=float).reshape(3, 3)
    if calibration.get("mode") != "hand_eye" or "base_from_tcp_capture" not in calibration:
        return np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    capture_tcp_rotation = np.asarray(calibration["base_from_tcp_capture"], dtype=float).reshape(4, 4)[:3, :3]
    tcp_z_base = normalized(capture_tcp_rotation[:, 2], "capture TCP local Z")
    opening_axis_base = base_gripper_opening_axis - np.dot(base_gripper_opening_axis, tcp_z_base) * tcp_z_base
    opening_axis_base = normalized(opening_axis_base, "projected gripper opening axis")
    tcp_x_base = normalized(np.cross(opening_axis_base, tcp_z_base), "derived TCP local X")
    ideal_tcp_rotation = np.column_stack([tcp_x_base, opening_axis_base, tcp_z_base])
    ideal_grasp_rotation = np.column_stack([tcp_z_base, opening_axis_base, -tcp_x_base])
    grasp_to_tcp_rotation = ideal_grasp_rotation.T @ ideal_tcp_rotation
    calibration["grasp_to_tcp_rotation"] = grasp_to_tcp_rotation
    calibration["grasp_to_tcp_rotation_convention"] = (
        "derived_from_capture_pose: grasp_x=tcp_z, grasp_y=base_y_projected_to_tcp_xy"
    )
    calibration["gripper_opening_axis_base"] = base_gripper_opening_axis.tolist()
    calibration["capture_tcp_local_z_base"] = tcp_z_base.tolist()
    calibration["capture_gripper_opening_axis_base"] = opening_axis_base.tolist()
    return grasp_to_tcp_rotation


def grasp_center_to_jaka_tcp_transform(robot_from_grasp: np.ndarray, offset_mm: float, grasp_to_tcp_rotation: np.ndarray, translation_offset_mm: list[float] | np.ndarray | None = None) -> np.ndarray:
    """Convert a GraspNet grasp-center frame into the physical JAKA TCP frame."""
    robot_from_tcp = robot_from_grasp.copy()
    robot_from_tcp[:3, :3] = robot_from_grasp[:3, :3] @ grasp_to_tcp_rotation
    tcp_axis = robot_from_tcp[:3, 2]
    robot_from_tcp[:3, 3] = robot_from_grasp[:3, 3] - tcp_axis * float(offset_mm)
    if translation_offset_mm is not None:
        robot_from_tcp[:3, 3] += np.asarray(translation_offset_mm, dtype=float).reshape(3)
    return robot_from_tcp


def compute_robot_targets(
    records: list[dict[str, Any]],
    calibration: dict[str, Any] | None,
    grasp_center_to_tcp_offset_mm: float = 0.0,
    gripper_roll_offset_deg: float = 0.0,
    base_gripper_opening_axis: np.ndarray | None = None,
    tcp_target_translation_offset_mm: list[float] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if calibration is None:
        return records
    if base_gripper_opening_axis is None:
        base_gripper_opening_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    grasp_to_tcp_rotation = build_grasp_to_tcp_rotation(calibration, np.asarray(base_gripper_opening_axis, dtype=float))
    roll_offset_rotation = Rotation.from_euler("z", float(gripper_roll_offset_deg), degrees=True).as_matrix()
    calibration["base_grasp_to_tcp_rotation"] = grasp_to_tcp_rotation
    grasp_to_tcp_rotation = grasp_to_tcp_rotation @ roll_offset_rotation
    calibration["grasp_to_tcp_rotation_with_roll_offset"] = grasp_to_tcp_rotation
    calibration["gripper_roll_offset_deg"] = float(gripper_roll_offset_deg)
    for record in records:
        robot_from_grasp = camera_grasp_to_robot_transform(record, calibration)
        robot_from_tcp = grasp_center_to_jaka_tcp_transform(
            robot_from_grasp,
            grasp_center_to_tcp_offset_mm,
            grasp_to_tcp_rotation,
            tcp_target_translation_offset_mm,
        )
        record["grasp_center_jaka_pose"] = transform_to_jaka_pose(robot_from_grasp)
        record["target_jaka_tcp_pose"] = transform_to_jaka_pose(robot_from_tcp)
        record["target_robot_from_grasp"] = robot_from_grasp
        record["target_robot_from_grasp_center"] = robot_from_grasp
        record["target_robot_from_tcp"] = robot_from_tcp
        record["grasp_center_to_tcp_offset_mm"] = float(grasp_center_to_tcp_offset_mm)
        record["grasp_center_to_tcp_offset_axis"] = "-tcp_local_z"
        record["grasp_to_tcp_rotation"] = grasp_to_tcp_rotation
        record["gripper_roll_offset_deg"] = float(gripper_roll_offset_deg)
        record["grasp_to_tcp_rotation_convention"] = calibration.get(
            "grasp_to_tcp_rotation_convention",
            "fallback_static: tcp_z=grasp_x,tcp_y=grasp_y,tcp_x=-grasp_z",
        )
        record["calibration_mode"] = calibration["mode"]
        record["calibration_source"] = calibration.get("source_path") or calibration.get("source_dir")
        if calibration["mode"] == "hand_eye":
            record["capture_tcp_pose"] = calibration["capture_tcp_pose"]
            record["hand_eye_runtime_transform"] = calibration["runtime_transform"]
    return records


def candidate_center_mm(record: dict[str, Any]) -> np.ndarray:
    if "target_robot_from_grasp" in record:
        return np.asarray(record["target_robot_from_grasp"], dtype=float).reshape(4, 4)[:3, 3]
    return np.asarray(record["translation_camera_m"], dtype=float).reshape(3) * 1000.0


def target_tcp_z_mm(record: dict[str, Any]) -> float | None:
    if "target_robot_from_tcp" in record:
        return float(np.asarray(record["target_robot_from_tcp"], dtype=float).reshape(4, 4)[2, 3])
    if "target_jaka_tcp_pose" in record:
        return float(record["target_jaka_tcp_pose"][2])
    return None


def filter_target_tcp_z(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": bool(args.filter_target_tcp_z),
        "method": "min_target_tcp_z",
        "min_z_mm": float(args.min_target_tcp_z_mm),
        "num_input_candidates": int(len(records)),
        "num_removed": 0,
        "removed": [],
    }
    if not args.filter_target_tcp_z or len(records) == 0:
        info["reason"] = "disabled_or_no_candidates"
        return records, info

    filtered_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    for record in records:
        z_mm = target_tcp_z_mm(record)
        kept = z_mm is not None and z_mm >= float(args.min_target_tcp_z_mm)
        record["target_tcp_z_filter"] = {
            "target_tcp_z_mm": z_mm,
            "min_z_mm": float(args.min_target_tcp_z_mm),
            "kept": bool(kept),
        }
        if kept:
            filtered_records.append(record)
        else:
            removed_records.append(record)

    info["num_removed"] = int(len(removed_records))
    info["removed"] = [
        {
            "raw_grasp_index": int(record.get("grasp_index", -1)),
            "score": float(record["score"]),
            "target_tcp_z_mm": record["target_tcp_z_filter"]["target_tcp_z_mm"],
            "min_z_mm": record["target_tcp_z_filter"]["min_z_mm"],
        }
        for record in removed_records
    ]
    if removed_records:
        removed_text = []
        for item in info["removed"]:
            z_value = item["target_tcp_z_mm"]
            z_text = "unavailable" if z_value is None else f"{z_value:.3f}mm"
            removed_text.append(
                f"raw_grasp_{item['raw_grasp_index']} "
                f"tcp_z={z_text} < {item['min_z_mm']:.3f}mm"
            )
        print(
            "[candidate-filter] removed low TCP-z candidates: "
            + ", ".join(removed_text),
            flush=True,
        )
    return filtered_records, info


def tcp_downward_angle_deg(record: dict[str, Any]) -> float | None:
    if "target_robot_from_tcp" not in record:
        return None
    transform = np.asarray(record["target_robot_from_tcp"], dtype=float).reshape(4, 4)
    tcp_z_axis = normalized(transform[:3, 2], "target TCP local Z")
    downward_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    cosine = float(np.clip(np.dot(tcp_z_axis, downward_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rerank_candidates_by_topdown(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": bool(args.prefer_topdown_candidate),
        "method": "min_tcp_local_z_angle_to_base_down",
        "window_size": int(args.topdown_rerank_window),
        "num_input_candidates": int(len(records)),
        "reranked": [],
    }
    if not args.prefer_topdown_candidate or len(records) == 0:
        info["reason"] = "disabled_or_no_candidates"
        return records, info

    window_size = max(1, min(int(args.topdown_rerank_window), len(records)))
    window = records[:window_size]
    tail = records[window_size:]
    for rank, record in enumerate(window):
        angle_deg = tcp_downward_angle_deg(record)
        record["topdown_rerank"] = {
            "input_rank": int(rank),
            "tcp_local_z_angle_to_base_down_deg": angle_deg,
        }

    reranked_window = sorted(
        window,
        key=lambda record: (
            float("inf") if record["topdown_rerank"]["tcp_local_z_angle_to_base_down_deg"] is None
            else float(record["topdown_rerank"]["tcp_local_z_angle_to_base_down_deg"]),
            -float(record["score"]),
        ),
    )
    for output_rank, record in enumerate(reranked_window):
        record["topdown_rerank"]["output_rank"] = int(output_rank)

    info["reranked"] = [
        {
            "raw_grasp_index": int(record.get("raw_grasp_index", record.get("grasp_index", -1))),
            "input_rank": int(record["topdown_rerank"]["input_rank"]),
            "output_rank": int(record["topdown_rerank"]["output_rank"]),
            "score": float(record["score"]),
            "tcp_local_z_angle_to_base_down_deg": record["topdown_rerank"]["tcp_local_z_angle_to_base_down_deg"],
        }
        for record in reranked_window
    ]
    if reranked_window:
        print(
            "[candidate-rerank] top-down preferred order: "
            + ", ".join(
                f"raw_grasp_{item['raw_grasp_index']} "
                f"angle={item['tcp_local_z_angle_to_base_down_deg']:.3f}deg "
                f"score={item['score']:.4f}"
                for item in info["reranked"]
                if item["tcp_local_z_angle_to_base_down_deg"] is not None
            ),
            flush=True,
        )
    return reranked_window + tail, info


def filter_grasp_centers_in_target_mask(
    records: list[dict[str, Any]],
    target_mask: np.ndarray | None,
    intrinsics: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": bool(args.filter_grasp_centers_in_mask),
        "method": "project_grasp_center_to_target_mask",
        "num_input_candidates": int(len(records)),
        "num_removed": 0,
        "removed": [],
    }
    if not args.filter_grasp_centers_in_mask or len(records) == 0:
        info["reason"] = "disabled_or_no_candidates"
        return records, info
    if target_mask is None or intrinsics is None:
        info["reason"] = "missing_target_mask_or_intrinsics"
        return records, info

    mask = np.asarray(target_mask, dtype=bool)
    height, width = mask.shape
    tolerance_px = max(0.0, float(getattr(args, "target_mask_center_tolerance_px", 0.0)))
    outside_distance_px = None
    if tolerance_px > 0.0 and np.any(mask):
        outside_distance_px = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    info["tolerance_px"] = tolerance_px
    filtered_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    for record in records:
        pixel = camera_point_to_pixel(record["translation_camera_m"], intrinsics)
        kept = False
        reason = "invalid_projection"
        distance_to_mask_px: float | None = None
        if pixel is not None:
            u, v = pixel
            if 0 <= u < width and 0 <= v < height:
                if bool(mask[v, u]):
                    kept = True
                    reason = "inside_target_mask"
                    distance_to_mask_px = 0.0
                elif outside_distance_px is not None:
                    distance_to_mask_px = float(outside_distance_px[v, u])
                    kept = distance_to_mask_px <= tolerance_px
                    reason = "near_target_mask" if kept else "outside_target_mask"
                else:
                    reason = "outside_target_mask"
            else:
                reason = "projection_outside_image"
        record["target_mask_center_filter"] = {
            "pixel_uv": None if pixel is None else [int(pixel[0]), int(pixel[1])],
            "kept": bool(kept),
            "reason": reason,
            "distance_to_mask_px": distance_to_mask_px,
            "tolerance_px": tolerance_px,
        }
        if kept:
            filtered_records.append(record)
        else:
            removed_records.append(record)

    info["num_removed"] = int(len(removed_records))
    info["removed"] = [
        {
            "raw_grasp_index": int(record.get("grasp_index", -1)),
            "score": float(record["score"]),
            "pixel_uv": record["target_mask_center_filter"]["pixel_uv"],
            "reason": record["target_mask_center_filter"]["reason"],
            "distance_to_mask_px": record["target_mask_center_filter"].get("distance_to_mask_px"),
            "translation_camera_m": record["translation_camera_m"],
        }
        for record in removed_records
    ]
    if removed_records:
        print(
            "[candidate-filter] removed target-mask misses: "
            + ", ".join(
                f"raw_grasp_{item['raw_grasp_index']} "
                f"pixel={item['pixel_uv']} reason={item['reason']} distance_px={item.get('distance_to_mask_px')}"
                for item in info["removed"]
            ),
            flush=True,
        )
    return filtered_records, info


def connected_components_from_distance(centers_mm: np.ndarray, radius_mm: float) -> list[list[int]]:
    if len(centers_mm) == 0:
        return []
    distances = np.linalg.norm(centers_mm[:, None, :] - centers_mm[None, :, :], axis=-1)
    adjacency = distances <= float(radius_mm)
    visited = np.zeros(len(centers_mm), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(centers_mm)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor in np.flatnonzero(adjacency[index]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        components.append(sorted(component))
    return components


def filter_grasp_center_outliers(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": bool(args.filter_grasp_outliers),
        "method": "largest_center_cluster",
        "radius_mm": float(args.grasp_outlier_radius_mm),
        "min_cluster_size": int(args.grasp_outlier_min_cluster_size),
        "num_input_candidates": int(len(records)),
        "num_removed": 0,
        "removed": [],
    }
    if not args.filter_grasp_outliers or len(records) < 3:
        info["reason"] = "disabled_or_too_few_candidates"
        return records, info

    centers = np.stack([candidate_center_mm(record) for record in records], axis=0)
    components = connected_components_from_distance(centers, args.grasp_outlier_radius_mm)
    if not components:
        info["reason"] = "no_components"
        return records, info

    scores = np.asarray([float(record["score"]) for record in records], dtype=float)
    best_component = max(
        components,
        key=lambda component: (len(component), float(np.max(scores[component]))),
    )
    info["clusters"] = [
        {
            "candidate_indices": [int(index) for index in component],
            "size": int(len(component)),
            "max_score": float(np.max(scores[component])),
        }
        for component in components
    ]
    info["selected_cluster_indices"] = [int(index) for index in best_component]

    if len(best_component) < int(args.grasp_outlier_min_cluster_size):
        info["reason"] = "largest_cluster_smaller_than_min_cluster_size"
        return records, info

    kept_indices = set(best_component)
    filtered_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        record["center_outlier_filter"] = {
            "center_mm": centers[index].astype(float).tolist(),
            "kept": bool(index in kept_indices),
        }
        if index in kept_indices:
            filtered_records.append(record)
        else:
            removed_records.append(record)

    info["num_removed"] = int(len(removed_records))
    info["removed"] = [
        {
            "raw_grasp_index": int(record.get("grasp_index", -1)),
            "score": float(record["score"]),
            "center_mm": record["center_outlier_filter"]["center_mm"],
        }
        for record in removed_records
    ]
    if removed_records:
        print(
            "[candidate-filter] removed center outliers: "
            + ", ".join(
                f"raw_grasp_{item['raw_grasp_index']} center={np.round(item['center_mm'], 3).tolist()}"
                for item in info["removed"]
            ),
            flush=True,
        )
    return filtered_records, info


def filter_grasp_collisions(
    records: list[dict[str, Any]],
    grasp_group: GraspGroup,
    obstacle_cloud: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": bool(args.filter_grasp_collisions),
        "method": "model_free_collision_detector",
        "scene": "obstacle_point_cloud",
        "voxel_size": float(args.grasp_collision_voxel_size),
        "approach_dist": float(args.grasp_collision_approach_dist),
        "collision_thresh": float(args.grasp_collision_thresh),
        "filter_empty_grasps": bool(args.filter_empty_grasps),
        "empty_thresh": float(args.empty_grasp_thresh),
        "num_obstacle_points": int(len(obstacle_cloud)),
        "num_input_candidates": int(len(records)),
        "num_removed": 0,
        "removed": [],
    }
    if not args.filter_grasp_collisions or len(records) == 0:
        info["reason"] = "disabled_or_no_candidates"
        return records, info

    obstacle_points = np.asarray(obstacle_cloud, dtype=np.float32)
    if obstacle_points.ndim != 2 or obstacle_points.shape[1] != 3 or len(obstacle_points) == 0:
        info["reason"] = f"invalid_obstacle_cloud_shape_{obstacle_points.shape}"
        return records, info

    detector = ModelFreeCollisionDetector(obstacle_points, voxel_size=args.grasp_collision_voxel_size)
    detect_result = detector.detect(
        grasp_group,
        approach_dist=args.grasp_collision_approach_dist,
        collision_thresh=args.grasp_collision_thresh,
        return_empty_grasp=bool(args.filter_empty_grasps),
        empty_thresh=args.empty_grasp_thresh,
        return_ious=True,
    )
    if args.filter_empty_grasps:
        collision_mask, empty_mask, iou_list = detect_result
    else:
        collision_mask, iou_list = detect_result
        empty_mask = np.zeros_like(collision_mask, dtype=bool)

    iou_names = ["global", "left_finger", "right_finger", "bottom", "shifting"]
    collision_mask = np.asarray(collision_mask, dtype=bool)
    empty_mask = np.asarray(empty_mask, dtype=bool)
    remove_mask = collision_mask | empty_mask

    filtered_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        collision_info = {
            "collision": bool(collision_mask[index]),
            "empty_grasp": bool(empty_mask[index]),
            "kept": not bool(remove_mask[index]),
            "ious": {
                name: float(np.asarray(values, dtype=float)[index])
                for name, values in zip(iou_names, iou_list)
            },
        }
        record["model_free_collision_filter"] = collision_info
        if remove_mask[index]:
            removed_records.append(record)
        else:
            filtered_records.append(record)

    info["num_removed"] = int(len(removed_records))
    info["removed"] = [
        {
            "raw_grasp_index": int(record.get("grasp_index", -1)),
            "score": float(record["score"]),
            "translation_camera_m": record["translation_camera_m"],
            "collision": bool(record["model_free_collision_filter"]["collision"]),
            "empty_grasp": bool(record["model_free_collision_filter"]["empty_grasp"]),
            "ious": record["model_free_collision_filter"]["ious"],
        }
        for record in removed_records
    ]
    if removed_records:
        print(
            "[candidate-filter] removed collision candidates: "
            + ", ".join(
                f"raw_grasp_{item['raw_grasp_index']} "
                f"collision={item['collision']} empty={item['empty_grasp']} "
                f"global_iou={item['ious']['global']:.4f}"
                for item in info["removed"]
            ),
            flush=True,
        )
    return filtered_records, info


def renumber_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for new_index, record in enumerate(records):
        record["raw_grasp_index"] = int(record["grasp_index"])
        record["grasp_index"] = int(new_index)
    return records


def print_candidate_target_centers(records: list[dict[str, Any]], selected_index: int, approach_offset_mm: float) -> None:
    print("[candidate-target] final grasp centers and TCP targets in robot base frame:")
    selected_record = None
    for record in records:
        index = record["grasp_index"]
        score = record["score"]
        camera_translation = np.asarray(record["translation_camera_m"], dtype=float)
        if "target_robot_from_grasp" not in record:
            print(
                f"[candidate-target] grasp_{index}: score={score:.4f} "
                f"camera_center_m={camera_translation.tolist()} robot_center=unavailable",
                flush=True,
            )
            continue
        center_transform = np.asarray(record["target_robot_from_grasp"], dtype=float).reshape(4, 4)
        tcp_transform = np.asarray(record.get("target_robot_from_tcp", center_transform), dtype=float).reshape(4, 4)
        center_mm = center_transform[:3, 3]
        tcp_mm = tcp_transform[:3, 3]
        if index == selected_index:
            selected_record = record
        print(
            f"grasp_{index} center x={center_mm[0]:.3f} y={center_mm[1]:.3f} z={center_mm[2]:.3f} mm "
            f"tcp x={tcp_mm[0]:.3f} y={tcp_mm[1]:.3f} z={tcp_mm[2]:.3f} mm",
            flush=True,
        )
    if selected_record is None and records:
        selected_record = records[0]
    if selected_record is not None and "target_robot_from_grasp" in selected_record:
        center_transform = np.asarray(selected_record["target_robot_from_grasp"], dtype=float).reshape(4, 4)
        target_transform = np.asarray(selected_record.get("target_robot_from_tcp", center_transform), dtype=float).reshape(4, 4)
        center_mm = center_transform[:3, 3]
        tcp_mm = target_transform[:3, 3]
        moving_vector_mm = target_transform[:3, 2] * float(approach_offset_mm)
        pre_grasp_mm = tcp_mm - moving_vector_mm
        capture_tcp_pose = selected_record.get("capture_tcp_pose")
        if capture_tcp_pose is not None:
            capture_tcp_mm = np.asarray(capture_tcp_pose[:3], dtype=float)
            print(
                f"capture tcp = [{capture_tcp_mm[0]:.3f}, {capture_tcp_mm[1]:.3f}, {capture_tcp_mm[2]:.3f}] mm",
                flush=True,
            )
        print(
            f"pre grasp = [{pre_grasp_mm[0]:.3f}, {pre_grasp_mm[1]:.3f}, {pre_grasp_mm[2]:.3f}] mm",
            flush=True,
        )
        print(
            f"moving vector = [{moving_vector_mm[0]:.3f}, {moving_vector_mm[1]:.3f}, {moving_vector_mm[2]:.3f}] mm",
            flush=True,
        )
        print(
            f"grasp center = [{center_mm[0]:.3f}, {center_mm[1]:.3f}, {center_mm[2]:.3f}] mm",
            flush=True,
        )
        print(
            f"tcp target = [{tcp_mm[0]:.3f}, {tcp_mm[1]:.3f}, {tcp_mm[2]:.3f}] mm",
            flush=True,
        )
