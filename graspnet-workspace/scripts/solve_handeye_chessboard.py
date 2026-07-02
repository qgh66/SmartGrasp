#!/usr/bin/env python
"""Solve eye-in-hand chessboard calibration from collected RealSense/JAKA samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "calibration" / "handeye_chessboard_raw"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "calibration" / "hand_eye_tcp_camera.json"


def load_samples(input_dir: Path) -> list[dict[str, Any]]:
    samples_path = input_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    samples = []
    with samples_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {samples_path}:{line_no}: {exc}") from exc
            samples.append(sample)
    if not samples:
        raise RuntimeError(f"No samples found in {samples_path}")
    return samples


def make_chessboard_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    obj_points = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_points[:, :2] = grid * float(square_mm)
    return obj_points


def camera_matrix_from_intrinsics(intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intrinsics.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    return camera_matrix, dist_coeffs


def resolve_sample_path(input_dir: Path, sample: dict[str, Any], key: str) -> Path | None:
    value = sample.get(key)
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return input_dir / path


def jaka_pose_to_matrix(pose: list[float]) -> np.ndarray:
    if len(pose) != 6:
        raise ValueError(f"JAKA TCP pose must be 6D, got {pose!r}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    transform[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    return transform


def matrix_inverse(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def solve_target_to_camera(
    image_path: Path,
    intrinsics: dict[str, Any],
    obj_points: np.ndarray,
    pattern_size: tuple[int, int],
    corners_path: Path | None = None,
) -> dict[str, Any] | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    source = "detected_in_solver"
    if corners_path is not None and corners_path.exists():
        corners_payload = json.loads(corners_path.read_text(encoding="utf-8"))
        corners_array = np.asarray(corners_payload["corners_px"], dtype=np.float32)
        expected_count = int(pattern_size[0] * pattern_size[1])
        if corners_array.shape != (expected_count, 2):
            raise ValueError(f"{corners_path} contains {corners_array.shape}, expected {(expected_count, 2)}")
        refined_corners = corners_array.reshape(-1, 1, 2)
        source = "saved_by_collector"
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    camera_matrix, dist_coeffs = camera_matrix_from_intrinsics(intrinsics)
    ok, rvec, tvec = cv2.solvePnP(obj_points, refined_corners, camera_matrix, dist_coeffs)
    if not ok:
        return None

    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_error_px = float(np.linalg.norm(refined_corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1).mean())
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "R_target2cam": rotation,
        "t_target2cam": tvec.reshape(3),
        "reprojection_error_px": reprojection_error_px,
        "corner_count": int(len(refined_corners)),
        "corners_source": source,
    }


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(rotation).magnitude()))


def compute_board_consistency(used: list[dict[str, Any]], T_tcp_camera: np.ndarray) -> dict[str, Any]:
    base_from_boards = []
    for item in used:
        T_base_tcp = item["T_base_tcp"]
        T_camera_board = transform_from_rt(item["R_target2cam"], item["t_target2cam"])
        base_from_boards.append(T_base_tcp @ T_tcp_camera @ T_camera_board)

    translations = np.asarray([transform[:3, 3] for transform in base_from_boards], dtype=np.float64)
    mean_translation = translations.mean(axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)

    rotations = [transform[:3, :3] for transform in base_from_boards]
    first_rotation = rotations[0]
    rotation_errors = [rotation_angle_deg(first_rotation.T @ rotation) for rotation in rotations]
    return {
        "base_board_translation_mean_mm": mean_translation.tolist(),
        "base_board_translation_std_mm": translations.std(axis=0).tolist(),
        "base_board_translation_error_mean_mm": float(translation_errors.mean()),
        "base_board_translation_error_max_mm": float(translation_errors.max()),
        "base_board_rotation_error_mean_deg": float(np.mean(rotation_errors)),
        "base_board_rotation_error_max_deg": float(np.max(rotation_errors)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve eye-in-hand chessboard calibration.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing samples.jsonl and images.")
    parser.add_argument("--pattern-cols", type=int, default=11, help="Chessboard inner corner count along columns.")
    parser.add_argument("--pattern-rows", type=int, default=8, help="Chessboard inner corner count along rows.")
    parser.add_argument("--square-mm", type=float, default=10.0, help="Chessboard square size in millimeters.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path for T_tcp_camera.")
    parser.add_argument("--min-samples", type=int, default=12, help="Minimum detected samples required before solving.")
    parser.add_argument(
        "--method",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        default="tsai",
        help="OpenCV hand-eye calibration method.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    samples = load_samples(input_dir)
    obj_points = make_chessboard_points(args.pattern_cols, args.pattern_rows, args.square_mm)
    pattern_size = (args.pattern_cols, args.pattern_rows)

    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    used = []
    rejected = []
    for sample in samples:
        image_path = resolve_sample_path(input_dir, sample, "rgb_path")
        if image_path is None:
            rejected.append({"index": sample.get("index"), "reason": "missing_rgb_path"})
            continue
        corners_path = resolve_sample_path(input_dir, sample, "corners_path")
        solved = solve_target_to_camera(image_path, sample["intrinsics"], obj_points, pattern_size, corners_path)
        if solved is None:
            rejected.append({"index": sample.get("index"), "reason": "chessboard_not_detected", "rgb_path": sample.get("rgb_path")})
            continue
        T_base_tcp = jaka_pose_to_matrix([float(value) for value in sample["tcp_pose"]])
        used.append(
            {
                "index": int(sample.get("index", len(used))),
                "rgb_path": sample["rgb_path"],
                "T_base_tcp": T_base_tcp,
                "R_gripper2base": T_base_tcp[:3, :3],
                "t_gripper2base": T_base_tcp[:3, 3],
                **solved,
            }
        )

    if len(used) < args.min_samples:
        raise RuntimeError(
            f"Only {len(used)} samples detected chessboard, but --min-samples={args.min_samples}. "
            f"Rejected samples: {rejected}"
        )

    R_tcp_camera, t_tcp_camera = cv2.calibrateHandEye(
        [item["R_gripper2base"] for item in used],
        [item["t_gripper2base"] for item in used],
        [item["R_target2cam"] for item in used],
        [item["t_target2cam"] for item in used],
        method=methods[args.method],
    )
    T_tcp_camera = transform_from_rt(R_tcp_camera, t_tcp_camera.reshape(3))
    T_camera_tcp = matrix_inverse(T_tcp_camera)
    consistency = compute_board_consistency(used, T_tcp_camera)

    output = {
        "frame": "tcp_camera",
        "unit": "mm",
        "convention": "T_tcp_camera maps camera-frame poses into TCP frame when used as T_base_tcp @ T_tcp_camera @ T_camera_pose.",
        "board_squares": [args.pattern_cols + 1, args.pattern_rows + 1],
        "pattern_inner_corners": [args.pattern_cols, args.pattern_rows],
        "square_mm": float(args.square_mm),
        "method": args.method,
        "T_tcp_camera": T_tcp_camera.tolist(),
        "T_camera_tcp": T_camera_tcp.tolist(),
        "num_total_samples": len(samples),
        "num_used_samples": len(used),
        "used_sample_indices": [item["index"] for item in used],
        "rejected_samples": rejected,
        "mean_reprojection_error_px": float(np.mean([item["reprojection_error_px"] for item in used])),
        "max_reprojection_error_px": float(np.max([item["reprojection_error_px"] for item in used])),
        "validation": consistency,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "num_used_samples": len(used), "validation": consistency}, indent=2))


if __name__ == "__main__":
    main()
