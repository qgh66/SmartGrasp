#!/usr/bin/env python
"""Diagnose chessboard height in JAKA base coordinates from one aligned RGB-D frame."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
JAKA_WORKER = WORKSPACE_ROOT / "scripts" / "jaka_motion_worker.py"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT.parent / "result" / "chessboard_height_check"
DEFAULT_HAND_EYE_CALIBRATION = WORKSPACE_ROOT / "calibration" / "hand_eye_tcp_camera.json"
DEFAULT_CAMERA_INDEX = 1
DEFAULT_CAMERA_SERIAL_SUFFIX = "76630"
DEFAULT_JAKA_IP = "192.168.1.199"
DEFAULT_JAKA_PYTHON = "/home/admin128/anaconda3/envs/smartgrasp310/bin/python"
DEFAULT_JKRC_DIR = WORKSPACE_ROOT / "jkrc"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEFAULT_PATTERN_COLS = 11
DEFAULT_PATTERN_ROWS = 8


def list_realsense_devices(rs) -> list[dict[str, Any]]:
    devices = []
    for index, device in enumerate(rs.context().query_devices()):
        devices.append(
            {
                "index": index,
                "serial": device.get_info(rs.camera_info.serial_number),
                "name": device.get_info(rs.camera_info.name),
                "product_line": device.get_info(rs.camera_info.product_line),
            }
        )
    return devices


def select_realsense_device(devices: list[dict[str, Any]], camera_serial: str | None, camera_index: int) -> dict[str, Any]:
    if not devices:
        raise RuntimeError("No RealSense device found.")
    if camera_serial:
        suffix_matches = []
        for device in devices:
            if device["serial"] == camera_serial:
                return device
            if str(device["serial"]).endswith(camera_serial):
                suffix_matches.append(device)
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            available = ", ".join(f"{d['index']}:{d['serial']}" for d in suffix_matches)
            raise RuntimeError(f"Camera serial suffix {camera_serial!r} matched multiple devices: {available}")
        available = ", ".join(f"{d['index']}:{d['serial']}" for d in devices)
        raise RuntimeError(f"Camera serial {camera_serial!r} not found. Available: {available}")
    if camera_index < 0 or camera_index >= len(devices):
        available = ", ".join(f"{d['index']}:{d['serial']}" for d in devices)
        raise RuntimeError(f"Camera index {camera_index} is out of range. Available: {available}")
    return devices[camera_index]


def read_tcp_pose(args: argparse.Namespace) -> list[float]:
    command = [
        str(Path(args.jaka_python).expanduser()),
        str(JAKA_WORKER),
        "--print-tcp-pose",
        "--json-only",
        "--jaka-ip",
        args.jaka_ip,
        "--jkrc-dir",
        str(Path(args.jkrc_dir).expanduser()),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        tcp_pose = payload.get("tcp_pose")
        if isinstance(tcp_pose, list) and len(tcp_pose) == 6:
            return [float(value) for value in tcp_pose]
    raise RuntimeError(f"Could not parse TCP pose from JAKA worker output: {result.stdout!r}")


def jaka_pose_to_matrix(pose: list[float]) -> np.ndarray:
    if len(pose) != 6:
        raise ValueError(f"JAKA TCP pose must be 6D, got {pose!r}")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    transform[:3, 3] = np.asarray(pose[:3], dtype=float)
    return transform


def detect_chessboard(color_bgr: np.ndarray, pattern_size: tuple[int, int]) -> dict[str, Any]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return {"found": False, "corners": None}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return {"found": True, "corners": refined.reshape(-1, 2)}


def camera_xyz_from_pixel(u: float, v: float, depth_m: float, intrinsics: dict[str, float]) -> np.ndarray:
    z_mm = float(depth_m) * 1000.0
    x_mm = (float(u) - float(intrinsics["cx"])) * z_mm / float(intrinsics["fx"])
    y_mm = (float(v) - float(intrinsics["cy"])) * z_mm / float(intrinsics["fy"])
    return np.array([x_mm, y_mm, z_mm], dtype=float)


def depth_at_pixel(depth_raw: np.ndarray, u: float, v: float, depth_scale_m: float, radius: int) -> float | None:
    x = int(round(float(u)))
    y = int(round(float(v)))
    y0 = max(0, y - radius)
    y1 = min(depth_raw.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(depth_raw.shape[1], x + radius + 1)
    patch = depth_raw[y0:y1, x0:x1].astype(np.float64)
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    return float(np.median(valid) * float(depth_scale_m))


def transform_point(transform: np.ndarray, point_xyz_mm: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = point_xyz_mm
    return (transform @ homogeneous)[:3]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def draw_diagnostic(color_bgr: np.ndarray, corners: np.ndarray, pattern_size: tuple[int, int], output_path: Path) -> None:
    preview = color_bgr.copy()
    cv2.drawChessboardCorners(preview, pattern_size, corners.reshape(-1, 1, 2).astype(np.float32), True)
    cv2.imwrite(str(output_path), preview)


def analyze_frame(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    color_frame: Any,
    depth_scale_m: float,
    selected_device: dict[str, Any],
    devices: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pattern_size = (args.pattern_cols, args.pattern_rows)
    detection = detect_chessboard(color_bgr, pattern_size)
    if not detection["found"]:
        raise RuntimeError("Chessboard not detected. Move the board/camera or improve lighting.")

    color_intr = color_frame.profile.as_video_stream_profile().intrinsics
    intrinsics = {
        "fx": float(color_intr.fx),
        "fy": float(color_intr.fy),
        "cx": float(color_intr.ppx),
        "cy": float(color_intr.ppy),
        "model": str(color_intr.model),
        "coeffs": [float(value) for value in color_intr.coeffs],
    }
    tcp_pose = [float(value) for value in args.capture_tcp_pose] if args.capture_tcp_pose else read_tcp_pose(args)
    base_from_tcp = jaka_pose_to_matrix(tcp_pose)
    calibration = json.loads(Path(args.hand_eye_calibration).expanduser().read_text(encoding="utf-8"))
    tcp_from_camera = np.asarray(calibration["T_tcp_camera"], dtype=float).reshape(4, 4)
    translation_offset_mm = np.asarray(args.tcp_camera_translation_offset_mm, dtype=float).reshape(3)
    raw_tcp_from_camera_translation_mm = tcp_from_camera[:3, 3].copy()
    tcp_from_camera[:3, 3] += translation_offset_mm
    camera_from_tcp = np.linalg.inv(tcp_from_camera)
    transforms = {
        "T_base_tcp__T_tcp_camera": base_from_tcp @ tcp_from_camera,
        "T_base_tcp__inv_T_tcp_camera": base_from_tcp @ camera_from_tcp,
    }

    rows = []
    for index, (u, v) in enumerate(detection["corners"]):
        depth_m = depth_at_pixel(depth_raw, u, v, depth_scale_m, args.depth_patch_radius)
        if depth_m is None:
            rows.append({"corner_index": int(index), "pixel": [float(u), float(v)], "valid_depth": False})
            continue
        camera_xyz_mm = camera_xyz_from_pixel(u, v, depth_m, intrinsics)
        item = {
            "corner_index": int(index),
            "pixel": [float(u), float(v)],
            "valid_depth": True,
            "depth_m": float(depth_m),
            "camera_xyz_mm": camera_xyz_mm.tolist(),
        }
        for name, transform in transforms.items():
            item[f"{name}_base_xyz_mm"] = transform_point(transform, camera_xyz_mm).tolist()
        rows.append(item)

    z_by_transform: dict[str, list[float]] = {}
    for name in transforms:
        key = f"{name}_base_xyz_mm"
        z_by_transform[name] = [float(row[key][2]) for row in rows if row.get("valid_depth") and key in row]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    rgb_path = output_dir / f"chessboard_height_{timestamp}_rgb.png"
    depth_path = output_dir / f"chessboard_height_{timestamp}_depth.raw"
    overlay_path = output_dir / f"chessboard_height_{timestamp}_corners.png"
    result_path = output_dir / f"chessboard_height_{timestamp}.json"

    cv2.imwrite(str(rgb_path), color_bgr)
    depth_raw.astype(np.uint16, copy=False).tofile(depth_path)
    draw_diagnostic(color_bgr, detection["corners"], pattern_size, overlay_path)

    result = {
        "timestamp": time.time(),
        "expected_board_height_mm": float(args.expected_board_height_mm),
        "height_error_summary_mm": {
            name: summarize([value - float(args.expected_board_height_mm) for value in values])
            for name, values in z_by_transform.items()
        },
        "base_z_summary_mm": {name: summarize(values) for name, values in z_by_transform.items()},
        "camera_depth_summary_m": summarize([row["depth_m"] for row in rows if row.get("valid_depth")]),
        "num_corners": int(len(rows)),
        "num_valid_depth_corners": int(sum(1 for row in rows if row.get("valid_depth"))),
        "tcp_pose": tcp_pose,
        "hand_eye_calibration": str(Path(args.hand_eye_calibration).expanduser().resolve()),
        "tcp_camera_translation_offset_mm": translation_offset_mm.tolist(),
        "raw_T_tcp_camera_translation_mm": raw_tcp_from_camera_translation_mm.tolist(),
        "T_tcp_camera_translation_mm": tcp_from_camera[:3, 3].tolist(),
        "camera_origin_base_mm": {
            name: transform_point(transform, np.zeros(3, dtype=float)).tolist()
            for name, transform in transforms.items()
        },
        "camera_z_axis_in_base": {
            name: transform[:3, 2].tolist()
            for name, transform in transforms.items()
        },
        "resolution": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
        "depth_scale_m": float(depth_scale_m),
        "intrinsics": intrinsics,
        "selected_device": selected_device,
        "available_devices": devices,
        "rgb_path": str(rgb_path),
        "depth_raw_path": str(depth_path),
        "corners_overlay_path": str(overlay_path),
        "corners": rows,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def draw_preview(color_bgr: np.ndarray, detection: dict[str, Any], pattern_size: tuple[int, int]) -> np.ndarray:
    preview = color_bgr.copy()
    if detection["found"]:
        cv2.drawChessboardCorners(preview, pattern_size, detection["corners"].reshape(-1, 1, 2).astype(np.float32), True)
    status = "FOUND" if detection["found"] else "NOT FOUND"
    color = (0, 255, 0) if detection["found"] else (0, 0, 255)
    cv2.putText(preview, f"chessboard={status} | c: capture | q: quit", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return preview


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check chessboard height in JAKA base coordinates with aligned RealSense RGB-D.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for diagnostic JSON/images.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="Fallback RealSense device index.")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL_SUFFIX, help="RealSense serial number or unique suffix.")
    parser.add_argument("--jaka-python", default=DEFAULT_JAKA_PYTHON, help="Python executable that can import jkrc.")
    parser.add_argument("--jkrc-dir", default=str(DEFAULT_JKRC_DIR), help="Directory containing jkrc.so and libjakaAPI.so.")
    parser.add_argument("--jaka-ip", default=DEFAULT_JAKA_IP, help="JAKA controller IP.")
    parser.add_argument("--hand-eye-calibration", default=str(DEFAULT_HAND_EYE_CALIBRATION), help="JSON containing T_tcp_camera.")
    parser.add_argument("--tcp-camera-translation-offset-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"), help="Temporary XYZ offset in TCP frame added to T_tcp_camera translation before height checks.")
    parser.add_argument("--capture-tcp-pose", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "RX", "RY", "RZ"), help="Use this TCP pose instead of reading the robot.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to discard before capture.")
    parser.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_COLS, help="Chessboard inner corner count along columns.")
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_ROWS, help="Chessboard inner corner count along rows.")
    parser.add_argument("--expected-board-height-mm", type=float, default=3.0, help="Expected chessboard corner height above base z=0.")
    parser.add_argument("--depth-patch-radius", type=int, default=2, help="Median depth patch radius around each detected corner.")
    parser.add_argument("--capture-on-found", action="store_true", help="Capture the first frame where the chessboard is detected without opening a preview window.")
    parser.add_argument("--list-devices", action="store_true", help="List RealSense devices and exit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import pyrealsense2 as rs

    devices = list_realsense_devices(rs)
    if args.list_devices:
        print(json.dumps(devices, ensure_ascii=False, indent=2))
        return
    selected_device = select_realsense_device(devices, args.camera_serial, args.camera_index)
    print(f"[height-check] using RealSense index={selected_device['index']} serial={selected_device['serial']}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_device["serial"])
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
    pattern_size = (args.pattern_cols, args.pattern_rows)
    try:
        for _ in range(max(1, args.warmup_frames)):
            align.process(pipeline.wait_for_frames())
        while True:
            aligned_frames = align.process(pipeline.wait_for_frames())
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_raw = np.asanyarray(depth_frame.get_data())
            color_bgr = np.asanyarray(color_frame.get_data())
            if depth_raw.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or color_bgr.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise RuntimeError(f"Expected 1280x720 RGB-D, got depth={depth_raw.shape[::-1]} color={color_bgr.shape[1]}x{color_bgr.shape[0]}")
            detection = detect_chessboard(color_bgr, pattern_size)
            if args.capture_on_found and detection["found"]:
                result = analyze_frame(color_bgr, depth_raw, color_frame, depth_scale_m, selected_device, devices, args)
                print(json.dumps({key: result[key] for key in ["result_path", "base_z_summary_mm", "height_error_summary_mm", "camera_origin_base_mm"]}, indent=2))
                return
            preview = draw_preview(color_bgr, detection, pattern_size)
            cv2.imshow("Chessboard Height Check", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord("c"):
                result = analyze_frame(color_bgr, depth_raw, color_frame, depth_scale_m, selected_device, devices, args)
                print(json.dumps({key: result[key] for key in ["result_path", "base_z_summary_mm", "height_error_summary_mm", "camera_origin_base_mm"]}, indent=2))
                return
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
