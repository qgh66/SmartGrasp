#!/usr/bin/env python
"""Collect eye-in-hand chessboard samples with RealSense RGB-D and JAKA TCP pose."""

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


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
JAKA_WORKER = WORKSPACE_ROOT / "scripts" / "jaka_motion_worker.py"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "calibration" / "handeye_chessboard_raw"
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


def next_sample_index(samples_path: Path) -> int:
    existing_dirs = []
    if samples_path.parent.exists():
        for path in samples_path.parent.glob("sample_*"):
            if path.is_dir():
                try:
                    existing_dirs.append(int(path.name.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
    if not samples_path.exists():
        return max(existing_dirs, default=-1) + 1
    index = max(existing_dirs, default=-1) + 1
    with samples_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = max(index, int(item.get("index", -1)) + 1)
    return index


def detect_chessboard(color_bgr: np.ndarray, pattern_size: tuple[int, int]) -> dict[str, Any]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return {"found": False, "corners": None}

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return {"found": True, "corners": refined}


def draw_status(image: np.ndarray, sample_index: int, detection: dict[str, Any], pattern_size: tuple[int, int]) -> np.ndarray:
    preview = image.copy()
    if detection["found"]:
        cv2.drawChessboardCorners(preview, pattern_size, detection["corners"], True)
    status = "FOUND" if detection["found"] else "NOT FOUND"
    color = (0, 255, 0) if detection["found"] else (0, 0, 255)
    text = f"sample_{sample_index} chessboard={status} | c: capture | q: quit"
    cv2.putText(preview, text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(
        preview,
        "12x9 squares => 11x8 inner corners, 10mm square",
        (24, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return preview


def save_sample(
    output_dir: Path,
    samples_path: Path,
    index: int,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    color_frame: Any,
    profile: Any,
    devices: list[dict[str, Any]],
    selected_device: dict[str, Any],
    detection: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if not detection["found"]:
        print("[collect] chessboard not detected; sample not saved. Move camera or improve lighting and try again.")
        return

    sample_dir = output_dir / f"sample_{index}"
    sample_dir.mkdir(parents=True, exist_ok=False)
    rgb_path = sample_dir / "rgb.png"
    depth_path = sample_dir / "depth.png"
    corners_path = sample_dir / "corners.json"
    visualization_path = sample_dir / "corners_visualization.png"
    metadata_path = sample_dir / "metadata.json"
    cv2.imwrite(str(rgb_path), color_bgr)
    cv2.imwrite(str(depth_path), depth_raw)

    color_intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    tcp_pose = read_tcp_pose(args)
    corners = detection["corners"].reshape(-1, 2).astype(float)
    visualization = cv2.drawChessboardCorners(color_bgr.copy(), (args.pattern_cols, args.pattern_rows), detection["corners"], True)
    cv2.imwrite(str(visualization_path), visualization)
    corners_payload = {
        "pattern_inner_corners": [args.pattern_cols, args.pattern_rows],
        "corner_count": int(len(corners)),
        "corners_px": corners.tolist(),
    }
    corners_path.write_text(json.dumps(corners_payload, indent=2), encoding="utf-8")
    record = {
        "index": index,
        "sample_dir": sample_dir.name,
        "timestamp": time.time(),
        "rgb_path": str(rgb_path.relative_to(output_dir)),
        "depth_path": str(depth_path.relative_to(output_dir)),
        "corners_path": str(corners_path.relative_to(output_dir)),
        "corners_visualization_path": str(visualization_path.relative_to(output_dir)),
        "metadata_path": str(metadata_path.relative_to(output_dir)),
        "chessboard_detected": True,
        "pattern_inner_corners": [args.pattern_cols, args.pattern_rows],
        "square_mm": float(args.square_mm),
        "corner_count": int(len(corners)),
        "tcp_pose": tcp_pose,
        "selected_device": selected_device,
        "available_devices": devices,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "depth_scale_m": depth_scale,
        "intrinsics": {
            "fx": float(color_intr.fx),
            "fy": float(color_intr.fy),
            "cx": float(color_intr.ppx),
            "cy": float(color_intr.ppy),
            "model": str(color_intr.model),
            "coeffs": [float(value) for value in color_intr.coeffs],
        },
    }
    metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    with samples_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[collect] saved sample_{index}: {sample_dir}, corners={len(corners)}, tcp={tcp_pose}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect RealSense + JAKA TCP samples for eye-in-hand chessboard calibration.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for samples.jsonl and captured images.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="Fallback RealSense device index.")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL_SUFFIX, help="RealSense serial number or unique suffix.")
    parser.add_argument("--jaka-python", default=DEFAULT_JAKA_PYTHON, help="Python executable that can import jkrc.")
    parser.add_argument("--jkrc-dir", default=str(DEFAULT_JKRC_DIR), help="Directory containing jkrc.so and libjakaAPI.so.")
    parser.add_argument("--jaka-ip", default=DEFAULT_JAKA_IP, help="JAKA controller IP.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to discard before preview/capture.")
    parser.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_COLS, help="Chessboard inner corner count along columns.")
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_ROWS, help="Chessboard inner corner count along rows.")
    parser.add_argument("--square-mm", type=float, default=10.0, help="Chessboard square size in millimeters, stored with each sample.")
    parser.add_argument("--list-devices", action="store_true", help="List RealSense devices and exit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import pyrealsense2 as rs

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"

    devices = list_realsense_devices(rs)
    if args.list_devices:
        print(json.dumps(devices, ensure_ascii=False, indent=2))
        return
    selected_device = select_realsense_device(devices, args.camera_serial, args.camera_index)
    print(f"[collect] using RealSense index={selected_device['index']} serial={selected_device['serial']}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_device["serial"])
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    sample_index = next_sample_index(samples_path)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    try:
        for _ in range(max(1, args.warmup_frames)):
            align.process(pipeline.wait_for_frames())
        print("[collect] preview started. Move the robot manually, press c to capture, q to quit.")
        while True:
            aligned_frames = align.process(pipeline.wait_for_frames())
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_raw = np.asanyarray(depth_frame.get_data())
            color_bgr = np.asanyarray(color_frame.get_data())
            detection = detect_chessboard(color_bgr, pattern_size)
            preview = draw_status(color_bgr, sample_index, detection, pattern_size)
            cv2.imshow("Hand-Eye Chessboard Collection", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                save_sample(
                    output_dir,
                    samples_path,
                    sample_index,
                    color_bgr,
                    depth_raw,
                    color_frame,
                    profile,
                    devices,
                    selected_device,
                    detection,
                    args,
                )
                if detection["found"]:
                    sample_index += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
