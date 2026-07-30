#!/usr/bin/env python
"""Interactively capture aligned RealSense RGB-D steps into numbered scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEFAULT_CAMERA_INDEX = 1
DEFAULT_CAMERA_SERIAL_SUFFIX = "76630"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "realworld_data"
WINDOW_NAME = "RealSense Scene Capture"


def list_realsense_devices() -> list[dict[str, Any]]:
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


def select_realsense_device(
    devices: list[dict[str, Any]],
    camera_serial: str | None,
    camera_index: int,
) -> dict[str, Any]:
    if not devices:
        raise RuntimeError("No RealSense camera was detected.")
    if camera_serial:
        exact_matches = [device for device in devices if device["serial"] == camera_serial]
        if exact_matches:
            return exact_matches[0]
        suffix_matches = [device for device in devices if device["serial"].endswith(camera_serial)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            matches = ", ".join(f"{device['index']}:{device['serial']}" for device in suffix_matches)
            raise RuntimeError(f"Camera serial suffix {camera_serial!r} matches multiple devices: {matches}")
        available = ", ".join(f"{device['index']}:{device['serial']}" for device in devices)
        raise RuntimeError(f"Camera serial {camera_serial!r} was not found. Available devices: {available}")
    if camera_index < 0 or camera_index >= len(devices):
        available = ", ".join(f"{device['index']}:{device['serial']}" for device in devices)
        raise RuntimeError(f"Camera index {camera_index} is out of range. Available devices: {available}")
    return devices[camera_index]


def next_scene_directory(parent: Path) -> Path:
    index = 1
    while (parent / f"scene_{index}").exists():
        index += 1
    return parent / f"scene_{index}"


def make_preview(color_bgr: np.ndarray, depth_raw: np.ndarray, scene_name: str, step_index: int) -> np.ndarray:
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_raw, alpha=0.03),
        cv2.COLORMAP_JET,
    )
    color_preview = cv2.resize(color_bgr, (640, 360), interpolation=cv2.INTER_AREA)
    depth_preview = cv2.resize(depth_colormap, (640, 360), interpolation=cv2.INTER_NEAREST)
    preview = np.hstack((color_preview, depth_preview))
    cv2.putText(
        preview,
        f"{scene_name}  next: step_{step_index}  [c] capture  [q] quit",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def save_step(
    step_dir: Path,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    color_frame: rs.video_frame,
    depth_scale_m: float,
    selected_device: dict[str, Any],
) -> None:
    step_dir.mkdir(parents=False, exist_ok=False)
    rgb_path = step_dir / "rgb.png"
    depth_path = step_dir / "depth.png"
    meta_path = step_dir / "camera_meta.json"
    if not cv2.imwrite(str(rgb_path), color_bgr):
        raise RuntimeError(f"Failed to save RGB image: {rgb_path}")
    if not cv2.imwrite(str(depth_path), depth_raw):
        raise RuntimeError(f"Failed to save depth image: {depth_path}")

    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    metadata = {
        "camera": selected_device,
        "width": int(color_bgr.shape[1]),
        "height": int(color_bgr.shape[0]),
        "depth_format": "uint16_png",
        "depth_scale_m": float(depth_scale_m),
        "rgb_path": str(rgb_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "intrinsics": {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.ppx),
            "cy": float(intrinsics.ppy),
            "model": str(intrinsics.model),
            "coeffs": [float(value) for value in intrinsics.coeffs],
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a numbered multi-step RealSense RGB-D scene.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory for scene_N folders.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="Fallback device index when no serial is selected.")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL_SUFFIX, help="Full RealSense serial number or unique suffix.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene_dir = next_scene_directory(output_root)
    scene_dir.mkdir(parents=False, exist_ok=False)

    devices = list_realsense_devices()
    selected_device = select_realsense_device(devices, args.camera_serial, args.camera_index)
    print(f"[capture] scene directory: {scene_dir}")
    print(f"[capture] camera index={selected_device['index']} serial={selected_device['serial']}")
    print("[capture] press c to save the current RGB-D frame; press q to quit")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_device["serial"])
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)
    step_index = 0
    pipeline_started = False
    try:
        profile = pipeline.start(config)
        pipeline_started = True
        align = rs.align(rs.stream.color)
        depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            aligned_frames = align.process(pipeline.wait_for_frames())
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_raw = np.asanyarray(depth_frame.get_data())
            color_bgr = np.asanyarray(color_frame.get_data())
            cv2.imshow(WINDOW_NAME, make_preview(color_bgr, depth_raw, scene_dir.name, step_index))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                step_dir = scene_dir / f"step_{step_index}"
                save_step(step_dir, color_bgr, depth_raw, color_frame, depth_scale_m, selected_device)
                print(f"[capture] saved {step_dir}")
                step_index += 1
    finally:
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[capture] finished: scene={scene_dir.name} saved_steps={step_index}")


if __name__ == "__main__":
    main()
