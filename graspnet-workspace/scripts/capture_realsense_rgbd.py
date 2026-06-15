#!/usr/bin/env python3
"""Capture one aligned RealSense RGB-D frame for reveal-push simulation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is not installed in this environment. "
        "Run this script with: conda run -n calib python "
        "scripts/capture_realsense_rgbd.py"
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture color, depth aligned to color, and aligned camera "
            "intrinsics from an Intel RealSense camera."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="real_rgbd_capture",
        help="Directory in which the captured files are written",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: realsense_YYYYmmdd_HHMMSS)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames discarded before the saved capture",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="Optional RealSense device serial number",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("Width, height and fps must be positive")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or datetime.now().strftime("realsense_%Y%m%d_%H%M%S")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.depth,
        args.width,
        args.height,
        rs.format.z16,
        args.fps,
    )
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )

    started = False
    try:
        profile = pipeline.start(config)
        started = True
        depth_sensor = profile.get_device().first_depth_sensor()
        meters_per_depth_unit = float(depth_sensor.get_depth_scale())
        if meters_per_depth_unit <= 0:
            raise RuntimeError("RealSense returned an invalid depth scale")

        align = rs.align(rs.stream.color)
        aligned_depth_frame = None
        color_frame = None
        for _ in range(args.warmup_frames + 1):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                raise RuntimeError("RealSense returned an incomplete RGB-D frame")

        depth = np.asanyarray(aligned_depth_frame.get_data()).copy()
        color = np.asanyarray(color_frame.get_data()).copy()
        if depth.ndim != 2 or color.ndim != 3:
            raise RuntimeError("RealSense returned unexpected image dimensions")
        if depth.shape != color.shape[:2]:
            raise RuntimeError(
                f"Aligned depth shape {depth.shape} does not match "
                f"color shape {color.shape[:2]}"
            )
        if depth.dtype != np.uint16:
            raise RuntimeError(
                f"Expected uint16 Z16 depth, received {depth.dtype}"
            )

        intrinsics = (
            aligned_depth_frame.profile
            .as_video_stream_profile()
            .get_intrinsics()
        )
        rgb_path = output_dir / f"{prefix}_rgb.png"
        depth_npy_path = output_dir / f"{prefix}_depth.npy"
        depth_png_path = output_dir / f"{prefix}_depth.png"
        intrinsics_path = output_dir / f"{prefix}_intrinsics.json"
        manifest_path = output_dir / f"{prefix}_capture.json"

        if not cv2.imwrite(str(rgb_path), color):
            raise RuntimeError(f"Failed to save RGB image: {rgb_path}")
        np.save(depth_npy_path, depth)
        if not cv2.imwrite(str(depth_png_path), depth):
            raise RuntimeError(f"Failed to save depth PNG: {depth_png_path}")

        units_per_meter = 1.0 / meters_per_depth_unit
        intrinsics_data = {
            "camera_intrinsics": {
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.ppx),
                "cy": float(intrinsics.ppy),
            },
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "distortion_model": str(intrinsics.model),
            "distortion_coefficients": [
                float(value) for value in intrinsics.coeffs
            ],
            "aligned_to": "color",
            "depth_scale_m_per_unit": meters_per_depth_unit,
            "depth_units_per_meter": units_per_meter,
        }
        with intrinsics_path.open("w", encoding="utf-8") as file:
            json.dump(intrinsics_data, file, indent=2)

        manifest = {
            "rgb": str(rgb_path),
            "depth_npy": str(depth_npy_path),
            "depth_png": str(depth_png_path),
            "intrinsics": str(intrinsics_path),
            "depth_scale_for_demo": units_per_meter,
            "image_shape": [int(depth.shape[0]), int(depth.shape[1])],
            "valid_depth_pixels": int(np.count_nonzero(depth)),
        }
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

        print(f"[RGB] {rgb_path}")
        print(f"[Depth NPY] {depth_npy_path}")
        print(f"[Depth PNG] {depth_png_path}")
        print(f"[Intrinsics] {intrinsics_path}")
        print(f"[Manifest] {manifest_path}")
        print(f"[Use --depth-scale] {units_per_meter:g}")
    finally:
        if started:
            pipeline.stop()


if __name__ == "__main__":
    main()
