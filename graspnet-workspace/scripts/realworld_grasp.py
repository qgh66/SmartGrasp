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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SMARTGRASP_ROOT = WORKSPACE_ROOT.parent
DEFAULT_CAMERA_COORDINATES_DIR = Path(
    "/home/admin128/ChengyuanWang/high_low_comm/scripts/human_playdata_process/"
    "hand_object_detector/camera_coordinates/camera_coordinates - 副本"
)
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
from models.graspnet import GraspNet, pred_decode  # noqa: E402
from simulation.candidate_visualizer import export_candidate_html, export_candidate_ply, export_candidate_png  # noqa: E402
from utils.collision_detector import ModelFreeCollisionDetector  # noqa: E402


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEFAULT_OUTPUT_DIR = SMARTGRASP_ROOT / "result"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "checkpoint-rs.tar"
DEFAULT_HAND_EYE_CALIBRATION = WORKSPACE_ROOT / "calibration" / "hand_eye_tcp_camera.json"
DEFAULT_TCP_CAMERA_TRANSLATION_OFFSET_MM = [0.0, 0.0, -82.5]
DEFAULT_GRASP_CENTER_TO_TCP_OFFSET_MM = 174.0
DEFAULT_JAKA_IP = "192.168.1.199"
DEFAULT_ROBOTIQ_PORT = "/dev/ttyUSB0"
DEFAULT_READY_POSE = [300.0, 0.0, 350.0, 3.141592653589793, 0.0, 0.0]
DEFAULT_CAMERA_INDEX = 1
DEFAULT_CAMERA_SERIAL_SUFFIX = "72508"
DEFAULT_JAKA_PYTHON = os.environ.get("JAKA_PYTHON", "/home/admin128/anaconda3/envs/smartgrasp310/bin/python")
VENDOR_DIR = WORKSPACE_ROOT / "vendor"
JKRC_DIR = WORKSPACE_ROOT / "jkrc"
JAKA_WORKER = WORKSPACE_ROOT / "scripts" / "jaka_motion_worker.py"
_SAM_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
DEFAULT_PLATE_TO_ROBOT_MM = np.array(
    [
        [1.0, 0.0, 0.0, -550.0],
        [0.0, -1.0, 0.0, 67.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def import_jkrc_backend():
    jkrc_path = str(JKRC_DIR)
    if JKRC_DIR.exists() and jkrc_path not in sys.path:
        sys.path.insert(0, jkrc_path)

    local_jaka_api = JKRC_DIR / "libjakaAPI.so"
    if local_jaka_api.exists():
        import ctypes

        ctypes.CDLL(str(local_jaka_api), mode=ctypes.RTLD_GLOBAL)

    try:
        import jkrc
    except Exception as exc:
        raise RuntimeError(
            "JAKA 执行模式需要 jkrc。已优先尝试加载本项目下的 "
            f"{JKRC_DIR / 'jkrc.so'}，但导入失败: {exc!r}。如果错误包含 "
            "Py_TPFLAGS_HAVE_GC，说明这个 jkrc.so 与当前 Python ABI 不兼容，"
            "需要换成当前 smartgrasp Python 版本匹配的 JAKA jkrc.so/wheel，"
            "或切到该 jkrc 编译时对应的 Python 环境。"
        ) from exc

    print(f"[jkrc] loaded from {getattr(jkrc, '__file__', 'unknown')}")
    return jkrc


def jaka_return_code(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
        first = raw[0]
        if isinstance(first, (int, np.integer)):
            return int(first)
    return None


def check_jaka_call(name: str, raw: Any, allow_none: bool = True) -> None:
    print(f"[jaka] {name} returned: {raw!r}")
    code = jaka_return_code(raw)
    if code is None:
        if allow_none:
            return
        raise RuntimeError(f"{name} returned unexpected format: {raw!r}")
    if code != 0:
        if name == "login" and code == -1:
            raise RuntimeError(
                "JAKA login failed with ret=-1. The loaded JAKA SDK V2.2.7 requires "
                "controller version 1.7.2_28 or newer. For controller 1.7.0_x or 1.5.x, "
                "use SDK v2.1.11 or earlier, or upgrade the robot controller firmware."
            )
        raise RuntimeError(f"{name} failed: ret={code}, raw={raw!r}")


def _realsense_devices(rs) -> list[dict[str, str]]:
    devices = []
    for index, device in enumerate(rs.context().query_devices()):
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        product_line = device.get_info(rs.camera_info.product_line)
        devices.append(
            {
                "index": str(index),
                "serial": serial,
                "name": name,
                "product_line": product_line,
            }
        )
    return devices


def _select_realsense_device(devices: list[dict[str, str]], serial: str | None, index: int) -> dict[str, str]:
    if not devices:
        raise RuntimeError("No RealSense device found.")
    if serial:
        suffix_matches = []
        for device in devices:
            if device["serial"] == serial:
                return device
            if device["serial"].endswith(serial):
                suffix_matches.append(device)
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            available = ", ".join(f'{d["index"]}:{d["serial"]}' for d in suffix_matches)
            raise RuntimeError(f"RealSense serial suffix {serial!r} matched multiple devices: {available}")
        available = ", ".join(f'{d["index"]}:{d["serial"]}' for d in devices)
        raise RuntimeError(f"Requested RealSense serial {serial!r} not found. Available: {available}")
    if index < 0 or index >= len(devices):
        available = ", ".join(f'{d["index"]}:{d["serial"]}' for d in devices)
        raise RuntimeError(f"Requested RealSense camera index {index} is out of range. Available: {available}")
    return devices[index]


def capture_realsense(output_dir: Path, warmup_frames: int, camera_serial: str | None, camera_index: int) -> dict[str, Any]:
    """Capture one aligned 1280x720 RealSense RGB-D frame and save it."""
    import pyrealsense2 as rs

    output_dir.mkdir(parents=True, exist_ok=True)
    devices = _realsense_devices(rs)
    selected_device = _select_realsense_device(devices, camera_serial, camera_index)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_device["serial"])
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    try:
        aligned_frames = None
        for _ in range(max(1, warmup_frames)):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
        assert aligned_frames is not None

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense did not return both aligned color and depth frames.")

        depth_raw = np.asanyarray(depth_frame.get_data())
        color_bgr = np.asanyarray(color_frame.get_data())
        if depth_raw.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise RuntimeError(f"Depth resolution must be 1280x720, got {depth_raw.shape[::-1]}.")
        if color_bgr.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise RuntimeError(f"RGB resolution must be 1280x720, got {color_bgr.shape[1]}x{color_bgr.shape[0]}.")

        rgb_path = output_dir / "rgb.png"
        depth_raw_path = output_dir / "depth.raw"
        cv2.imwrite(str(rgb_path), color_bgr)
        depth_raw.astype(np.uint16, copy=False).tofile(depth_raw_path)

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        meta = {
            "timestamp": time.time(),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "depth_format": "uint16_z16_little_endian",
            "depth_scale_m": depth_scale,
            "rgb_path": str(rgb_path.resolve()),
            "depth_raw_path": str(depth_raw_path.resolve()),
            "selected_device": selected_device,
            "available_devices": devices,
            "intrinsics": {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.ppx),
                "cy": float(intr.ppy),
                "model": str(intr.model),
                "coeffs": [float(value) for value in intr.coeffs],
            },
        }
        (output_dir / "camera_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"color_bgr": color_bgr, "depth_raw": depth_raw, "meta": meta}
    finally:
        pipeline.stop()


def load_captured_frame(output_dir: Path) -> dict[str, Any]:
    """Load a previously captured result/rgb.png + result/depth.raw frame."""
    meta_path = output_dir / "camera_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing camera metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    color_bgr = cv2.imread(str(output_dir / "rgb.png"), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Missing RGB image: {output_dir / 'rgb.png'}")
    depth_raw = np.fromfile(output_dir / "depth.raw", dtype=np.uint16).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    return {"color_bgr": color_bgr, "depth_raw": depth_raw, "meta": meta}


def organized_point_cloud_from_depth(
    depth_raw: np.ndarray,
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project aligned depth to organized camera-frame xyz in meters."""
    intr = meta["intrinsics"]
    depth_m = depth_raw.astype(np.float32) * float(meta["depth_scale_m"])
    ys, xs = np.indices(depth_m.shape)
    valid = np.isfinite(depth_m) & (depth_m > args.min_depth) & (depth_m < args.max_depth)

    z = depth_m
    x = (xs.astype(np.float32) - float(intr["cx"])) * z / float(intr["fx"])
    y = (ys.astype(np.float32) - float(intr["cy"])) * z / float(intr["fy"])
    xyz_image = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)

    if args.bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = args.bounds
        valid &= (
            (xyz_image[..., 0] >= xmin)
            & (xyz_image[..., 0] <= xmax)
            & (xyz_image[..., 1] >= ymin)
            & (xyz_image[..., 1] <= ymax)
            & (xyz_image[..., 2] >= zmin)
            & (xyz_image[..., 2] <= zmax)
        )

    for box in args.exclude_camera_box or []:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(value) for value in box]
        inside = (
            (xyz_image[..., 0] >= xmin)
            & (xyz_image[..., 0] <= xmax)
            & (xyz_image[..., 1] >= ymin)
            & (xyz_image[..., 1] <= ymax)
            & (xyz_image[..., 2] >= zmin)
            & (xyz_image[..., 2] <= zmax)
        )
        valid &= ~inside

    return xyz_image, valid


def save_object_mask_images(
    mask: np.ndarray,
    color_bgr: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_path = output_dir / "mask.png"
    overlay_path = output_dir / "mask_overlay.png"
    cv2.imwrite(str(mask_path), mask_u8)

    overlay = color_bgr.copy()
    overlay[mask] = (0, 255, 0)
    debug = cv2.addWeighted(overlay, 0.45, color_bgr, 0.55, 0)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, contours, -1, (0, 0, 255), thickness=2)
    cv2.imwrite(str(overlay_path), debug)
    return mask_path, overlay_path


def draw_sam_prompt_preview(
    color_bgr: np.ndarray,
    foreground_points: list[tuple[int, int]],
    background_points: list[tuple[int, int]],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    preview = color_bgr.copy()
    if mask is not None:
        overlay = preview.copy()
        overlay[mask] = (0, 255, 0)
        preview = cv2.addWeighted(overlay, 0.45, preview, 0.55, 0)
        contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 0, 255), thickness=2)

    for index, (x, y) in enumerate(foreground_points):
        cv2.circle(preview, (x, y), 7, (0, 255, 0), thickness=-1)
        cv2.putText(preview, f"F{index}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    for index, (x, y) in enumerate(background_points):
        cv2.circle(preview, (x, y), 7, (0, 0, 255), thickness=-1)
        cv2.putText(preview, f"B{index}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    help_lines = [
        "LButton: foreground | RButton: background | u: undo | r: reset",
        "g/Space: generate SAM mask | Enter: accept | q/Esc: quit",
    ]
    for row, text in enumerate(help_lines):
        y = 28 + row * 28
        cv2.putText(preview, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4)
        cv2.putText(preview, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return preview


def load_sam_model(sam_model_id: str, device_name: str | None):
    from transformers import SamModel, SamProcessor

    device = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (sam_model_id, device)
    if cache_key not in _SAM_CACHE:
        processor = SamProcessor.from_pretrained(sam_model_id)
        model = SamModel.from_pretrained(sam_model_id).to(device)
        model.eval()
        _SAM_CACHE[cache_key] = (processor, model)
    return _SAM_CACHE[cache_key], device


def sam_mask_from_points(
    color_bgr: np.ndarray,
    foreground_points: list[tuple[int, int]],
    background_points: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    if not foreground_points:
        raise RuntimeError("At least one foreground point is required before generating a SAM mask.")

    (processor, model), device = load_sam_model(args.sam_model_id, args.sam_device)
    image_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    input_points = [[int(x), int(y)] for x, y in foreground_points + background_points]
    input_labels = [1 for _ in foreground_points] + [0 for _ in background_points]
    inputs = processor(
        image_rgb,
        input_points=[[input_points]],
        input_labels=[[input_labels]],
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)
    processed_masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu(),
    )[0]

    masks_np = np.asarray(processed_masks)
    if masks_np.ndim == 4 and masks_np.shape[0] == 1:
        masks_np = masks_np[0]
    if masks_np.ndim != 3:
        raise RuntimeError(f"Unexpected SAM mask tensor shape: {masks_np.shape}")
    scores = outputs.iou_scores[0].detach().cpu().numpy().reshape(-1)
    best_index = int(np.argmax(scores))
    mask = masks_np[best_index] > 0
    if args.mask_clean_kernel > 1:
        kernel = np.ones((args.mask_clean_kernel, args.mask_clean_kernel), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        mask = cleaned > 0
    return mask, float(scores[best_index])


def collect_interactive_sam_mask(frame: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    color_bgr = frame["color_bgr"]
    window_name = "SmartGrasp SAM mask prompt"
    foreground_points: list[tuple[int, int]] = []
    background_points: list[tuple[int, int]] = []
    history: list[tuple[str, tuple[int, int]]] = []
    current_mask: np.ndarray | None = None
    current_iou: float | None = None

    def on_mouse(event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (int(x), int(y))
            foreground_points.append(point)
            history.append(("foreground", point))
        elif event == cv2.EVENT_RBUTTONDOWN:
            point = (int(x), int(y))
            background_points.append(point)
            history.append(("background", point))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, color_bgr.shape[1]), min(720, color_bgr.shape[0]))
    cv2.setMouseCallback(window_name, on_mouse)
    print(
        "[sam-mask] Interactive window opened. "
        "Left-click foreground, right-click background, press g/Space to generate, Enter to accept.",
        flush=True,
    )

    try:
        while True:
            preview = draw_sam_prompt_preview(color_bgr, foreground_points, background_points, current_mask)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("g"), ord(" ")):
                current_mask, current_iou = sam_mask_from_points(color_bgr, foreground_points, background_points, args)
                save_object_mask_images(current_mask, color_bgr, output_dir)
                print(
                    f"[sam-mask] generated preview mask: pixels={int(np.count_nonzero(current_mask))} "
                    f"predicted_iou={current_iou:.4f}",
                    flush=True,
                )
            elif key in (13, 10):
                if current_mask is None:
                    current_mask, current_iou = sam_mask_from_points(color_bgr, foreground_points, background_points, args)
                break
            elif key == ord("u"):
                if history:
                    point_type, point = history.pop()
                    if point_type == "foreground" and foreground_points:
                        foreground_points.remove(point)
                    elif point_type == "background" and background_points:
                        background_points.remove(point)
                    current_mask = None
                    current_iou = None
            elif key == ord("r"):
                foreground_points.clear()
                background_points.clear()
                history.clear()
                current_mask = None
                current_iou = None
            elif key in (ord("q"), 27):
                raise RuntimeError("Interactive SAM mask selection was cancelled.")
    finally:
        cv2.destroyWindow(window_name)

    assert current_mask is not None
    mask_path, overlay_path = save_object_mask_images(current_mask, color_bgr, output_dir)
    prompt_payload = {
        "mode": "interactive_sam",
        "foreground_points": [{"x": int(x), "y": int(y)} for x, y in foreground_points],
        "background_points": [{"x": int(x), "y": int(y)} for x, y in background_points],
        "sam_model_id": args.sam_model_id,
        "sam_device": args.sam_device,
        "mask_clean_kernel": int(args.mask_clean_kernel),
        "predicted_iou": current_iou,
        "mask_pixels": int(np.count_nonzero(current_mask)),
        "mask_path": str(mask_path.resolve()),
        "mask_overlay_path": str(overlay_path.resolve()),
    }
    (output_dir / "sam_prompt_points.json").write_text(
        json.dumps(_json_safe(prompt_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return current_mask, prompt_payload


def mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("Cannot build a crop from an empty object mask.")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def expand_bbox_xyxy(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin_px: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    height, width = image_shape
    pad_x = int(round((x2 - x1) * float(margin_ratio))) + int(margin_px)
    pad_y = int(round((y2 - y1) * float(margin_ratio))) + int(margin_px)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def bbox_mask_xyxy(image_shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=bool)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = True
    return mask


def save_grasp_crop_overlay(
    color_bgr: np.ndarray,
    object_mask: np.ndarray,
    object_bbox: tuple[int, int, int, int],
    grasp_bbox: tuple[int, int, int, int],
    output_dir: Path,
) -> Path:
    overlay = color_bgr.copy()
    overlay[object_mask] = (0, 255, 0)
    debug = cv2.addWeighted(overlay, 0.35, color_bgr, 0.65, 0)
    ox1, oy1, ox2, oy2 = object_bbox
    gx1, gy1, gx2, gy2 = grasp_bbox
    cv2.rectangle(debug, (gx1, gy1), (gx2 - 1, gy2 - 1), (0, 0, 255), 2)
    cv2.rectangle(debug, (ox1, oy1), (ox2 - 1, oy2 - 1), (255, 0, 0), 2)
    path = output_dir / "grasp_crop_overlay.png"
    cv2.imwrite(str(path), debug)
    return path


def camera_point_to_pixel(point_camera_m: np.ndarray, intrinsics: dict[str, Any]) -> tuple[int, int] | None:
    x, y, z = np.asarray(point_camera_m, dtype=float).reshape(3)
    if not np.isfinite(z) or z <= 0:
        return None
    u = int(round(x * float(intrinsics["fx"]) / z + float(intrinsics["cx"])))
    v = int(round(y * float(intrinsics["fy"]) / z + float(intrinsics["cy"])))
    return u, v


def build_grasp_point_cloud(
    frame: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    xyz_image, valid = organized_point_cloud_from_depth(frame["depth_raw"], frame["meta"], args)
    full_cloud = xyz_image[valid]
    color_rgb_image = cv2.cvtColor(frame["color_bgr"], cv2.COLOR_BGR2RGB)
    full_cloud_rgb = color_rgb_image[valid]
    if len(full_cloud) == 0:
        raise RuntimeError("No valid point cloud points after depth/bounds filtering.")

    if not args.use_sam_mask:
        return full_cloud, full_cloud, full_cloud_rgb, full_cloud_rgb, full_cloud, {
            "point_cloud_source": "full_depth",
            "grasp_point_cloud_source": "full_depth",
            "obstacle_point_cloud_source": "full_depth_fallback_no_object_mask",
            "num_obstacle_points": int(len(full_cloud)),
            "object_mask": None,
            "camera_intrinsics": frame["meta"]["intrinsics"],
        }

    sam_mask, object_mask_info = collect_interactive_sam_mask(frame, output_dir, args)
    target_mask = sam_mask.astype(bool)
    object_mask = target_mask & valid
    obstacle_mask = valid & ~object_mask
    object_cloud = xyz_image[object_mask]
    object_cloud_rgb = color_rgb_image[object_mask]
    obstacle_cloud = xyz_image[obstacle_mask]
    if len(object_cloud) < args.object_min_points:
        raise RuntimeError(
            f"Object mask produced only {len(object_cloud)} valid depth points; "
            f"minimum is {args.object_min_points}."
        )
    object_bbox = mask_bbox_xyxy(target_mask)
    grasp_bbox = expand_bbox_xyxy(
        object_bbox,
        object_mask.shape,
        args.grasp_crop_margin_px,
        args.grasp_crop_margin_ratio,
    )
    grasp_crop_mask = bbox_mask_xyxy(object_mask.shape, grasp_bbox)
    grasp_input_mask = valid & grasp_crop_mask
    grasp_cloud = xyz_image[grasp_input_mask]
    grasp_cloud_rgb = color_rgb_image[grasp_input_mask]
    if len(grasp_cloud) == 0:
        raise RuntimeError("Expanded object crop produced no valid depth points for GraspNet.")

    crop_overlay_path = save_grasp_crop_overlay(frame["color_bgr"], target_mask, object_bbox, grasp_bbox, output_dir)
    object_mask_info["num_depth_valid_mask_pixels"] = int(np.count_nonzero(object_mask))
    object_mask_info["num_object_points"] = int(len(object_cloud))
    object_mask_info["object_bbox_xyxy"] = [int(value) for value in object_bbox]
    object_mask_info["grasp_crop_bbox_xyxy"] = [int(value) for value in grasp_bbox]
    object_mask_info["grasp_crop_margin_px"] = int(args.grasp_crop_margin_px)
    object_mask_info["grasp_crop_margin_ratio"] = float(args.grasp_crop_margin_ratio)
    object_mask_info["num_grasp_crop_points"] = int(len(grasp_cloud))
    object_mask_info["num_obstacle_points"] = int(len(obstacle_cloud))
    np.save(output_dir / "point_cloud_object_camera.npy", object_cloud.astype(np.float32, copy=False))
    np.save(output_dir / "point_cloud_grasp_input_camera.npy", grasp_cloud.astype(np.float32, copy=False))
    np.save(output_dir / "point_cloud_obstacles_camera.npy", obstacle_cloud.astype(np.float32, copy=False))
    print(
        "[object-mask] saved mask.png and FreeGrasp-style crop input; "
        f"mask_pixels={object_mask_info['mask_pixels']} "
        f"object_points={len(object_cloud)} "
        f"grasp_crop_points={len(grasp_cloud)} "
        f"obstacle_points={len(obstacle_cloud)} "
        f"grasp_crop_bbox={object_mask_info['grasp_crop_bbox_xyxy']} "
        f"mode={object_mask_info['mode']}",
        flush=True,
    )
    return full_cloud, grasp_cloud, full_cloud_rgb, grasp_cloud_rgb, obstacle_cloud, {
        "point_cloud_source": "interactive_sam_expanded_bbox_crop",
        "grasp_point_cloud_source": "valid_depth_inside_expanded_object_bbox",
        "obstacle_point_cloud_source": "full_depth_minus_interactive_sam_object_mask",
        "mask_path": object_mask_info["mask_path"],
        "mask_overlay_path": object_mask_info["mask_overlay_path"],
        "grasp_crop_overlay_path": str(crop_overlay_path.resolve()),
        "sam_prompt_points_path": str((output_dir / "sam_prompt_points.json").resolve()),
        "object_point_cloud_path": str((output_dir / "point_cloud_object_camera.npy").resolve()),
        "grasp_input_point_cloud_path": str((output_dir / "point_cloud_grasp_input_camera.npy").resolve()),
        "obstacle_point_cloud_path": str((output_dir / "point_cloud_obstacles_camera.npy").resolve()),
        "num_grasp_input_points": int(len(grasp_cloud)),
        "num_object_points": int(len(object_cloud)),
        "num_obstacle_points": int(len(obstacle_cloud)),
        "object_mask": object_mask_info,
        "object_mask_array": target_mask,
        "camera_intrinsics": frame["meta"]["intrinsics"],
    }


def sample_points(cloud: np.ndarray, num_points: int) -> np.ndarray:
    replace = len(cloud) < num_points
    indices = np.random.choice(len(cloud), num_points, replace=replace)
    return cloud[indices].astype(np.float32, copy=False)


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


def load_legacy_plate_calibration(camera_coordinates_dir: Path) -> dict[str, Any]:
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
        "plate_to_robot": DEFAULT_PLATE_TO_ROBOT_MM.copy(),
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


def offset_grasp_center_to_tcp(robot_from_grasp: np.ndarray, offset_mm: float) -> np.ndarray:
    """Move from GraspNet grasp center back to the physical TCP along approach."""
    robot_from_tcp = robot_from_grasp.copy()
    approach_axis = robot_from_grasp[:3, 0]
    robot_from_tcp[:3, 3] = robot_from_grasp[:3, 3] - approach_axis * float(offset_mm)
    return robot_from_tcp


def compute_robot_targets(
    records: list[dict[str, Any]],
    calibration: dict[str, Any] | None,
    grasp_center_to_tcp_offset_mm: float = 0.0,
) -> list[dict[str, Any]]:
    if calibration is None:
        return records
    for record in records:
        robot_from_grasp = camera_grasp_to_robot_transform(record, calibration)
        robot_from_tcp = offset_grasp_center_to_tcp(robot_from_grasp, grasp_center_to_tcp_offset_mm)
        record["grasp_center_jaka_pose"] = transform_to_jaka_pose(robot_from_grasp)
        record["target_jaka_tcp_pose"] = transform_to_jaka_pose(robot_from_tcp)
        record["target_robot_from_grasp"] = robot_from_grasp
        record["target_robot_from_grasp_center"] = robot_from_grasp
        record["target_robot_from_tcp"] = robot_from_tcp
        record["grasp_center_to_tcp_offset_mm"] = float(grasp_center_to_tcp_offset_mm)
        record["grasp_center_to_tcp_offset_axis"] = "-grasp_local_x"
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
    filtered_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    for record in records:
        pixel = camera_point_to_pixel(record["translation_camera_m"], intrinsics)
        kept = False
        reason = "invalid_projection"
        if pixel is not None:
            u, v = pixel
            if 0 <= u < width and 0 <= v < height:
                kept = bool(mask[v, u])
                reason = "inside_target_mask" if kept else "outside_target_mask"
            else:
                reason = "projection_outside_image"
        record["target_mask_center_filter"] = {
            "pixel_uv": None if pixel is None else [int(pixel[0]), int(pixel[1])],
            "kept": bool(kept),
            "reason": reason,
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
            "translation_camera_m": record["translation_camera_m"],
        }
        for record in removed_records
    ]
    if removed_records:
        print(
            "[candidate-filter] removed target-mask misses: "
            + ", ".join(
                f"raw_grasp_{item['raw_grasp_index']} "
                f"pixel={item['pixel_uv']} reason={item['reason']}"
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
        moving_vector_mm = target_transform[:3, 0] * float(approach_offset_mm)
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


def move_jaka_pose(target_pose: list[float], ip: str, velocity: float, acceleration: float) -> None:
    jkrc = import_jkrc_backend()

    robot = jkrc.RC(ip)
    check_jaka_call("login", robot.login())
    try:
        check_jaka_call("power_on", robot.power_on())
        check_jaka_call("enable_robot", robot.enable_robot())
        ret = robot.linear_move_extend(target_pose, 0, True, velocity, acceleration, 1)
        check_jaka_call("linear_move_extend", ret)
    finally:
        robot.logout()


def run_jaka_sequence(sequence: list[dict[str, Any]], args: argparse.Namespace, label: str) -> None:
    if args.jaka_executor == "direct":
        for step in sequence:
            if step["type"] == "move":
                move_jaka_pose(step["pose"], args.jaka_ip, args.velocity, args.acceleration)
            elif step["type"] == "gripper":
                command_gripper(step["command"], args.robotiq_port)
            else:
                raise ValueError(f"Unsupported JAKA step: {step}")
        return

    jaka_python = Path(args.jaka_python).expanduser()
    if not jaka_python.exists():
        raise FileNotFoundError(
            f"JAKA subprocess Python does not exist: {jaka_python}. "
            "Create a Python 3.10 env for jkrc or pass --jaka-python /path/to/python."
        )
    if not JAKA_WORKER.exists():
        raise FileNotFoundError(f"JAKA worker script does not exist: {JAKA_WORKER}")

    command = [
        str(jaka_python),
        str(JAKA_WORKER),
        "--sequence-json",
        json.dumps(_json_safe(sequence)),
        "--jaka-ip",
        args.jaka_ip,
        "--robotiq-port",
        args.robotiq_port,
        "--velocity",
        str(args.velocity),
        "--acceleration",
        str(args.acceleration),
        "--jkrc-dir",
        args.jkrc_dir,
    ]
    print(f"[jaka] running {label} via subprocess: {' '.join(command[:2])}")
    subprocess.run(command, check=True)


def read_jaka_tcp_pose(args: argparse.Namespace) -> list[float]:
    if args.jaka_executor == "direct":
        jkrc = import_jkrc_backend()
        robot = jkrc.RC(args.jaka_ip)
        check_jaka_call("login", robot.login())
        try:
            raw = robot.get_tcp_position()
        finally:
            robot.logout()
        return parse_jaka_tcp_pose(raw)

    jaka_python = Path(args.jaka_python).expanduser()
    command = [
        str(jaka_python),
        str(JAKA_WORKER),
        "--print-tcp-pose",
        "--json-only",
        "--jaka-ip",
        args.jaka_ip,
        "--jkrc-dir",
        args.jkrc_dir,
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


def parse_jaka_tcp_pose(raw: Any) -> list[float]:
    ret = None
    pos = None
    if isinstance(raw, (list, tuple, np.ndarray)):
        raw_list = list(raw)
        if len(raw_list) == 2:
            ret, pos = raw_list[0], raw_list[1]
        elif len(raw_list) == 7:
            ret, pos = raw_list[0], raw_list[1:]
        elif len(raw_list) == 6:
            ret, pos = 0, raw_list
        elif len(raw_list) >= 2:
            ret, pos = raw_list[0], raw_list[1]
        elif len(raw_list) == 1:
            ret, pos = raw_list[0], None
    elif isinstance(raw, (int, np.integer)):
        ret, pos = int(raw), None
    if ret is None or pos is None:
        raise RuntimeError(f"get_tcp_position returned unexpected format: {raw!r}")
    if int(ret) != 0:
        raise RuntimeError(f"get_tcp_position failed: ret={ret}, raw={raw!r}")
    if not isinstance(pos, (list, tuple, np.ndarray)) or len(pos) != 6:
        raise RuntimeError(f"get_tcp_position pose must be 6D: pos={pos!r}, raw={raw!r}")
    return [float(value) for value in pos]


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


def import_robotiq_backend():
    try:
        from robotiq_gripper_python import RobotiqGripper

        return "robotiq_gripper_python", RobotiqGripper
    except Exception as first_error:
        vendor_path = str(VENDOR_DIR)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        try:
            from pyrobotiqgripper import RobotiqGripper

            return "pyrobotiqgripper", RobotiqGripper
        except Exception as second_error:
            raise RuntimeError(
                "Failed to import a Robotiq gripper backend. Tried robotiq_gripper_python "
                f"and pyrobotiqgripper. First error: {first_error}; second error: {second_error}"
            ) from second_error


def command_gripper(opening: str, comport: str) -> None:
    backend, robotiq_cls = import_robotiq_backend()
    if backend == "robotiq_gripper_python":
        gripper = robotiq_cls(comport=comport)
        gripper.start()
    else:
        gripper = robotiq_cls(portname=comport, slaveAddress=9)
        if hasattr(gripper, "activate"):
            gripper.activate()
    try:
        if opening == "open":
            if backend == "robotiq_gripper_python":
                gripper.move(pos=0, vel=30, force=30, block=True)
            else:
                gripper.goTo(position=0, speed=30, force=30)
        elif opening == "close":
            if backend == "robotiq_gripper_python":
                gripper.move(pos=255, vel=30, force=30, block=True)
            else:
                gripper.goTo(position=255, speed=30, force=30)
        else:
            raise ValueError(f"Unsupported gripper command: {opening}")
    finally:
        if backend == "robotiq_gripper_python" and hasattr(gripper, "shutdown"):
            gripper.shutdown()
        elif hasattr(gripper, "disconnect"):
            gripper.disconnect()


def prepare_robot(args: argparse.Namespace) -> None:
    if args.skip_ready:
        return
    run_jaka_sequence(
        [
            {"type": "gripper", "command": "open"},
            {"type": "move", "pose": args.ready_pose},
        ],
        args,
        label="prepare_robot",
    )


def offset_pose_along_approach(target_transform: np.ndarray, offset_mm: float) -> list[float]:
    approach_axis = target_transform[:3, 0]
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
    pre_grasp_pose = offset_pose_along_approach(target_transform, args.approach_offset_mm)
    grasp_pose = transform_to_jaka_pose(target_transform)
    lift_pose = lift_pose_from_target(target_transform, args.lift_mm)

    run_jaka_sequence(
        [
            {"type": "move", "pose": pre_grasp_pose},
            {"type": "move", "pose": grasp_pose},
            {"type": "gripper", "command": "close"},
            {"type": "move", "pose": lift_pose},
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
) -> list[dict[str, Any]]:
    top_count = min(top_k, len(grasps))
    top_grasps = grasps[:top_count]
    records = [grasp_to_record(top_grasps[index], index) for index in range(top_count)]
    target_mask = None if point_cloud_info is None else point_cloud_info.get("object_mask_array")
    intrinsics = None if point_cloud_info is None else point_cloud_info.get("camera_intrinsics")
    records, target_mask_filter_info = filter_grasp_centers_in_target_mask(records, target_mask, intrinsics, args)
    top_grasps_for_collision = GraspGroup()
    for record in records:
        raw_index = int(record["grasp_index"])
        if 0 <= raw_index < len(top_grasps):
            top_grasps_for_collision.add(top_grasps[raw_index])
    collision_obstacle_cloud = grasp_cloud if obstacle_cloud is None else obstacle_cloud
    records, collision_filter_info = filter_grasp_collisions(
        records,
        top_grasps_for_collision,
        collision_obstacle_cloud,
        args,
    )
    records = compute_robot_targets(records, calibration, args.grasp_center_to_tcp_offset_mm)
    records, target_tcp_z_filter_info = filter_target_tcp_z(records, args)
    records, outlier_filter_info = filter_grasp_center_outliers(records, args)
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
        "num_candidates_after_filter": len(records),
        "grasp_candidates_ply": str((output_dir / "grasp_candidates.ply").resolve()),
        "target_mask_center_filter": target_mask_filter_info,
        "model_free_collision_filter": collision_filter_info,
        "target_tcp_z_filter": target_tcp_z_filter_info,
        "center_outlier_filter": outlier_filter_info,
        "point_cloud_source": None if point_cloud_info is None else point_cloud_info.get("point_cloud_source"),
        "grasp_point_cloud_source": None if point_cloud_info is None else point_cloud_info.get("grasp_point_cloud_source"),
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
        "grasp_center_to_tcp_offset_axis": "-grasp_local_x",
        "requires_ready_pose_matching_calibration": not (calibration is not None and calibration["mode"] == "hand_eye"),
        "candidates": records,
    }
    (output_dir / "grasp_candidates.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture real RGB-D, run GraspNet, and optionally move JAKA.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for rgb.png, depth.raw, and results.")
    parser.add_argument("--reuse-capture", action="store_true", help="Use existing output-dir/rgb.png + depth.raw.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="RealSense warmup frames before capture.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="Fallback RealSense device index if --camera-serial is empty.")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL_SUFFIX, help="RealSense serial number or unique suffix. Default matches the camera ending with 72508.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT), help="GraspNet checkpoint path.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num-points", type=int, default=20000, help="Point count sampled for GraspNet.")
    parser.add_argument("--top-k", type=int, default=50, help="Number of candidates to save and visualize.")
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
        default=True,
        help="Project each candidate center to the RGB image and keep it only if it lands inside the SAM target mask.",
    )
    parser.add_argument(
        "--filter-grasp-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove grasp candidates whose centers are outside the largest spatial cluster.",
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
        default=0.0,
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
    parser.add_argument("--skip-ready", action="store_true", help="Do not move to ready pose/open gripper before capture.")
    parser.add_argument(
        "--ready-pose",
        type=float,
        nargs=6,
        default=DEFAULT_READY_POSE,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Initial JAKA TCP pose before capture: mm + xyz Euler radians.",
    )
    parser.add_argument("--jaka-ip", default=DEFAULT_JAKA_IP, help="JAKA controller IP.")
    parser.add_argument("--robotiq-port", default=DEFAULT_ROBOTIQ_PORT, help="Robotiq serial port.")
    parser.add_argument(
        "--jaka-executor",
        choices=["subprocess", "direct"],
        default="subprocess",
        help="Run JAKA in a separate Python process by default, so GraspNet can stay in smartgrasp.",
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
    parser.add_argument("--candidate-index", type=int, default=0, help="Candidate index to execute.")
    parser.add_argument("--velocity", type=float, default=60.0, help="JAKA linear_move_extend velocity.")
    parser.add_argument("--acceleration", type=float, default=60.0, help="JAKA linear_move_extend acceleration.")
    parser.add_argument(
        "--grasp-center-to-tcp-offset-mm",
        type=float,
        default=DEFAULT_GRASP_CENTER_TO_TCP_OFFSET_MM,
        help=(
            "Distance from GraspNet grasp center back to the physical Robotiq TCP, "
            "applied along -GraspNet local X before robot execution."
        ),
    )
    parser.add_argument("--approach-offset-mm", type=float, default=80.0, help="Pre-grasp retreat along GraspNet local X.")
    parser.add_argument("--lift-mm", type=float, default=120.0, help="Post-close vertical lift in robot base frame.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    camera_coordinates_dir = Path(args.camera_coordinates_dir).expanduser().resolve()
    if args.calibration_mode == "hand_eye":
        calibration = load_hand_eye_calibration(
            Path(args.hand_eye_calibration).expanduser().resolve(),
            args.tcp_camera_translation_offset_mm,
        )
    else:
        calibration = load_legacy_plate_calibration(camera_coordinates_dir)

    if args.execute:
        prepare_robot(args)
    capture_was_reused = bool(args.reuse_capture)
    frame = (
        load_captured_frame(output_dir)
        if capture_was_reused
        else capture_realsense(output_dir, args.warmup_frames, args.camera_serial, args.camera_index)
    )
    if calibration["mode"] == "hand_eye":
        capture_tcp_pose = resolve_capture_tcp_pose(output_dir, args, capture_was_reused)
        calibration["capture_tcp_pose"] = capture_tcp_pose
        calibration["base_from_tcp_capture"] = jaka_pose_to_transform(capture_tcp_pose)
    full_cloud, grasp_cloud, full_cloud_rgb, grasp_cloud_rgb, obstacle_cloud, point_cloud_info = build_grasp_point_cloud(
        frame,
        output_dir,
        args,
    )
    cloud_sampled = sample_points(grasp_cloud, args.num_points)
    np.save(output_dir / "point_cloud_camera.npy", full_cloud.astype(np.float32, copy=False))
    np.save(output_dir / "point_cloud_camera_rgb.npy", full_cloud_rgb.astype(np.uint8, copy=False))

    grasps = run_graspnet(cloud_sampled, checkpoint_path, args.device)
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
    )

    if args.execute:
        if args.candidate_index < 0 or args.candidate_index >= len(records):
            raise ValueError(f"--candidate-index out of range: {args.candidate_index}")
        execute_grasp_sequence(records[args.candidate_index], args)

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
        "num_obstacle_points": point_cloud_info.get("num_obstacle_points"),
        "grasp_candidates_png": str(output_dir / "grasp_candidates.png"),
        "grasp_candidates_ply": str(output_dir / "grasp_candidates.ply"),
        "grasp_candidates_3d_html": str(output_dir / "grasp_candidates_3d.html"),
        "grasp_candidates_json": str(output_dir / "grasp_candidates.json"),
        "num_candidates": len(records),
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
        "executed": bool(args.execute),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
