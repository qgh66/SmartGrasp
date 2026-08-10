"""Camera, mask, and point-cloud helpers for real-world grasping."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from utils.data_loader import json_safe

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEPTH_AVERAGING_WINDOW_SECONDS = 1.0
_SAM_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}

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


def _capture_averaged_depth(pipeline: Any, align: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Average nonzero aligned depth samples collected over a fixed time window."""
    depth_sum = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint64)
    valid_sample_count = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint16)
    frame_count = 0
    started_at = time.monotonic()
    deadline = started_at + DEPTH_AVERAGING_WINDOW_SECONDS

    while time.monotonic() < deadline:
        aligned_frames = align.process(pipeline.wait_for_frames())
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame:
            continue

        depth_sample = np.asanyarray(depth_frame.get_data())
        if depth_sample.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise RuntimeError(f"Depth resolution must be 1280x720, got {depth_sample.shape[::-1]}.")

        valid_sample = depth_sample > 0
        depth_sum += depth_sample
        valid_sample_count += valid_sample
        frame_count += 1

    if frame_count == 0:
        raise RuntimeError("RealSense did not return a valid aligned depth frame during the averaging window.")

    averaged_depth = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint16)
    valid_pixel = valid_sample_count > 0
    # Add half the divisor before integer division so the mean is rounded to
    # the nearest raw Z16 unit instead of always being rounded down.
    averaged_depth[valid_pixel] = (
        (depth_sum[valid_pixel] + valid_sample_count[valid_pixel] // 2)
        // valid_sample_count[valid_pixel]
    ).astype(np.uint16)

    elapsed_seconds = time.monotonic() - started_at
    averaging_meta = {
        "method": "mean_of_nonzero_aligned_depth_samples",
        "requested_duration_seconds": DEPTH_AVERAGING_WINDOW_SECONDS,
        "actual_duration_seconds": elapsed_seconds,
        "frame_count": frame_count,
        "pixels_without_valid_depth": int(np.count_nonzero(~valid_pixel)),
    }
    return averaged_depth, averaging_meta


def capture_realsense(output_dir: Path, warmup_frames: int, camera_serial: str | None, camera_index: int) -> dict[str, Any]:
    """Capture one RGB frame and a one-second averaged aligned depth frame."""
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

        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense did not return an aligned color frame.")

        # RGB is captured once after warmup. Depth is then sampled for one
        # second and averaged per pixel using only nonzero (valid) readings.
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        if color_bgr.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise RuntimeError(f"RGB resolution must be 1280x720, got {color_bgr.shape[1]}x{color_bgr.shape[0]}.")
        depth_raw, depth_averaging = _capture_averaged_depth(pipeline, align)
        print(
            "Depth averaging complete: "
            f"{depth_averaging['frame_count']} frames over "
            f"{depth_averaging['actual_duration_seconds']:.3f}s, "
            f"{depth_averaging['pixels_without_valid_depth']} pixels had no valid depth."
        )

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
            "depth_averaging": depth_averaging,
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
        json.dumps(json_safe(prompt_payload), ensure_ascii=False, indent=2),
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

    if not args.use_sam_mask and getattr(args, "perception_mask", None) is None:
        return full_cloud, full_cloud, full_cloud_rgb, full_cloud_rgb, full_cloud, {
            "point_cloud_source": "full_depth",
            "grasp_point_cloud_source": "full_depth",
            "grasp_input_mode": "full_depth",
            "obstacle_point_cloud_source": "full_depth_fallback_no_object_mask",
            "num_grasp_input_points": int(len(full_cloud)),
            "num_obstacle_points": int(len(full_cloud)),
            "object_mask": None,
            "camera_intrinsics": frame["meta"]["intrinsics"],
        }

    # --- perception mask (pre-computed SAM2 from perception pipeline) ---
    perception_mask_path = getattr(args, "perception_mask", None)
    if perception_mask_path is not None:
        perception_mask_path = Path(perception_mask_path)
        if not perception_mask_path.exists():
            raise FileNotFoundError(f"Perception mask not found: {perception_mask_path}")
        sam_mask = np.asarray(Image.open(perception_mask_path).convert("L")) > 0
        if sam_mask.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            sam_mask = cv2.resize(sam_mask.astype(np.uint8), (IMAGE_WIDTH, IMAGE_HEIGHT),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        object_mask_info = {
            "mode": "perception_pipeline",
            "mask_path": str(perception_mask_path.resolve()),
            "mask_pixels": int(np.count_nonzero(sam_mask)),
            "mask_overlay_path": None,
        }
    else:
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
    bbox_input_mask = valid & grasp_crop_mask
    bbox_cloud = xyz_image[bbox_input_mask]
    bbox_cloud_rgb = color_rgb_image[bbox_input_mask]
    if len(bbox_cloud) == 0:
        raise RuntimeError("Expanded object crop produced no valid depth points for GraspNet.")

    if args.grasp_input_mode == "mask":
        grasp_cloud = object_cloud
        grasp_cloud_rgb = object_cloud_rgb
        point_cloud_source = "interactive_sam_mask"
        grasp_point_cloud_source = "valid_depth_inside_interactive_sam_mask"
    else:
        grasp_cloud = bbox_cloud
        grasp_cloud_rgb = bbox_cloud_rgb
        point_cloud_source = "interactive_sam_expanded_bbox_crop"
        grasp_point_cloud_source = "valid_depth_inside_expanded_object_bbox"

    crop_overlay_path = save_grasp_crop_overlay(frame["color_bgr"], target_mask, object_bbox, grasp_bbox, output_dir)
    object_mask_info["num_depth_valid_mask_pixels"] = int(np.count_nonzero(object_mask))
    object_mask_info["num_object_points"] = int(len(object_cloud))
    object_mask_info["object_bbox_xyxy"] = [int(value) for value in object_bbox]
    object_mask_info["grasp_crop_bbox_xyxy"] = [int(value) for value in grasp_bbox]
    object_mask_info["grasp_crop_margin_px"] = int(args.grasp_crop_margin_px)
    object_mask_info["grasp_crop_margin_ratio"] = float(args.grasp_crop_margin_ratio)
    object_mask_info["num_grasp_crop_points"] = int(len(bbox_cloud))
    object_mask_info["grasp_input_mode"] = args.grasp_input_mode
    object_mask_info["num_grasp_input_points"] = int(len(grasp_cloud))
    object_mask_info["num_obstacle_points"] = int(len(obstacle_cloud))
    np.save(output_dir / "point_cloud_object_camera.npy", object_cloud.astype(np.float32, copy=False))
    np.save(output_dir / "point_cloud_grasp_input_camera.npy", grasp_cloud.astype(np.float32, copy=False))
    np.save(output_dir / "point_cloud_obstacles_camera.npy", obstacle_cloud.astype(np.float32, copy=False))
    print(
        "[object-mask] saved mask.png and selectable GraspNet input; "
        f"grasp_input_mode={args.grasp_input_mode} "
        f"mask_pixels={object_mask_info['mask_pixels']} "
        f"object_points={len(object_cloud)} "
        f"grasp_crop_points={len(bbox_cloud)} "
        f"grasp_input_points={len(grasp_cloud)} "
        f"obstacle_points={len(obstacle_cloud)} "
        f"grasp_crop_bbox={object_mask_info['grasp_crop_bbox_xyxy']} "
        f"mode={object_mask_info['mode']}",
        flush=True,
    )
    return full_cloud, grasp_cloud, full_cloud_rgb, grasp_cloud_rgb, obstacle_cloud, {
        "point_cloud_source": point_cloud_source,
        "grasp_point_cloud_source": grasp_point_cloud_source,
        "grasp_input_mode": args.grasp_input_mode,
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
        "num_grasp_crop_points": int(len(bbox_cloud)),
        "num_obstacle_points": int(len(obstacle_cloud)),
        "object_mask": object_mask_info,
        "object_mask_array": target_mask,
        "camera_intrinsics": frame["meta"]["intrinsics"],
    }
