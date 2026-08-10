"""Depth-based background detection using a RANSAC plane half-space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


BACKGROUND_OVERLAP_REJECTION_THRESHOLD = 0.5


@dataclass(frozen=True)
class PlaneBackgroundResult:
    """Background mask and diagnostics for the fitted plane half-space."""

    mask: np.ndarray
    far_side_mask: np.ndarray
    valid_depth_mask: np.ndarray
    signed_distance_cm: np.ndarray
    plane_normal: np.ndarray
    plane_offset_cm: float
    plane_center_cm: np.ndarray
    sampled_point_count: int
    ransac_inlier_count: int
    ransac_inlier_ratio: float
    camera_side_extension_cm: float
    intrinsics: dict[str, float]
    intrinsics_source: str


def _resolve_intrinsics(
    shape: tuple[int, int],
    camera_intrinsics: dict[str, Any] | None,
) -> tuple[dict[str, float], str]:
    height, width = shape
    if camera_intrinsics is not None:
        try:
            intrinsics = {
                "fx": float(camera_intrinsics["fx"]),
                "fy": float(camera_intrinsics["fy"]),
                "cx": float(camera_intrinsics["cx"]),
                "cy": float(camera_intrinsics["cy"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "camera_intrinsics must contain numeric fx, fy, cx and cy"
            ) from exc
        if (
            not all(np.isfinite(value) for value in intrinsics.values())
            or intrinsics["fx"] <= 0.0
            or intrinsics["fy"] <= 0.0
        ):
            raise ValueError(f"Invalid camera intrinsics: {intrinsics}")
        return intrinsics, "camera_meta"

    # Dataset scenes do not always carry camera metadata. A centered pinhole
    # approximation keeps XYZ in the same units as depth and still permits a
    # metric RANSAC plane fit; real camera scenes should always pass metadata.
    focal = float(max(width, height))
    return {
        "fx": focal,
        "fy": focal,
        "cx": (float(width) - 1.0) / 2.0,
        "cy": (float(height) - 1.0) / 2.0,
    }, "centered_pinhole_approximation"


def _depth_to_points_cm(
    depth_cm: np.ndarray,
    valid: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(valid)
    z = depth_cm[rows, cols]
    x = (cols.astype(np.float32) - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (rows.astype(np.float32) - intrinsics["cy"]) * z / intrinsics["fy"]
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    return points, rows, cols


def _fit_ransac_plane(
    points: np.ndarray,
    *,
    iterations: int,
    distance_threshold_cm: float,
    max_points: int,
    random_seed: int,
) -> tuple[np.ndarray, float, np.ndarray, int, float]:
    if points.shape[0] < 3:
        raise ValueError("At least three valid depth points are required for RANSAC")
    if iterations < 1:
        raise ValueError("RANSAC iterations must be at least 1")
    if distance_threshold_cm <= 0.0:
        raise ValueError("RANSAC distance threshold must be positive")
    if max_points < 3:
        raise ValueError("RANSAC max_points must be at least 3")

    rng = np.random.default_rng(random_seed)
    sample_count = min(int(max_points), int(points.shape[0]))
    if sample_count < points.shape[0]:
        sampled = points[rng.choice(points.shape[0], sample_count, replace=False)]
    else:
        sampled = points

    best_count = 0
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    for _ in range(iterations):
        p0, p1, p2 = sampled[rng.choice(sample_count, 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-6:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, p0))
        count = int(
            np.count_nonzero(np.abs(sampled @ normal + offset) <= distance_threshold_cm)
        )
        if count > best_count:
            best_count = count
            best_normal = normal.astype(np.float32, copy=False)
            best_offset = offset

    if best_normal is None or best_count < 3:
        raise RuntimeError("RANSAC could not fit a valid depth plane")

    initial_inliers = (
        np.abs(sampled @ best_normal + best_offset) <= distance_threshold_cm
    )
    inlier_points = sampled[initial_inliers]
    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1].astype(np.float32, copy=False)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, center))

    refined_inliers = np.abs(sampled @ normal + offset) <= distance_threshold_cm
    refined_points = sampled[refined_inliers]
    if refined_points.shape[0] >= 3:
        center = refined_points.mean(axis=0)
        _, _, vh = np.linalg.svd(refined_points - center, full_matrices=False)
        normal = vh[-1].astype(np.float32, copy=False)
        normal /= np.linalg.norm(normal)
        if normal[2] < 0.0:
            normal = -normal
        offset = -float(np.dot(normal, center))
        refined_inliers = (
            np.abs(sampled @ normal + offset) <= distance_threshold_cm
        )
        refined_points = sampled[refined_inliers]

    return (
        normal,
        offset,
        refined_points,
        sample_count,
        float(refined_points.shape[0] / sample_count),
    )


def fit_plane_background_mask(
    depth_map: np.ndarray,
    *,
    camera_intrinsics: dict[str, Any] | None = None,
    ransac_iterations: int = 300,
    ransac_distance_threshold_cm: float = 0.3,
    ransac_max_points: int = 30000,
    camera_side_extension_cm: float = 0.3,
    min_ransac_inlier_ratio: float = 0.1,
    random_seed: int = 42,
) -> PlaneBackgroundResult:
    """Fit the dominant plane and classify its farther half-space as background.

    ``depth_map`` must use centimetres, matching the Perception depth contract.
    The fitted normal is oriented toward increasing camera depth. Moving the
    decision plane toward the camera by ``camera_side_extension_cm`` means a
    valid point is background when its signed distance from the original plane
    is greater than or equal to ``-camera_side_extension_cm``. Invalid depth
    pixels remain unknown and are not included in the background mask.
    """
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D, got shape {depth.shape}")
    if camera_side_extension_cm < 0.0:
        raise ValueError("Camera-side plane extension must be non-negative")
    if not 0.0 < min_ransac_inlier_ratio <= 1.0:
        raise ValueError("Minimum RANSAC inlier ratio must be in (0, 1]")

    valid = np.isfinite(depth) & (depth > 0.0)
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("Depth map contains fewer than three valid pixels")

    intrinsics, intrinsics_source = _resolve_intrinsics(depth.shape, camera_intrinsics)
    points, rows, cols = _depth_to_points_cm(depth, valid, intrinsics)
    normal, offset, sampled_inliers, sampled_count, inlier_ratio = _fit_ransac_plane(
        points,
        iterations=ransac_iterations,
        distance_threshold_cm=ransac_distance_threshold_cm,
        max_points=ransac_max_points,
        random_seed=random_seed,
    )
    if inlier_ratio < min_ransac_inlier_ratio:
        raise RuntimeError(
            "Dominant plane is too weak: "
            f"inlier_ratio={inlier_ratio:.4f} < {min_ransac_inlier_ratio:.4f}"
        )

    center = sampled_inliers.mean(axis=0).astype(np.float32, copy=False)
    offset = -float(np.dot(normal, center))
    signed_point_distance = (points @ normal + offset).astype(np.float32, copy=False)
    far_side_points = signed_point_distance >= -float(camera_side_extension_cm)
    far_side_mask = np.zeros(depth.shape, dtype=bool)
    far_side_mask[rows[far_side_points], cols[far_side_points]] = True
    # A missing depth measurement carries no geometric evidence that the RGB
    # pixel is background. Only valid points classified on the plane's far
    # side are safe to exclude from object proposals.
    background = far_side_mask.copy()
    signed_distance = np.full(depth.shape, np.nan, dtype=np.float32)
    signed_distance[rows, cols] = signed_point_distance

    return PlaneBackgroundResult(
        mask=background,
        far_side_mask=far_side_mask,
        valid_depth_mask=valid,
        signed_distance_cm=signed_distance,
        plane_normal=normal,
        plane_offset_cm=offset,
        plane_center_cm=center,
        sampled_point_count=sampled_count,
        ransac_inlier_count=int(sampled_inliers.shape[0]),
        ransac_inlier_ratio=inlier_ratio,
        camera_side_extension_cm=float(camera_side_extension_cm),
        intrinsics=intrinsics,
        intrinsics_source=intrinsics_source,
    )


def generate_background_exclusion_mask(
    depth_map: np.ndarray,
    image: Image.Image | None = None,
    mask_clean_kernel: int = 3,
    camera_intrinsics: dict[str, Any] | None = None,
    camera_side_extension_cm: float = 0.3,
) -> np.ndarray:
    """Return the RANSAC-plane farther-half-space background mask."""
    del image, mask_clean_kernel  # Kept for compatibility with existing callers.
    return fit_plane_background_mask(
        depth_map,
        camera_intrinsics=camera_intrinsics,
        camera_side_extension_cm=camera_side_extension_cm,
    ).mask


def generate_gt_background_exclusion_mask(instances_objects: np.ndarray) -> np.ndarray:
    """Build an exclusion mask from GT object instance ids."""
    instances = np.asarray(instances_objects)
    if instances.ndim != 2:
        raise ValueError(f"GT instances_objects must be 2D, got shape {instances.shape}.")
    return instances <= 0


def background_overlap_fraction(mask: np.ndarray, background_mask: np.ndarray | None) -> float:
    if background_mask is None or int(np.count_nonzero(background_mask)) == 0:
        return 0.0
    mask_bool = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask_bool))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(mask_bool & np.asarray(background_mask, dtype=bool)) / area)


def exclude_background_pixels(mask: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
    """Return ``mask AND NOT background_mask`` without mutating inputs."""
    return np.asarray(mask, dtype=bool) & ~np.asarray(background_mask, dtype=bool)
