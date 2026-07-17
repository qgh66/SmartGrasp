"""Background detection from depth map: per-pixel depth match + depth seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_BACKGROUND_THRESHOLD = 79.802 - 0.05
BACKGROUND_OVERLAP_REJECTION_THRESHOLD = 0.5

_TRAY_BORDER_MASK_PATH = Path(__file__).resolve().parents[1] / "data" / "tray_border_mask.png"
_TRAY_BORDER_DEPTH_PATH = Path(__file__).resolve().parents[1] / "data" / "tray_border_depth.npy"
_TRAY_DEPTH_TOLERANCE = 0.01

_reference_tray_depth_cache: np.ndarray | None = None


def _load_tray_border_mask(shape: tuple[int, int]) -> np.ndarray | None:
    """Load tray_border_mask.png; return None if missing or size mismatch."""
    if not _TRAY_BORDER_MASK_PATH.exists():
        return None
    try:
        mask = np.array(Image.open(_TRAY_BORDER_MASK_PATH).convert("L"))
    except Exception:
        return None
    if mask.shape[:2] != shape:
        return None
    return mask > 128


def _get_reference_tray_depth(shape: tuple[int, int]) -> np.ndarray | None:
    """Load cached per-pixel reference tray depth from scene_4992 calibration."""
    global _reference_tray_depth_cache
    if _reference_tray_depth_cache is not None:
        if _reference_tray_depth_cache.shape[:2] == shape:
            return _reference_tray_depth_cache
        return None
    if not _TRAY_BORDER_DEPTH_PATH.exists():
        return None
    try:
        ref = np.load(_TRAY_BORDER_DEPTH_PATH).astype(np.float32)
    except Exception:
        return None
    if ref.shape[:2] != shape:
        return None
    _reference_tray_depth_cache = ref
    return ref


def generate_background_exclusion_mask(
    depth_map: np.ndarray,
    image: Image.Image | None = None,
    mask_clean_kernel: int = 3,
) -> np.ndarray:
    """Generate background exclusion mask via per-pixel depth matching.

    Inside tray_border_mask: pixel is background if its depth matches the
    per-pixel reference depth (from scene_4992 calibration).
    Outside tray_border_mask: pixel is background if depth >= threshold
    or depth is invalid.
    """
    depth = np.asarray(depth_map, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0)

    # Outside tray: depth seed (far plane + invalid depth)
    outside = (valid_depth & (depth >= DEPTH_BACKGROUND_THRESHOLD)) | ~valid_depth

    tray_border_mask = _load_tray_border_mask(depth.shape)
    ref = _get_reference_tray_depth(depth.shape)

    if tray_border_mask is None or ref is None:
        return outside

    # Inside tray: per-pixel depth match against reference
    inside = tray_border_mask & (np.abs(depth - ref) <= _TRAY_DEPTH_TOLERANCE)

    return outside | inside


def generate_gt_background_exclusion_mask(instances_objects: np.ndarray) -> np.ndarray:
    """Build an exclusion mask from GT object instance ids.

    Pixels with object id > 0 are foreground. Every remaining pixel is treated
    as background/tray/table exclusion area.
    """
    instances = np.asarray(instances_objects)
    if instances.ndim != 2:
        raise ValueError(f"GT instances_objects must be 2D, got shape {instances.shape}.")
    return instances <= 0


def generate_background_exclusion_mask_from_source(
    mask_source: str,
    depth_map: np.ndarray,
    image: Image.Image | None = None,
    instances_objects: np.ndarray | None = None,
    mask_clean_kernel: int = 3,
) -> np.ndarray:
    """Generate a background exclusion mask from the requested source."""
    depth = np.asarray(depth_map, dtype=np.float32)

    if mask_source == "gt":
        if instances_objects is None:
            raise ValueError("mask_source='gt' requires instances_objects.")
        background = generate_gt_background_exclusion_mask(instances_objects)
    elif mask_source == "depth":
        background = generate_background_exclusion_mask(
            depth_map=depth,
            image=image,
            mask_clean_kernel=mask_clean_kernel,
        )
    else:
        raise ValueError(f"Unsupported background mask source: {mask_source}")

    if background.shape != depth.shape:
        raise ValueError(
            f"Background mask shape must match depth shape: {background.shape} vs {depth.shape}."
        )
    return np.asarray(background, dtype=bool)


def background_overlap_fraction(mask: np.ndarray, background_mask: np.ndarray | None) -> float:
    if background_mask is None or int(np.count_nonzero(background_mask)) == 0:
        return 0.0
    mask_bool = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask_bool))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(mask_bool & np.asarray(background_mask, dtype=bool)) / area)


def exclude_background_pixels(mask: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
    """Return a copy of *mask* with all background pixels removed.

    This is a pure boolean operation — it does not mutate the inputs.
    The result is `mask AND NOT background_mask`.
    """
    return np.asarray(mask, dtype=bool) & ~np.asarray(background_mask, dtype=bool)
