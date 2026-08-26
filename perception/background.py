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
    reference_depth_map: np.ndarray | None = None,
    reference_image: Image.Image | np.ndarray | None = None,
    reference_depth_tolerance: float = 0.004,
    reference_rgb_tolerance: int = 18,
) -> np.ndarray:
    """Generate background exclusion mask via per-pixel depth matching.

    Inside tray_border_mask: pixel is background if its depth matches the
    per-pixel reference depth (from scene_4992 calibration).
    Outside tray_border_mask: pixel is background if depth >= threshold
    or depth is invalid.
    """
    depth = np.asarray(depth_map, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0)

    if reference_depth_map is not None:
        reference_depth = np.asarray(reference_depth_map, dtype=np.float32)
        if reference_depth.shape != depth.shape:
            raise ValueError(
                "Captured background depth shape does not match scene depth: "
                f"background={reference_depth.shape}, scene={depth.shape}"
            )
        reference_valid = np.isfinite(reference_depth) & (reference_depth > 0)
        depth_delta = np.abs(depth - reference_depth)
        depth_matches = (
            valid_depth
            & reference_valid
            & (depth_delta <= float(reference_depth_tolerance))
        )
        background = ~valid_depth

        if image is not None and reference_image is not None:
            scene_rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
            if isinstance(reference_image, Image.Image):
                reference_rgb = np.asarray(
                    reference_image.convert("RGB"),
                    dtype=np.int16,
                )
            else:
                reference_rgb = np.asarray(reference_image, dtype=np.int16)
                if reference_rgb.ndim == 3 and reference_rgb.shape[2] == 4:
                    reference_rgb = reference_rgb[..., :3]
            expected_rgb_shape = (*depth.shape, 3)
            if (
                scene_rgb.shape != expected_rgb_shape
                or reference_rgb.shape != expected_rgb_shape
            ):
                raise ValueError(
                    "Captured background RGB shape does not match scene image: "
                    f"background={reference_rgb.shape}, scene={scene_rgb.shape}, "
                    f"expected={expected_rgb_shape}"
                )
            rgb_delta = np.max(np.abs(scene_rgb - reference_rgb), axis=2)
            rgb_matches = rgb_delta <= int(reference_rgb_tolerance)
            exact_depth_matches = (
                valid_depth
                & reference_valid
                & (depth_delta <= 0.0005)
            )
            background |= depth_matches & (rgb_matches | exact_depth_matches)
        else:
            background |= depth_matches
        return np.asarray(background, dtype=bool)

    # Outside tray: depth seed (far plane + invalid depth)
    outside = (valid_depth & (depth >= DEPTH_BACKGROUND_THRESHOLD)) | ~valid_depth

    tray_border_mask = _load_tray_border_mask(depth.shape)
    ref = _get_reference_tray_depth(depth.shape)

    if tray_border_mask is None or ref is None:
        return outside

    # Inside tray: per-pixel depth match against reference
    inside = tray_border_mask & (np.abs(depth - ref) <= _TRAY_DEPTH_TOLERANCE)

    return outside | inside


def remove_background_from_image(
    image: Image.Image,
    background_mask: np.ndarray,
    fill_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Replace captured background pixels with a uniform RGB color."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(background_mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(
            "Background mask shape does not match image: "
            f"mask={mask.shape}, image={rgb.shape[:2]}"
        )
    rgb[mask] = np.asarray(fill_rgb, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def generate_gt_background_exclusion_mask(instances_objects: np.ndarray) -> np.ndarray:
    """Build an exclusion mask from GT object instance ids.

    Pixels with object id > 0 are foreground. Every remaining pixel is treated
    as background/tray/table exclusion area.
    """
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
    """Return a copy of *mask* with all background pixels removed.

    This is a pure boolean operation — it does not mutate the inputs.
    The result is `mask AND NOT background_mask`.
    """
    return np.asarray(mask, dtype=bool) & ~np.asarray(background_mask, dtype=bool)
