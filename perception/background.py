"""Background detection from depth map: seed generation + HSV color expansion."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from SmartGrasp.perception._shared import _clean_mask, _dilate_mask, _valid_depth_values

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

DEPTH_BACKGROUND_THRESHOLD = 79.802 - 0.05
DEPTH_BACKGROUND_EXPANSION_MIN = 73.5
DEPTH_BACKGROUND_EXPANSION_ANCHOR = 78.5
BACKGROUND_EXPANSION_HUE_TOLERANCE = 14.0
BACKGROUND_EXPANSION_MIN_ANCHOR_FRACTION = 0.025
BACKGROUND_EXPANSION_MAX_HUE_MODES = 4
BACKGROUND_EXPANSION_MAX_COMPONENT_AREA_RATIO = 0.23
LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD = 0.5


def _hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    diff = np.abs(hue.astype(np.float32) - float(center))
    return np.minimum(diff, 180.0 - diff)


def _seed_hue_profiles(hsv: np.ndarray, seed_mask: np.ndarray) -> list[tuple[float, float]]:
    saturation = hsv[..., 1]
    seed_chromatic = seed_mask & (saturation >= 35)
    if int(np.count_nonzero(seed_chromatic)) < 500:
        return []

    hues = hsv[..., 0][seed_chromatic].astype(np.int32)
    hist = np.bincount(hues, minlength=180).astype(np.float32)
    hist_total = float(hist.sum())
    if hist_total <= 0:
        return []

    padded = np.concatenate([hist[-3:], hist, hist[:3]])
    smoothed = np.convolve(padded, np.ones(7, dtype=np.float32), mode="valid")
    working = smoothed.copy()
    profiles: list[tuple[float, float]] = []
    min_mode_weight = max(500.0, hist_total * 0.025)

    for _ in range(BACKGROUND_EXPANSION_MAX_HUE_MODES):
        mode_idx = int(np.argmax(working)) % 180
        if float(working[mode_idx]) < min_mode_weight:
            break

        hue_center = float(mode_idx)
        mode_mask = seed_chromatic & (_hue_distance(hsv[..., 0], hue_center) <= BACKGROUND_EXPANSION_HUE_TOLERANCE)
        if int(np.count_nonzero(mode_mask)) >= 100:
            saturation_floor = float(max(25.0, np.percentile(saturation[mode_mask], 10) * 0.45))
            profiles.append((hue_center, saturation_floor))

        suppress = _hue_distance(np.arange(180, dtype=np.float32), hue_center) <= BACKGROUND_EXPANSION_HUE_TOLERANCE
        working[suppress] = 0.0

    return profiles


def _expand_background_from_seed(
    seed_mask: np.ndarray,
    image: Image.Image | None,
    depth_map: np.ndarray,
    mask_clean_kernel: int,
) -> np.ndarray:
    if image is None or cv2 is None or int(np.count_nonzero(seed_mask)) == 0:
        return seed_mask

    image_np = np.asarray(image.convert("RGB"))
    if image_np.shape[:2] != seed_mask.shape:
        return seed_mask

    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hue_profiles = _seed_hue_profiles(hsv, seed_mask)
    if not hue_profiles:
        return seed_mask

    depth = np.asarray(depth_map, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0)
    chromatic_candidate = np.zeros(seed_mask.shape, dtype=bool)
    for hue_center, saturation_floor in hue_profiles:
        chromatic_candidate |= (
            (_hue_distance(hsv[..., 0], hue_center) <= BACKGROUND_EXPANSION_HUE_TOLERANCE)
            & (hsv[..., 1].astype(np.float32) >= saturation_floor)
        )
    candidate = (
        valid_depth
        & (depth >= DEPTH_BACKGROUND_EXPANSION_MIN)
        & chromatic_candidate
    )

    component_source = candidate & ~seed_mask
    if int(np.count_nonzero(component_source)) == 0:
        return seed_mask

    seed_kernel = np.ones((9, 9), dtype=np.uint8)
    seed_neighborhood = cv2.dilate(seed_mask.astype(np.uint8), seed_kernel, iterations=1) > 0
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        component_source.astype(np.uint8),
        connectivity=8,
    )

    expanded = seed_mask.copy()
    image_area = float(seed_mask.shape[0] * seed_mask.shape[1])
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < 64 or area / image_area > BACKGROUND_EXPANSION_MAX_COMPONENT_AREA_RATIO:
            continue

        component = labels == label_idx
        if not np.any(component & seed_neighborhood):
            continue

        component_depth = depth[component]
        anchor_fraction = float(np.mean(component_depth >= DEPTH_BACKGROUND_EXPANSION_ANCHOR))
        if anchor_fraction < BACKGROUND_EXPANSION_MIN_ANCHOR_FRACTION:
            continue

        expanded |= component

    return _clean_mask(expanded, mask_clean_kernel)


def generate_background_exclusion_mask(
    depth_map: np.ndarray,
    image: Image.Image | None = None,
    mask_clean_kernel: int = 3,
) -> np.ndarray:
    """Use the tray-bottom depth plane as seed, then add adjacent tray/background edges."""
    depth = np.asarray(depth_map, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0)
    background = valid_depth & (depth >= DEPTH_BACKGROUND_THRESHOLD)
    background = _clean_mask(background, mask_clean_kernel)
    return _expand_background_from_seed(background, image, depth, mask_clean_kernel)


def background_overlap_fraction(mask: np.ndarray, background_mask: np.ndarray | None) -> float:
    if background_mask is None or int(np.count_nonzero(background_mask)) == 0:
        return 0.0
    mask_bool = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask_bool))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(mask_bool & np.asarray(background_mask, dtype=bool)) / area)


def mask_boundary_depth_delta(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    depth_map: np.ndarray,
    boundary_kernel_size: int = 3,
) -> tuple[float | None, int]:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    dilated_a = _dilate_mask(a, boundary_kernel_size)
    dilated_b = _dilate_mask(b, boundary_kernel_size)

    # The boundary samples are the pixels from each mask that touch the other
    # mask after dilation. Comparing medians on those two thin strips estimates
    # the physical depth step across the visual seam, while avoiding unrelated
    # depth values from the rest of either object.
    boundary_a = a & dilated_b
    boundary_b = b & dilated_a
    contact_pixels = int(np.count_nonzero(boundary_a) + np.count_nonzero(boundary_b))
    if contact_pixels == 0:
        return None, 0

    depth_a = _valid_depth_values(depth_map, boundary_a)
    depth_b = _valid_depth_values(depth_map, boundary_b)
    if depth_a.size == 0 or depth_b.size == 0:
        return None, contact_pixels

    return float(abs(np.median(depth_a) - np.median(depth_b))), contact_pixels


