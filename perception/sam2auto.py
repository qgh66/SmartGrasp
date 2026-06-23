"""SAM2 automatic mask generation: model loading, candidate pool, scoring, visualization, pipeline orchestrator."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from SmartGrasp.perception._shared import (
    _as_numpy_mask, _box_xywh_to_xyxy, _candidate_summary,
    _clean_mask, _draw_labeled_image_matplotlib, _log_step,
    _mask_bbox, _mask_centroid_xy, _mask_iou, _mask_overlap_fraction,
    _proposal_label, _safe_label, _save_mask_png, _dilate_mask,
    SMARTGRASP_ROOT,
)
from SmartGrasp.perception.background import (
    background_overlap_fraction, mask_boundary_depth_delta,
    LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD,
)
from SmartGrasp.perception.vlm_2_assemble import _openai_review_sam2_candidates
from SmartGrasp.perception.langsam import _langsam_predict, _select_langsam_mask_for_review_object, LANGSAM_MIN_AREA_RATIO

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LANGSAM_CACHE: dict[tuple[str, str], Any] = {}
GROUNDING_DINO_LOCAL_MODEL_PATH = Path(
    os.environ.get(
        "GROUNDING_DINO_MODEL_PATH",
        str(Path.home() / ".cache/huggingface/hub/models--IDEA-Research--grounding-dino-base/"
        "snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"),
    )
)

def _load_langsam(device: str) -> Any:
    cache_key = ("default", device)
    if cache_key not in _LANGSAM_CACHE:
        try:
            from lang_sam import LangSAM  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LangSAM is not installed in the active environment. "
                "Install lang-segment-anything/LangSAM before using --segmentation-backend langsam."
            ) from exc

        if GROUNDING_DINO_LOCAL_MODEL_PATH.exists():
            try:
                model = LangSAM(
                    device=device,
                    gdino_model_ckpt_path=str(GROUNDING_DINO_LOCAL_MODEL_PATH),
                    gdino_processor_ckpt_path=str(GROUNDING_DINO_LOCAL_MODEL_PATH),
                )
            except TypeError:
                try:
                    model = LangSAM(
                        gdino_model_ckpt_path=str(GROUNDING_DINO_LOCAL_MODEL_PATH),
                        gdino_processor_ckpt_path=str(GROUNDING_DINO_LOCAL_MODEL_PATH),
                    )
                except TypeError:
                    model = LangSAM(device=device)
        else:
            try:
                model = LangSAM(device=device)
            except TypeError:
                model = LangSAM()
        _LANGSAM_CACHE[cache_key] = model
    return _LANGSAM_CACHE[cache_key]



def _sam2_auto_generate(model: Any, image: Image.Image) -> list[dict[str, Any]]:
    # Fixed seeds for reproducible SAM2 output across Mac/Linux/CUDA/MPS
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    image_np = np.asarray(image.convert("RGB"))
    sam = getattr(model, "sam", None)
    if sam is None or not hasattr(sam, "generate"):
        raise RuntimeError("Loaded LangSAM object does not expose SAM2 automatic mask generation.")
    return list(sam.generate(image_np))


def _configure_sam2_auto_generator(
    model: Any,
    points_per_side: int | None = None,
    crop_n_layers: int | None = None,
    pred_iou_thresh: float | None = None,
    stability_score_thresh: float | None = None,
) -> dict[str, Any]:
    sam = getattr(model, "sam", None)
    generator = getattr(sam, "mask_generator", None)
    sam_model = getattr(sam, "model", None)
    if generator is None or sam_model is None:
        return {"configured": False, "reason": "sam2_mask_generator_unavailable"}

    kwargs: dict[str, Any] = {}
    for name in (
        "points_per_side",
        "points_per_batch",
        "pred_iou_thresh",
        "stability_score_thresh",
        "stability_score_offset",
        "mask_threshold",
        "box_nms_thresh",
        "crop_n_layers",
        "crop_nms_thresh",
        "crop_overlap_ratio",
        "crop_n_points_downscale_factor",
        "min_mask_region_area",
        "output_mode",
    ):
        if hasattr(generator, name):
            kwargs[name] = getattr(generator, name)

    if points_per_side is not None and points_per_side > 0:
        kwargs["points_per_side"] = int(points_per_side)
        kwargs.pop("point_grids", None)
    if crop_n_layers is not None and crop_n_layers >= 0:
        kwargs["crop_n_layers"] = int(crop_n_layers)
    if pred_iou_thresh is not None:
        kwargs["pred_iou_thresh"] = float(pred_iou_thresh)
    if stability_score_thresh is not None:
        kwargs["stability_score_thresh"] = float(stability_score_thresh)

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam.mask_generator = SAM2AutomaticMaskGenerator(sam_model, **kwargs)
    except Exception as exc:
        return {"configured": False, "reason": str(exc), "requested": kwargs}
    return {"configured": True, "settings": kwargs}


def _border_touch_fraction(mask: np.ndarray, border_pixels: int = 3) -> float:
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    height, width = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    border[:border_pixels, :] = True
    border[max(0, height - border_pixels) :, :] = True
    border[:, :border_pixels] = True
    border[:, max(0, width - border_pixels) :] = True
    return float(np.count_nonzero(mask & border) / area)





def _proposal_score(proposal: dict[str, Any], area_ratio: float, border_fraction: float) -> float:
    predicted_iou = float(proposal.get("predicted_iou", proposal.get("iou", 0.0)) or 0.0)
    stability = float(proposal.get("stability_score", 0.0) or 0.0)
    area_prior = -abs(area_ratio - 0.035)
    border_penalty = 0.5 * border_fraction
    return predicted_iou + 0.25 * stability + area_prior - border_penalty



def _is_support_like_horizontal_strip(mask: np.ndarray) -> bool:
    height, width = mask.shape
    x, y, bbox_width, bbox_height = _mask_bbox(mask)
    if bbox_width <= 0 or bbox_height <= 0:
        return False
    aspect = bbox_width / max(1, bbox_height)
    vertical_center = (y + 0.5 * bbox_height) / max(1, height)
    horizontal_coverage = bbox_width / max(1, width)
    thin_height = bbox_height / max(1, height)
    return bool(
        aspect >= 3.0
        and vertical_center >= 0.86
        and horizontal_coverage >= 0.22
        and thin_height <= 0.08
    )


def _is_tray_or_background_like_proposal(
    image_np: np.ndarray,
    mask: np.ndarray,
    background_exclusion_mask: np.ndarray | None = None,
) -> bool:
    x, y, bbox_width, bbox_height = _mask_bbox(mask)
    if bbox_width <= 0 or bbox_height <= 0:
        return True

    # Prefer the pre-computed depth-based background exclusion mask when available.
    if background_exclusion_mask is not None and int(np.count_nonzero(background_exclusion_mask)) > 0:
        background_overlap = background_overlap_fraction(mask, background_exclusion_mask)
        if background_overlap >= LANGSAM_BACKGROUND_OVERLAP_FALLBACK_THRESHOLD:
            return True

    height, width = mask.shape
    area = int(np.count_nonzero(mask))
    bbox_area = bbox_width * bbox_height
    fill = area / max(1, bbox_area)
    aspect = bbox_width / max(1, bbox_height)
    cx = (x + 0.5 * bbox_width) / max(1, width)
    cy = (y + 0.5 * bbox_height) / max(1, height)

    pixels = image_np[mask].astype(np.float32)
    if pixels.size == 0:
        return True
    median_rgb = np.median(pixels, axis=0)
    r, g, b = [float(value) for value in median_rgb]
    greenish = g >= 85 and g >= r * 1.02 and g >= b * 0.92
    low_saturation = (max(r, g, b) - min(r, g, b)) < 35
    near_tray_wall = cx <= 0.18 or cx >= 0.82 or cy <= 0.12 or cy >= 0.88

    if greenish and (near_tray_wall or fill >= 0.78):
        return True
    if greenish and cy >= 0.68 and aspect >= 1.45 and bbox_width / max(1, width) >= 0.25:
        return True
    if low_saturation and near_tray_wall and fill >= 0.72:
        return True
    if aspect >= 3.0 and bbox_width / max(1, width) >= 0.22 and bbox_height / max(1, height) <= 0.08:
        return True
    return False


def _sam2_auto_candidate_pool(
    image_path: Path,
    output_mask_dir: Path,
    min_area_ratio: float,
    max_area_ratio: float,
    border_fraction_threshold: float,
    mask_clean_kernel: int,
    save_candidates: bool,
    device: str | None,
    background_exclusion_mask: np.ndarray | None,
    points_per_side: int | None = None,
    crop_n_layers: int | None = None,
    pred_iou_thresh: float | None = None,
    stability_score_thresh: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, Image.Image]:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    image_area = float(image_np.shape[0] * image_np.shape[1])
    model = _load_langsam(device)
    generator_settings = _configure_sam2_auto_generator(
        model,
        points_per_side=points_per_side,
        crop_n_layers=crop_n_layers,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    raw_proposals = _sam2_auto_generate(model, image)
    candidates: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = [{"stage": "sam2_auto_generator", **generator_settings}]

    effective_min_area_ratio = max(0.0002, min_area_ratio * 0.15)
    effective_max_area_ratio = max(max_area_ratio, 0.45)

    for idx, proposal in enumerate(raw_proposals):
        raw_mask = proposal.get("segmentation")
        if raw_mask is None:
            continue
        mask = _clean_mask(_as_numpy_mask(raw_mask), mask_clean_kernel)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        area_ratio = float(area / image_area)
        border_fraction = _border_touch_fraction(mask)
        background_overlap = background_overlap_fraction(mask, background_exclusion_mask)
        bbox_xywh = [int(value) for value in proposal.get("bbox", _mask_bbox(mask))]
        bbox_xyxy = _box_xywh_to_xyxy(bbox_xywh)
        cx, cy = _mask_centroid_xy(mask)

        reason: str | None = None
        if area_ratio < effective_min_area_ratio:
            reason = "too_small"
        elif area_ratio > effective_max_area_ratio:
            reason = "too_large"
        elif border_fraction > border_fraction_threshold:
            reason = "touches_image_border"
        elif _is_support_like_horizontal_strip(mask) or _is_tray_or_background_like_proposal(image_np, mask, background_exclusion_mask):
            reason = "support_or_tray_like"
        elif background_overlap > 0.5:
            reason = "overlaps_background_exclusion"

        metadata = {
            "proposal_index": int(idx),
            "area": area,
            "area_ratio": area_ratio,
            "bbox": bbox_xywh,
            "bbox_xyxy": bbox_xyxy,
            "predicted_iou": float(proposal.get("predicted_iou", 0.0) or 0.0),
            "stability_score": float(proposal.get("stability_score", 0.0) or 0.0),
            "border_fraction": border_fraction,
            "background_exclusion_overlap": background_overlap,
            "centroid": {"x": int(cx), "y": int(cy)},
        }
        if reason is not None:
            metadata["rejection_reason"] = reason
            report.append(metadata)
            continue

        metadata["selection_score"] = _proposal_score(proposal, area_ratio, border_fraction)
        metadata["mask"] = mask
        candidates.append(metadata)
        report.append({key: value for key, value in metadata.items() if key != "mask"})

    candidates.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
    if save_candidates:
        candidate_dir = output_mask_dir.parent / "sam2_auto_candidates"
        for candidate in candidates:
            candidate_path = candidate_dir / f"proposal_{int(candidate['proposal_index']):03d}.png"
            _save_mask_png(candidate["mask"], candidate_path)
            candidate["mask_path"] = str(candidate_path.resolve())
    return candidates, report, model, image



def _overlay_mask_contours(ax: Any, masks: list[np.ndarray]) -> None:
    if cv2 is None:
        return
    for mask in masks:
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        contours, _hierarchy = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            contour_xy = contour[:, 0, :]
            ax.plot(contour_xy[:, 0], contour_xy[:, 1], color="#00ffff", linewidth=1.0, alpha=0.85)


def _draw_sam2_auto_label_image(
    image_path: Path,
    candidates: list[dict[str, Any]],
    out_path: Path,
    max_labels: int = 80,
) -> None:
    points_with_ids: list[tuple[int, int, int]] = []
    masks: list[np.ndarray] = []
    for index, candidate in enumerate(candidates[:max_labels], start=1):
        centroid = candidate.get("centroid", {})
        x = int(centroid.get("x", 0))
        y = int(centroid.get("y", 0))
        points_with_ids.append((index, x, y))
        masks.append(np.asarray(candidate["mask"], dtype=bool))

    with Image.open(image_path).convert("RGB") as image:
        _draw_labeled_image_matplotlib(image=image, points_with_ids=points_with_ids, out_png_path=out_path)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            _overlay_mask_contours(ax, masks)
            for obj_id, x, y in points_with_ids:
                ax.text(
                    x,
                    y,
                    str(obj_id),
                    color="yellow",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                )
            ax.axis("off")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
        except Exception:
            return


def _save_sam2_rgb_parts_sheet(
    image_path: Path,
    candidates: list[dict[str, Any]],
    out_dir: Path,
    max_labels: int = 40,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    parts_dir = out_dir / "sam2_rgb_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_images: list[tuple[int, Image.Image]] = []

    for index, candidate in enumerate(candidates[:max_labels], start=1):
        mask = np.asarray(candidate["mask"], dtype=bool)
        x, y, width, height = _mask_bbox(mask)
        if width <= 0 or height <= 0:
            continue
        pad = max(8, int(round(max(width, height) * 0.08)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image_np.shape[1], x + width + pad)
        y1 = min(image_np.shape[0], y + height + pad)
        crop = image_np[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]
        white = np.full_like(crop, 255)
        visible = np.where(crop_mask[..., None], crop, white)
        part_image = Image.fromarray(visible, mode="RGB")
        part_path = parts_dir / f"part_{index:03d}.png"
        part_image.save(part_path)
        part_images.append((index, part_image))

    if not part_images:
        sheet_path = out_dir / "sam2_rgb_parts_sheet.png"
        Image.new("RGB", (256, 256), "white").save(sheet_path)
        return sheet_path

    thumb_size = 160
    label_height = 26
    columns = 5
    rows = int(np.ceil(len(part_images) / columns))
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "white")
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 2.3))
        axes_arr = np.asarray(axes).reshape(rows, columns)
        for ax in axes_arr.flat:
            ax.axis("off")
        for item_index, (part_id, part_image) in enumerate(part_images):
            row = item_index // columns
            col = item_index % columns
            ax = axes_arr[row, col]
            ax.imshow(part_image)
            ax.set_title(str(part_id), fontsize=10, fontweight="bold")
            ax.axis("off")
        sheet_path = out_dir / "sam2_rgb_parts_sheet.png"
        fig.savefig(sheet_path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        return sheet_path
    except Exception:
        for item_index, (part_id, part_image) in enumerate(part_images):
            row = item_index // columns
            col = item_index % columns
            thumb = part_image.copy()
            thumb.thumbnail((thumb_size, thumb_size))
            x = col * thumb_size + (thumb_size - thumb.width) // 2
            y = row * (thumb_size + label_height) + label_height
            sheet.paste(thumb, (x, y))
        sheet_path = out_dir / "sam2_rgb_parts_sheet.png"
        sheet.save(sheet_path)
        return sheet_path


UNCLAIMED_MIN_AREA_RATIO = 0.0007  # Minimum fraction of image area for an unclaimed SAM2 candidate (0.07%)


def _append_unclaimed_sam2_candidates(
    mask_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    claimed_sam2_ids: set[int],
    image_np: np.ndarray,
    output_mask_dir: Path,
    max_unclaimed: int,
    depth_map: np.ndarray | None = None,
    depth_threshold: float = 0.012,
    min_contact_pixels: int = 4,
    max_area_ratio: float = 0.18,
    min_area_ratio: float = 0.0007,
    iou_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    if max_unclaimed <= 0:
        return mask_records

    image_area = float(image_np.shape[0] * image_np.shape[1])
    existing_masks = [np.asarray(record["mask_array"], dtype=bool) for record in mask_records]
    seed_items: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        if candidate_index in claimed_sam2_ids:
            continue
        mask = np.asarray(candidate["mask"], dtype=bool)
        if any(_mask_iou(mask, existing) > iou_threshold for existing in existing_masks):
            continue
        seed_items.append({"candidate_index": candidate_index, "candidate": candidate, "mask": mask})

    if not seed_items:
        return mask_records

    parent = list(range(len(seed_items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    dilated = [_dilate_mask(item["mask"], 3) for item in seed_items]
    for left in range(len(seed_items)):
        for right in range(left + 1, len(seed_items)):
            left_mask = seed_items[left]["mask"]
            right_mask = seed_items[right]["mask"]
            if not (np.any(dilated[left] & right_mask) or np.any(dilated[right] & left_mask)):
                continue
            should_merge = True
            if depth_map is not None:
                depth_delta, contact_pixels = mask_boundary_depth_delta(
                    left_mask,
                    right_mask,
                    depth_map,
                    boundary_kernel_size=3,
                )
                should_merge = (
                    depth_delta is not None
                    and contact_pixels >= min_contact_pixels
                    and depth_delta <= depth_threshold
                )
            if should_merge:
                union(left, right)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, item in enumerate(seed_items):
        groups.setdefault(find(index), []).append(item)

    grouped_items: list[dict[str, Any]] = []
    for members in groups.values():
        merged_mask = np.any(np.stack([member["mask"] for member in members], axis=0), axis=0)
        area = int(np.count_nonzero(merged_mask))
        area_ratio = float(area / max(1.0, image_area))
        if area_ratio > max_area_ratio or area_ratio < min_area_ratio:
            continue
        best_member = max(members, key=lambda item: float(item["candidate"].get("selection_score", 0.0)))
        grouped_items.append(
            {
                "mask": merged_mask,
                "area": area,
                "area_ratio": area_ratio,
                "candidate_indices": [int(member["candidate_index"]) for member in members],
                "members": [_candidate_summary(member["candidate"]) for member in members],
                "selection_score": max(float(member["candidate"].get("selection_score", 0.0)) for member in members),
                "best_candidate": best_member["candidate"],
            }
        )

    grouped_items.sort(key=lambda item: (int(item["area"]), float(item["selection_score"])), reverse=True)

    appended = 0
    next_id = max([int(record.get("object_id", 0)) for record in mask_records] + [0]) + 1
    for item in grouped_items:
        mask = np.asarray(item["mask"], dtype=bool)
        if any(_mask_iou(mask, existing) > iou_threshold for existing in existing_masks):
            continue
        label = _proposal_label(image_np, mask)
        cx, cy = _mask_centroid_xy(mask)
        object_id = next_id + appended
        proposal_name = "_".join(str(index) for index in item["candidate_indices"][:8])
        filename = f"{object_id:03d}_sam2_group_{proposal_name}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(mask, mask_path)
        mask_records.append(
            {
                "node_id": len(mask_records),
                "object_id": object_id,
                "label": f"sam2 group {proposal_name}",
                "description": label,
                "point": {"x": int(cx), "y": int(cy)},
                "segmentation_backend": "sam2_auto_unclaimed_depth_grouped",
                "sam2_ids": item["candidate_indices"],
                "selected_sam2_candidate": _candidate_summary(item["best_candidate"]),
                "sam2_group_members": item["members"],
                "sam2_group_area_ratio": float(item["area_ratio"]),
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(mask)),
                "mask_array": mask,
            }
        )
        existing_masks.append(mask)
        appended += 1
        if appended >= max_unclaimed:
            break
    return mask_records



def generate_masks_with_sam2_langsam_pipeline(
    image_path: Path,
    output_mask_dir: Path,
    review_model_id: str,
    review_api_key_env: str,
    review_base_url: str | None,
    review_timeout: float,
    min_area_ratio: float,
    max_area_ratio: float,
    proposal_border_fraction_threshold: float,
    mask_clean_kernel: int,
    save_candidates: bool,
    device: str | None,
    background_exclusion_mask: np.ndarray | None,
    depth_map: np.ndarray | None = None,
    sam2_points_per_side: int | None = None,
    sam2_crop_n_layers: int | None = None,
    sam2_pred_iou_thresh: float | None = None,
    sam2_stability_score_thresh: float | None = None,
    preserve_unclaimed_sam2: int = 24,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    t_p0 = _log_step("  ②a sam2_auto", None)

    candidates, sam2_report, model, image = _sam2_auto_candidate_pool(
        image_path=image_path,
        output_mask_dir=output_mask_dir,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        mask_clean_kernel=mask_clean_kernel,
        save_candidates=save_candidates,
        device=device,
        background_exclusion_mask=background_exclusion_mask,
        points_per_side=sam2_points_per_side,
        crop_n_layers=sam2_crop_n_layers,
        pred_iou_thresh=sam2_pred_iou_thresh,
        stability_score_thresh=sam2_stability_score_thresh,
        border_fraction_threshold=proposal_border_fraction_threshold,
    )
    t_p1 = _log_step("  ②a sam2_auto", t_p0)
    out_dir = output_mask_dir.parent
    label_path = out_dir / "label_1_sam2auto.png"
    _draw_sam2_auto_label_image(image_path, candidates, label_path)
    parts_sheet_path = _save_sam2_rgb_parts_sheet(image_path, candidates, out_dir)
    try:
        review_objects, raw_review = _openai_review_sam2_candidates(
            image_path=image_path,
            label_image_path=label_path,
            parts_sheet_path=parts_sheet_path,
            candidates=candidates,
            model_id=review_model_id,
            api_key_env=review_api_key_env,
            base_url=review_base_url,
            timeout=review_timeout,
            out_dir=out_dir,
        )
    except Exception as exc:
        _log_step("  ②b openai_review FAILED", t_p1)
        raise RuntimeError(f"OpenAI review failed — API error. Aborting. Error: {exc}") from exc

    if not review_objects:
        _log_step("  ②b openai_review returned empty", t_p1)
        raise RuntimeError("OpenAI review returned no objects — empty response. Aborting.")

    mask_records: list[dict[str, Any]] = []
    claimed_sam2_ids: set[int] = set()
    with Image.open(image_path).convert("RGB") as source_image:
        image_np = np.asarray(source_image)
        for item in review_objects:
            claimed_sam2_ids.update(int(value) for value in item.get("sam2_ids", []) if int(value) > 0)
            object_id = int(item["id"])
            description = str(item["description"])
            prompt = (
                f"{description}. Segment the complete physical object instance. "
                "Include all visible parts of this one object. Do not include the tray, table, bin, or adjacent objects."
            )
            masks, scores, boxes = _langsam_predict(model, source_image, prompt)
            best_mask, selected_candidate, candidate_metadata = _select_langsam_mask_for_review_object(
                masks=masks,
                scores=scores,
                review_object=item,
                candidates=candidates,
                background_exclusion_mask=background_exclusion_mask,
                mask_clean_kernel=mask_clean_kernel,
                image_shape=source_image.size[::-1],  # (H, W) from PIL (W, H)
            )
            if best_mask is None:
                continue
            best_mask = _clean_mask(best_mask, mask_clean_kernel)
            if int(np.count_nonzero(best_mask)) == 0:
                continue
            cx, cy = _mask_centroid_xy(best_mask)
            filename = f"{object_id:03d}_langsam_{_safe_label(description)}.png"
            mask_path = output_mask_dir / filename
            _save_mask_png(best_mask, mask_path)
            mask_records.append(
                {
                    "node_id": len(mask_records),
                    "object_id": object_id,
                    "label": description,
                    "description": description,
                    "point": {"x": int(cx), "y": int(cy)},
                    "segmentation_backend": "sam2_auto_review_langsam",
                    "sam2_ids": item.get("sam2_ids", []),
                    "review_status": item.get("status"),
                    "semantic_prompt": prompt,
                    "selected_langsam_candidate": selected_candidate,
                    "langsam_candidates": candidate_metadata,
                    "mask_path": str(mask_path.resolve()),
                    "mask_area": int(np.count_nonzero(best_mask)),
                    "mask_array": best_mask,
                }
            )

        mask_records = _append_unclaimed_sam2_candidates(
            mask_records=mask_records,
            candidates=candidates,
            claimed_sam2_ids=claimed_sam2_ids,
            image_np=image_np,
            output_mask_dir=output_mask_dir,
            max_unclaimed=preserve_unclaimed_sam2,
            depth_map=depth_map,
        )

    report = [
        {
            "stage": "sam2_auto_initial",
            "label_png": str(label_path.resolve()),
            "rgb_parts_sheet_png": str(parts_sheet_path.resolve()),
            "candidates": sam2_report[:200],
        },
        {"stage": "openai_sam2_review", "objects": review_objects, "raw_model_output": raw_review, "error": None},
        {"stage": "sam2_unclaimed_preservation", "max_unclaimed": int(preserve_unclaimed_sam2), "claimed_sam2_ids": sorted(claimed_sam2_ids)},
    ]
    _log_step("  ②c langsam_refine + unclaimed", t_p1)
    return mask_records, report


