"""End-to-end pipeline: Molmo points -> SAM masks -> occlusion graph JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SMARTGRASP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SMARTGRASP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MOLMO_OUT = SMARTGRASP_ROOT / "perception" / "molmo" / "out"

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from SmartGrasp.perception.occul_map.org import build_occlusion_graph, graph_to_jsonable

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only in cv2-less environments
    cv2 = None

_SAM_CACHE: dict[tuple[str, str], tuple[SamProcessor, SamModel]] = {}


@dataclass(frozen=True)
class MolmoPoint:
    molmo_id: int
    x: int
    y: int
    label: str


def _safe_label(label: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in label.strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "object"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _resolve_path(points_json_path: Path, candidate: str) -> Path:
    raw = Path(candidate)
    if raw.is_absolute() and raw.exists():
        return raw

    search_roots = [
        points_json_path.parent,
        points_json_path.parent.parent,
        Path.cwd(),
    ]
    for root in search_roots:
        resolved = (root / raw).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Could not resolve path {candidate!r} relative to {points_json_path}.")


def _load_points(points_json_path: Path) -> tuple[dict[str, Any], list[MolmoPoint], Path]:
    payload = _load_json(points_json_path)
    raw_points = payload.get("points", [])
    if not raw_points:
        raise ValueError(f"No points found in {points_json_path}.")

    points: list[MolmoPoint] = []
    for point in raw_points:
        points.append(
            MolmoPoint(
                molmo_id=int(point["molmo_id"]),
                x=int(point["x"]),
                y=int(point["y"]),
                label=str(point.get("label", f"object_{point['molmo_id']}")),
            )
        )

    image_meta = payload.get("image", {})
    image_path_value = image_meta.get("path")
    if not image_path_value:
        raise ValueError(f"Missing image.path in {points_json_path}.")
    image_path = _resolve_path(points_json_path, image_path_value)
    return payload, points, image_path


def _load_depth_map(depth_path: Path) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(depth_path)
    elif suffix == ".npz":
        npz = np.load(depth_path)
        if len(npz.files) != 1:
            raise ValueError(f"Expected exactly one array inside {depth_path}, found {npz.files}.")
        depth = npz[npz.files[0]]
    else:
        depth = np.array(Image.open(depth_path))

    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D after loading, got shape {depth.shape}.")
    return depth.astype(np.float32, copy=False)


def _save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(out_path)


def _point_grid(point: MolmoPoint, width: int, height: int, radius: int) -> list[list[int]]:
    offsets = [(0, 0)]
    if radius > 0:
        offsets.extend([(radius, 0), (-radius, 0), (0, radius), (0, -radius)])
    coords: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for dx, dy in offsets:
        x = int(np.clip(point.x + dx, 0, width - 1))
        y = int(np.clip(point.y + dy, 0, height - 1))
        if (x, y) not in seen:
            seen.add((x, y))
            coords.append([x, y])
    return coords


def _clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if kernel_size <= 1:
        return mask_bool

    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned > 0

    pad = kernel_size // 2
    padded = np.pad(mask_bool, pad_width=pad, mode="constant", constant_values=False)
    closed = np.zeros_like(mask_bool, dtype=bool)
    for row_offset in range(kernel_size):
        for col_offset in range(kernel_size):
            closed |= padded[row_offset : row_offset + mask_bool.shape[0], col_offset : col_offset + mask_bool.shape[1]]

    padded = np.pad(closed, pad_width=pad, mode="constant", constant_values=True)
    opened = np.ones_like(mask_bool, dtype=bool)
    for row_offset in range(kernel_size):
        for col_offset in range(kernel_size):
            opened &= padded[row_offset : row_offset + mask_bool.shape[0], col_offset : col_offset + mask_bool.shape[1]]
    return opened


def _select_best_mask(processed_masks: Any, iou_scores: torch.Tensor) -> np.ndarray:
    masks_np = np.asarray(processed_masks)
    if masks_np.ndim == 4 and masks_np.shape[0] == 1:
        masks_np = masks_np[0]
    if masks_np.ndim != 3:
        raise ValueError(f"Unexpected mask tensor shape after post-processing: {masks_np.shape}.")

    scores_np = iou_scores.detach().cpu().numpy().reshape(-1)
    best_idx = int(np.argmax(scores_np))
    return masks_np[best_idx] > 0


def _load_sam(sam_model_id: str, device: str) -> tuple[SamProcessor, SamModel]:
    cache_key = (sam_model_id, device)
    if cache_key not in _SAM_CACHE:
        processor = SamProcessor.from_pretrained(sam_model_id)
        model = SamModel.from_pretrained(sam_model_id).to(device)
        model.eval()
        _SAM_CACHE[cache_key] = (processor, model)
    return _SAM_CACHE[cache_key]


def generate_masks_with_sam(
    image_path: Path,
    points: list[MolmoPoint],
    output_mask_dir: Path,
    sam_model_id: str,
    point_grid_radius: int = 0,
    mask_clean_kernel: int = 3,
    device: str | None = None,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = _load_sam(sam_model_id, device)

    mask_records: list[dict[str, Any]] = []
    for point in points:
        input_points = _point_grid(point, width, height, point_grid_radius)
        inputs = processor(
            image,
            input_points=[[input_points]],
            input_labels=[[[1 for _ in input_points]]],
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
        best_mask = _clean_mask(_select_best_mask(processed_masks, outputs.iou_scores[0]), mask_clean_kernel)

        filename = f"mask_{point.molmo_id:03d}_{_safe_label(point.label)}.png"
        mask_path = output_mask_dir / filename
        _save_mask_png(best_mask, mask_path)

        mask_records.append(
            {
                "node_id": len(mask_records),
                "molmo_id": point.molmo_id,
                "label": point.label,
                "point": {"x": point.x, "y": point.y},
                "sam_positive_points": [{"x": int(x), "y": int(y)} for x, y in input_points],
                "mask_path": str(mask_path.resolve()),
                "mask_area": int(np.count_nonzero(best_mask)),
                "predicted_iou": float(torch.max(outputs.iou_scores[0]).item()),
                "mask_array": best_mask,
            }
        )

    return mask_records


def build_org_json(
    points_json_path: Path,
    depth_path: Path,
    output_json_path: Path,
    output_mask_dir: Path,
    sam_model_id: str = "facebook/sam-vit-base",
    epsilon: float = 0.01,
    kernel_size: int = 3,
    min_contact_pixels: int = 1,
    min_contact_ratio: float = 0.0,
    sam_point_grid_radius: int = 0,
    mask_clean_kernel: int = 3,
    device: str | None = None,
) -> dict[str, Any]:
    points_payload, points, image_path = _load_points(points_json_path)
    depth_map = _load_depth_map(depth_path)

    mask_records = generate_masks_with_sam(
        image_path=image_path,
        points=points,
        output_mask_dir=output_mask_dir,
        sam_model_id=sam_model_id,
        point_grid_radius=sam_point_grid_radius,
        mask_clean_kernel=mask_clean_kernel,
        device=device,
    )
    masks = np.stack([record["mask_array"] for record in mask_records], axis=0)

    if masks.shape[1:] != depth_map.shape:
        raise ValueError(
            f"Mask shape {masks.shape[1:]} does not match depth map shape {depth_map.shape}."
        )

    graph, adjacency = build_occlusion_graph(
        masks=masks,
        depth_map=depth_map,
        epsilon=epsilon,
        kernel_size=kernel_size,
        min_contact_pixels=min_contact_pixels,
        min_contact_ratio=min_contact_ratio,
    )

    node_records: list[dict[str, Any]] = []
    for record in mask_records:
        node_record = dict(record)
        node_record.pop("mask_array", None)
        node_records.append(node_record)

    graph_payload = graph_to_jsonable(graph, adjacency, node_records=node_records)
    payload = {
        "image": {
            "path": str(image_path.resolve()),
            "width": int(points_payload.get("image", {}).get("width", masks.shape[2])),
            "height": int(points_payload.get("image", {}).get("height", masks.shape[1])),
        },
        "depth_map": {
            "path": str(depth_path.resolve()),
            "shape": [int(depth_map.shape[0]), int(depth_map.shape[1])],
        },
        "points_source": str(points_json_path.resolve()),
        "molmo_points": points_payload.get("points", []),
        "sam_model_id": sam_model_id,
        "graph": graph_payload,
    }

    for edge in payload["graph"]["edges"]:
        source_node = node_records[edge["source"]]
        target_node = node_records[edge["target"]]
        edge["source_molmo_id"] = int(source_node["molmo_id"])
        edge["target_molmo_id"] = int(target_node["molmo_id"])
        edge["source_label"] = str(source_node["label"])
        edge["target_label"] = str(target_node["label"])

    _write_json(output_json_path, payload)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SAM masks from Molmo points and build an occlusion graph JSON.",
    )
    parser.add_argument(
        "--points-json",
        default=str(DEFAULT_MOLMO_OUT / "molmo_points.json"),
        help="Path to Molmo points JSON.",
    )
    parser.add_argument(
        "--depth-map",
        required=True,
        help="Path to the depth map (.npy, .npz, or image file).",
    )
    parser.add_argument(
        "--mask-dir",
        default=str(DEFAULT_MOLMO_OUT / "mask"),
        help="Directory where per-object mask PNGs will be written.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_MOLMO_OUT / "occlusion_graph.json"),
        help="Path to the final occlusion graph JSON file.",
    )
    parser.add_argument(
        "--sam-model-id",
        default="facebook/sam-vit-base",
        help="Hugging Face SAM model id.",
    )
    parser.add_argument("--epsilon", type=float, default=0.01, help="Depth margin for occlusion decisions.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Dilation kernel size.")
    parser.add_argument(
        "--min-contact-pixels",
        type=int,
        default=1,
        help="Ignore contact areas smaller than this many pixels.",
    )
    parser.add_argument(
        "--min-contact-ratio",
        type=float,
        default=0.0,
        help="Ignore contacts smaller than this fraction of the smaller object mask.",
    )
    parser.add_argument(
        "--sam-point-grid-radius",
        type=int,
        default=0,
        help="Add four positive SAM prompts around each Molmo point with this pixel radius.",
    )
    parser.add_argument(
        "--mask-clean-kernel",
        type=int,
        default=3,
        help="Morphological cleanup kernel for SAM masks; use 1 to disable.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cuda, cuda:0, or cpu.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    payload = build_org_json(
        points_json_path=Path(args.points_json).resolve(),
        depth_path=Path(args.depth_map).resolve(),
        output_json_path=Path(args.output_json).resolve(),
        output_mask_dir=Path(args.mask_dir).resolve(),
        sam_model_id=args.sam_model_id,
        epsilon=args.epsilon,
        kernel_size=args.kernel_size,
        min_contact_pixels=args.min_contact_pixels,
        min_contact_ratio=args.min_contact_ratio,
        sam_point_grid_radius=args.sam_point_grid_radius,
        mask_clean_kernel=args.mask_clean_kernel,
        device=args.device,
    )
    print(f"Saved occlusion graph JSON to: {args.output_json}")
    print(f"Saved {len(payload['graph']['nodes'])} masks to: {args.mask_dir}")


if __name__ == "__main__":
    main()
