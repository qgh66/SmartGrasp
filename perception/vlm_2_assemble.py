"""VLM round 2: assemble SAM2 candidate fragments into coherent scene objects."""

from __future__ import annotations

import json
import time
from html import unescape
from pathlib import Path
from typing import Any

from SmartGrasp.perception._shared import _log_step
from SmartGrasp.perception.vlm_1_detection import (
    _openai_client, _image_data_url, _response_text,
    _extract_json_from_text, _openai_list_scene_objects,
)

def _openai_review_sam2_candidates(
    image_path: Path,
    label_image_path: Path,
    parts_sheet_path: Path,
    candidates: list[dict[str, Any]],
    model_id: str,
    api_key_env: str,
    base_url: str | None,
    timeout: float,
    out_dir: Path,
    max_labels: int = 30,
    scene_objects: list[dict[str, Any]] | None = None,
    scene_raw_output: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    t_r0 = time.time()
    candidate_lines: list[str] = []
    for index, candidate in enumerate(candidates[:max_labels], start=1):
        bbox = candidate.get("bbox", [])
        area_ratio = float(candidate.get("area_ratio", 0.0))
        candidate_lines.append(f"{index}: bbox={bbox}, area_ratio={area_ratio:.5f}")

    if scene_objects is None:
        scene_objects, scene_raw_output = _openai_list_scene_objects(
            image_path=image_path,
            model_id=model_id,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout=timeout,
            out_dir=out_dir,
        )
        _log_step("    ②b1 api_scene_objects (1img)", t_r0)
    else:
        _log_step(f"    ②b1 api_scene_objects prefetched ({len(scene_objects)} objects)", t_r0)
    scene_lines: list[str] = []
    for obj in scene_objects:
        parts = obj.get("visible_parts", [])
        parts_text = f"; visible_parts={parts}" if parts else ""
        position = obj.get("relative_position") or "unspecified position"
        scene_lines.append(f"{int(obj['id'])}: {obj['description']} at {position}{parts_text}")

    prompt = (
        "You are assigning automatic SAM2 mask parts to a known scene object list. "
        "First use the original scene image to understand complete physical objects, then use the numbered scene overlay and the contact sheet of numbered RGB cutouts to choose the SAM2 parts for each object. "
        "Ignore the green tray/box, table, bin, background, shadows, reflections, and support surfaces. "
        "Important: color alone is not a valid reason to merge parts. "
        "Objects can have multiple colors or materials, such as pliers with red/yellow handles and black jaws; include all parts that belong to that one complete physical object. "
        "Also do not merge two separate objects just because their parts share the same color, material, category, or shape. "
        "For each scene object, output one record with all corresponding SAM2 ids in `sam2_ids`. "
        "Every object from the known scene object list must appear in the output exactly once; do not drop small objects just because they are close to a larger object. "
        "If a listed scene object has no usable SAM2 part, include it with an empty `sam2_ids` list. "
        "Return only JSON with this schema: "
        "{\"objects\":[{\"id\":1,\"scene_object_id\":1,\"description\":\"red and yellow handled pliers with black jaws on the right\","
        "\"sam2_ids\":[3,7,12]}]}. "
        "The `description` must describe the final complete object mask, not just one color part. "
        "Known scene objects:\n"
        + "\n".join(scene_lines)
        + "\nAvailable SAM2 mask ids:\n"
        + "\n".join(candidate_lines)
        + "\nImage order: original scene, numbered SAM2 overlay, numbered RGB cutout sheet."
    )

    client = _openai_client(api_key_env=api_key_env, base_url=base_url, timeout=timeout)
    response = client.responses.create(
        model=model_id,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _image_data_url(image_path)},
                    {"type": "input_image", "image_url": _image_data_url(label_image_path)},
                    {"type": "input_image", "image_url": _image_data_url(parts_sheet_path)},
                ],
            }
        ],
        max_output_tokens=2200,
        store=False,
    )
    raw_output = _response_text(response)
    (out_dir / "openai_sam2_review_raw.txt").write_text(raw_output, encoding="utf-8")
    _log_step("    ②b2 api_sam2_review (3img)", t_r0)

    payload = _extract_json_from_text(raw_output)
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Object SAM2 review response must contain an `objects` list.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(objects, start=1):
        if not isinstance(item, dict):
            continue
        description = unescape(str(item.get("description") or item.get("label") or "")).strip()
        if not description:
            continue
        try:
            scene_object_id = int(item.get("scene_object_id") or item.get("object_id") or item.get("id") or index)
        except Exception:
            scene_object_id = index
        raw_ids = item.get("sam2_ids", [])
        if raw_ids is None:
            raw_ids = []
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids]
        sam2_ids: list[int] = []
        for raw_id in raw_ids:
            try:
                sam2_id = int(raw_id)
            except Exception:
                continue
            if 1 <= sam2_id <= min(len(candidates), max_labels):
                sam2_ids.append(sam2_id)
        normalized.append(
            {
                "id": int(item.get("id") or index),
                "scene_object_id": scene_object_id,
                "description": description,
                "sam2_ids": sorted(set(sam2_ids)),
                "status": str(item.get("status") or "incomplete"),
            }
        )
    if not normalized:
        raise ValueError("Object SAM2 review returned no valid objects.")

    review_payload = {
        "model_id": model_id,
        "review_backend": "openai_responses",
        "image": {
            "path": str(image_path.resolve()),
            "sam2_label_path": str(label_image_path.resolve()),
            "sam2_rgb_parts_sheet_path": str(parts_sheet_path.resolve()),
        },
        "scene_objects_raw_model_output": scene_raw_output,
        "scene_objects": scene_objects,
        "raw_model_output": raw_output,
        "objects": normalized,
    }
    (out_dir / "sam2_review.json").write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "openai_sam2_review.json").write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized, raw_output

