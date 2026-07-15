"""VLM interface — single-shot object listing + SAM2 mask assignment via OpenAI API."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from html import unescape
from pathlib import Path
from typing import Any

from PIL import Image

from SmartGrasp.perception._shared import _log_step

VLM_MAX_IMAGE_DIM = 768


# ── utilities ────────────────────────────────────────────────────────────────

def _extract_json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            line for line in stripped.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def _image_data_url(image_path: Path) -> str:
    """Encode image as JPEG base64 data URL, resizing if too large."""
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > VLM_MAX_IMAGE_DIM:
        ratio = VLM_MAX_IMAGE_DIM / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def _openai_client(api_key_env: str, base_url: str | None, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI package not installed.") from exc
    kwargs: dict[str, Any] = {
        "timeout": min(float(timeout), 600.0),
        "max_retries": 0,
    }
    key = os.environ.get(api_key_env)
    if key:
        kwargs["api_key"] = key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


# ── main: single-shot list + assign ──────────────────────────────────────────

def review_and_assign_sam2(
    image_path: Path,
    label_image_path: Path,
    parts_sheet_path: Path,
    candidates: list[dict[str, Any]],
    model_id: str,
    api_key_env: str,
    base_url: str | None,
    timeout: float,
    out_dir: Path,
    max_labels: int = 35,
) -> list[dict[str, Any]]:
    """Single VLM call: list objects + assign SAM2 parts in one prompt.

    Returns list of objects, each with:
      - id, description, sam2_ids
      - visible_parts: list of {description, sam2_ids}
    """
    t0 = time.time()

    prompt = "\n".join([
        "Top-down scene understanding task.",
        "Images: original RGB, numbered SAM2 mask overlay, and numbered RGB cutout sheet.",
        "",
        "Task:",
        "1. First identify every visible physical object instance from the RGB image. Ignore tray/table/background/shadows.",
        "2. Then use the SAM2 overlay and cutout sheet to assign mask ids to each object.",
        "3. Return visible parts for each object.",
        "",
        "Instance rules:",
        "- Same category/color/material/texture is not enough to merge objects.",
        "- Split repeated instances even when identical, touching, overlapping, or partially occluded.",
        "- Do not split true parts of one object; parts may differ in color/material and may be fragmented by occlusion.",
        "- Disconnected visible regions belong to one object only with positive physical evidence: continuous path, shared endpoint, matching geometry across short occlusion, or plausible hidden path behind the same occluder.",
        "- Do not merge distant disconnected regions just because they look similar or are both occluded.",
        "- Small rigid items are often separate objects if visible and separated by background/gaps; merge them into a larger object only when clearly mounted, embedded, or mechanically continuous.",
        "- Ignore printed graphics, texture, shadows, highlights, and segmentation noise.",
        "",
        "SAM2 rules:",
        "- Each SAM2 id can be assigned to at most one object.",
        "- An object may have multiple SAM2 ids if fragmented.",
        "- Do not force a nearby mask into an object if it is better explained as another instance.",
        "- Name each object by the complete physical instance in the RGB image. Use neutral descriptions when uncertain.",
        "",
        "Output only valid JSON with this schema:",
        '{"objects":[',
        '  {"id":1, "description":"red and yellow handled pliers",',
        '   "relative_position":"lower right",',
        '   "sam2_ids":[3,7,12],',
        '   "visible_parts":[',
        '     {"description":"red handle","sam2_ids":[3]},',
        '     {"description":"yellow handle","sam2_ids":[7]},',
        '     {"description":"black jaws","sam2_ids":[12]}',
        '   ]}',
        ']}',
    ])

    client = _openai_client(api_key_env, base_url, timeout)
    response = client.responses.create(
        model=model_id,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": _image_data_url(image_path)},
                {"type": "input_image", "image_url": _image_data_url(label_image_path)},
                {"type": "input_image", "image_url": _image_data_url(parts_sheet_path)},
            ],
        }],
        max_output_tokens=3000,
        #reasoning={"effort": "high"},
        store=False,
    )

    _log_step("  ②b vlm_review (3img)", t0)

    payload = _extract_json_from_text(_response_text(response))
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("VLM response must contain an `objects` list.")

    max_id = min(len(candidates), max_labels)
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_objects, start=1):
        if not isinstance(item, dict):
            continue
        desc = unescape(str(item.get("description") or "").strip())
        if not desc:
            continue

        ids = item.get("sam2_ids", []) or []
        if not isinstance(ids, list):
            ids = [ids]
        sam2_ids = sorted(set(int(i) for i in ids if str(i).isdigit() and 1 <= int(i) <= max_id))

        visible_parts: list[dict[str, Any]] = []
        for part in item.get("visible_parts", []) or []:
            if not isinstance(part, dict):
                continue
            pd = unescape(str(part.get("description") or "").strip())
            if not pd:
                continue
            pids = part.get("sam2_ids", []) or []
            if not isinstance(pids, list):
                pids = [pids]
            part_ids = sorted(set(int(p) for p in pids if str(p).isdigit() and 1 <= int(p) <= max_id))
            visible_parts.append({"description": pd, "sam2_ids": part_ids})

        normalized.append({
            "id": int(item.get("id") or idx),
            "description": desc,
            "relative_position": unescape(str(item.get("relative_position") or "").strip()),
            "sam2_ids": sam2_ids,
            "visible_parts": visible_parts,
        })

    if not normalized:
        raise ValueError("VLM returned no valid objects.")

    payload = {
        "model_id": model_id,
        "review_backend": "openai_responses",
        "image": {
            "path": str(image_path.resolve()),
            "sam2_label_path": str(label_image_path.resolve()),
            "sam2_rgb_parts_sheet_path": str(parts_sheet_path.resolve()),
        },
        "objects": normalized,
    }
    (out_dir / "vlm.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized
