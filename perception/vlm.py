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
    max_dim = 768
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
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
    kwargs: dict[str, Any] = {"timeout": timeout}
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
    candidate_lines = [
        f"{idx}: bbox={c.get('bbox', [])}, area_ratio={float(c.get('area_ratio', 0)):.5f}"
        for idx, c in enumerate(candidates[:max_labels], start=1)
    ]

    prompt = (
        "You are looking at a top-down scene photo. "
        "Three images are provided: (1) the original RGB photo, "
        "(2) a numbered overlay where each colored region is a SAM2 mask candidate, "
        "(3) a contact sheet of numbered RGB cutouts, one per SAM2 candidate.\n\n"

        "==== STEP 1: LIST OBJECT INSTANCES ====\n"
        "Examine the original scene image. Ignore the tray, table, background surface, and shadows. "
        "List every real physical object instance that is visible or partially visible. "
        "Important: Use the general geometries and usages of the objects to identify their boundaries. "
        "Apart from the original RGB photo, you can use the numbered overlay and the cutout sheet to help understand the scene."
        "Include small objects (e.g. bolts, screws). Separate different object instances, especially when they are similar.\n"

        "An OBJECT INSTANCE is one complete physical entity. It may have multiple parts with different colors, shapes or materials "
        "The parts may be disconnected visually if partially occluded."
        "(e.g. pliers with red/yellow handles and black jaws is ONE object. "
        "A package with labels, flaps, or printed patterns is ONE object. "
        "Printed graphics on a surface are NOT separate objects.)\n"

        "Strictly separate two object instances that touch if they are physically different instances. "
        "Strictly separate two object instances that are same but are different instances. "
        "Never merge object instances especially when they are of the same category. "
        "Use relative position words (e.g. upper left, lower right, top center) to describe where each object is.\n\n"

        "==== STEP 2: ASSIGN SAM2 MASKS ====\n"
        "For each object, look at the numbered overlay and the cutout sheet. "
        "Assign the SAM2 mask ids that belong to that object. "
        "A single object may have multiple SAM2 masks if it is fragmented. "
        "Each SAM2 mask may be assigned to AT MOST ONE object.\n\n"

        "==== STEP 3: MAP VISIBLE PARTS ====\n"
        "For each object, list its visible parts and which SAM2 mask ids "
        "correspond to each part. If an object has only one visible part, "
        "include it as a single entry in visible_parts. "
        "If an object has no usable SAM2 masks, give it empty sam2_ids.\n\n"

        "==== OUTPUT ====\n"
        "Include relative position for every object, especially for multiple instances of the same category.\n"
        "Return only valid JSON with this schema:\n"
        '{"objects":[\n'
        '  {"id":1, "description":"red and yellow handled pliers",\n'
        '   "relative_position":"lower right",\n'
        '   "sam2_ids":[3,7,12],\n'
        '   "visible_parts":[\n'
        '     {"description":"red handle","sam2_ids":[3]},\n'
        '     {"description":"yellow handle","sam2_ids":[7]},\n'
        '     {"description":"black jaws","sam2_ids":[12]}\n'
        '   ]}\n'
        ']}\n\n'

        #"Available SAM2 mask ids:\n" + "\n".join(candidate_lines)
    )

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
