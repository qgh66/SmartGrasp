"""VLM interface — single-shot object listing + SAM2 mask assignment via OpenAI API."""

from __future__ import annotations

import base64
import json
import os
import time
from html import unescape
from pathlib import Path
from typing import Any

from SmartGrasp.perception._shared import _log_step

_ALLOWED_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}


# ── utilities ────────────────────────────────────────────────────────────────

def _configured_reasoning_effort() -> str | None:
    value = os.environ.get("SMARTGRASP_REASONING_EFFORT", "").strip().lower()
    if not value:
        return None
    if value not in _ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            "SMARTGRASP_REASONING_EFFORT must be one of "
            f"{sorted(_ALLOWED_REASONING_EFFORTS)}, got {value!r}"
        )
    return value

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
    """Encode the original image bytes without resizing or lossy recompression."""
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_types.get(image_path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported VLM image format: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
        "Grouping rules — apply in this priority order:",
        "- Build the physical-object inventory from the original RGB image before assigning SAM2 ids. SAM2 masks are region proposals, not object identities.",
        "- Default to separate objects. Independent-object evidence has priority over all merge evidence and vetoes a merge.",
        "- Mandatory independence test: if two regions each have a coherent, recognizable physical-object identity or their own independently operable structure (for example, each has its own grip, articulation, or working mechanism), they are two objects, even when touching, crossing, stacked, mutually occluding, or one appears to hold the other.",
        "- Mandatory whole-instance rule: complementary, non-independent regions that together form one canonical manufactured instance MUST be grouped as one physical object and all of their SAM2 ids MUST share one object_id. Apply this strictly even when SAM2 splits the instance into several masks or occlusion makes its regions disconnected. Non-exhaustive illustrations include a hammer handle with its head, a screwdriver handle with its shaft/tip, and a drill housing with its grip/chuck. These examples are explanatory only and never override visual evidence or the independence test.",
        "- A category/name match or resemblance to an illustration is never by itself merge evidence and is never a reason to omit an object. Every visible physical object must still be inventoried from the image.",
        "- Never return a constituent region, such as an attached handle, housing section, shaft, working end, or other integral component, as a separate object merely because it has its own SAM2 id. Keep it separate only when there is clear visual evidence that it is detached, is a standalone item, or belongs to a different independently complete physical object.",
        "- Merge complementary, non-independent regions of one canonical manufactured instance when there is positive ownership evidence such as a real attachment, shared articulation, unique geometric continuation, aligned material/shape transition, or an unambiguous part-to-whole structure. A region that itself forms another independently complete physical object may not be merged.",
        "- Touching, overlap, proximity, functional relation, or a convenient combined silhouette is not attachment evidence. Never invent a hidden joint or treat one object resting on another as a single object.",
        "- Occluded disconnected regions may belong to one object only when their geometry, orientation, scale, and structure align across the same occluder and this is the unique plausible explanation; otherwise keep them separate.",
        "- Without positive ownership evidence, represent a partial or uncertain region as a separate partially visible object with a neutral appearance-based name.",
        "- Keep repeated instances separate and ignore background, shadows, printed texture, highlights, and segmentation noise.",
        "",
        "SAM2 assignment rules:",
        "- Assign every available SAM2 id exactly once across objects[].sam2_ids: no omissions and no duplicates. This ownership requirement never authorizes an object merge.",
        "- Assign an ambiguous or coarse id to only its single best-fitting owner; do not create or merge objects merely to accommodate it.",
        "- One object may use multiple ids only when the grouping rules above are satisfied. Before returning, audit every multi-id object: if its ids contain two recognizable physical-object identities or two independent operating structures, split them into separate objects. Use visible_parts only for true constituent regions of one object, never for an adjacent, held, or stacked object.",
        "- Before returning, perform a mandatory whole-instance audit: if two proposed objects are complementary, geometrically aligned constituent regions that together complete one canonical manufactured instance, merge their records and SAM2 ids unless both regions independently form complete physical objects.",
        "- Every visible_parts[].sam2_ids value must also appear in that object's top-level sam2_ids.",
        "- Name each object as a complete physical instance; keep names and visible-part descriptions concise.",
        "",
        "Output only valid JSON with this schema:",
        '{"objects":[',
        '  {"id":1, "description":"blue and gray manufactured object",',
        '   "relative_position":"lower right",',
        '   "sam2_ids":[3,7,12],',
        '   "visible_parts":[',
        '     {"description":"upper blue region","sam2_ids":[3]},',
        '     {"description":"central gray region","sam2_ids":[7]},',
        '     {"description":"lower blue region","sam2_ids":[12]}',
        '   ]}',
        ']}',
    ])

    client = _openai_client(api_key_env, base_url, timeout)
    reasoning_effort = _configured_reasoning_effort()
    request_options: dict[str, Any] = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_data_url(image_path), "detail": "high"}},
                {"type": "image_url", "image_url": {"url": _image_data_url(label_image_path), "detail": "high"}},
                {"type": "image_url", "image_url": {"url": _image_data_url(parts_sheet_path), "detail": "high"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort is not None:
        request_options["reasoning_effort"] = reasoning_effort
    llm_started_at = time.monotonic()
    response = client.chat.completions.create(**request_options)
    llm_call_seconds = time.monotonic() - llm_started_at

    _log_step("  ②b vlm_review (3img)", t0)

    payload = _extract_json_from_text(response.choices[0].message.content or "")
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
        object_sam2_id_set = set(sam2_ids)

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
            part_ids = sorted(
                set(
                    int(p)
                    for p in pids
                    if (
                        str(p).isdigit()
                        and 1 <= int(p) <= max_id
                        and int(p) in object_sam2_id_set
                    )
                )
            )
            if not part_ids:
                continue
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
        "review_backend": "openai_chat_completions",
        "llm_timing": {
            "call_count": 1,
            "call_seconds": llm_call_seconds,
            "calls_seconds": [llm_call_seconds],
            "reasoning_effort": reasoning_effort or "model_default",
        },
        "image": {
            "path": str(image_path.resolve()),
            "sam2_label_path": str(label_image_path.resolve()),
            "sam2_rgb_parts_sheet_path": str(parts_sheet_path.resolve()),
        },
        "objects": normalized,
    }
    (out_dir / "vlm.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized
