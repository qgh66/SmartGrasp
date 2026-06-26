"""VLM round 1: list scene objects from image via OpenAI API."""

from __future__ import annotations

import base64
import json
import os
import time
from html import unescape
from pathlib import Path
from typing import Any

from PIL import Image

from SmartGrasp.perception._shared import _log_step

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
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")
    # Resize if larger than 768px to keep base64 + prompt under proxy token limit
    max_dim = 768
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"



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
        raise RuntimeError("The OpenAI Python package is not installed in the smartgrasp environment.") from exc

    client_kwargs: dict[str, Any] = {"timeout": timeout}
    api_key = os.environ.get(api_key_env)
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)



def _normalize_scene_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("Scene inventory response must contain an `objects` list.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_objects, start=1):
        if not isinstance(item, dict):
            continue
        description = unescape(str(item.get("description") or item.get("label") or "")).strip()
        if not description:
            continue
        visible_parts = item.get("visible_parts", [])
        if not isinstance(visible_parts, list):
            visible_parts = [visible_parts]
        relative_position = unescape(str(item.get("relative_position") or item.get("position") or "")).strip()
        normalized.append(
            {
                "id": int(item.get("id") or index),
                "description": description,
                "relative_position": relative_position,
                "visible_parts": [
                    unescape(str(part)).strip()
                    for part in visible_parts
                    if unescape(str(part)).strip()
                ],
            }
        )
    if not normalized:
        raise ValueError("Scene inventory response returned no valid objects.")
    return normalized



def _openai_list_scene_objects(
    image_path: Path,
    model_id: str,
    api_key_env: str,
    base_url: str | None,
    timeout: float,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    prompt = (
        "List every visible or partially visible physical object in this scene.\n\n"
        "Important: Use the overall geometry and function of an object rather than color alone to determine object boundaries."
        "Ignore the tray, table, bin, support surface, background and reflections. "
        "Include small objects. Be careful with ring and fan shaped objects."      
        "Strictly separate different instances that are overlapping or similar.\n\n"  
        "However, treat an object as a whole when it has multiple parts with different colors or materials. "
        "For example, pliers with red/yellow handles and black jaws are one object, not separate red, yellow, and black objects. "
        "An object may not be contiguously visible if partially occluded, whose parts may be far away from each other.\n\n"
        "Use explicit relative position words for every object, especially repeated similar objects. "
        "For each object, describe the complete object and list its visible parts.\n\n"
        "Return only JSON with this schema: "
        "{\"objects\":[{\"id\":1,\"description\":\"red and yellow handled pliers with black jaws on the right\","
        "\"relative_position\":\"lower right\",\"visible_parts\":[\"red handle\",\"yellow handle\",\"black jaws\"]}]}."
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
                ],
            }
        ],
        max_output_tokens=1600,
        reasoning={"effort": "medium"},
        store=False,
    )
    raw_output = _response_text(response)
    (out_dir / "openai_scene_objects_raw.txt").write_text(raw_output, encoding="utf-8")
    scene_objects = _normalize_scene_objects(_extract_json_from_text(raw_output))
    scene_payload = {
        "model_id": model_id,
        "review_backend": "openai_responses",
        "image": {"path": str(image_path.resolve())},
        "raw_model_output": raw_output,
        "objects": scene_objects,
    }
    (out_dir / "openai_scene_objects.json").write_text(json.dumps(scene_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return scene_objects, raw_output


