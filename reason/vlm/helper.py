import base64
import io
from PIL import Image
import json
from typing import Any

import numpy as np


def _build_user_text_partial(
    target_mid: int,
    target_label: str,
    occluders: list[dict[str, Any]],
    relations: list[tuple[int, int]],
    prompt_mode: str = "original",
) -> str:
    """Build the user prompt for a partially visible target."""
    lines = [
        f"Target: mid={target_mid}, label={target_label}",
        "",
        "Candidate occluders:",
    ]
    for o in occluders:
        part_text = ""
        if prompt_mode == "graspability":
            part_ids = o.get("part_ids") or []
            top = "top-layer" if o.get("is_top_layer") else "non-top-layer"
            part_text = f", layer={top}, sam2_part_ids={part_ids}"
        lines.append(f"  - mid={o['mid']}, label={o['label']}{part_text}")
    lines.append("")
    lines.append("Occlusion relations (a -> b means a covers b):")
    if relations:
        for a, b in relations:
            lines.append(f"  - {a} -> {b}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(
        "Reminders:\n"
        "  - Score range is [0, 1]. Avoid only 0 or 1; use intermediate values.\n"
        "  - A candidate that blocks a DIRECT occluder of the target is an\n"
        "    INDIRECT occluder and deserves a moderate score (around 0.5-0.7),\n"
        "    not 0.\n"
        "  - A candidate that directly covers the target should be high\n"
        "    (around 0.9-1.0).\n"
    )
    mids = [o["mid"] for o in occluders]
    if prompt_mode == "graspability":
        lines.append("")
        lines.append(
            "Also estimate one integrated graspability score for each candidate "
            "object in [0, 1] for the next removal grasp. Use the labeled scene "
            "image and the SAM2 part contact sheet when available to identify the "
            "best feasible visible part or region, but return only one object-level "
            "graspability score. Assume the robot uses a parallel gripper. A high "
            "score means there is an exposed, reachable, stable, sufficiently thick "
            "part/region with enough clearance, and grasping it can move and remove "
            "the whole object safely. A low score means the visible parts are tiny, "
            "thin, buried, slippery-looking, merged with neighbors, occluded, hard "
            "to approach, likely to collide, or unlikely to move the whole object "
            "stably. Mention the best graspable part/region and any object-level "
            "penalty in the reason."
        )
        lines.append(
            f'Reply as JSON: {{"scores": {{"<mid>": <0..1>, ...}}, '
            f'"graspability": {{"<mid>": <0..1>, ...}}, '
            f'"reason": "<brief reason for the scores and graspability>"}}. '
            f'Mids must be exactly: {mids}.'
        )
    else:
        lines.append(
            f'Reply as JSON: {{"scores": {{"<mid>": <0..1>, ...}}, '
            f'"reason": "<brief reason for the scores>"}}. '
            f'Mids must be exactly: {mids}.'
        )
    return "\n".join(lines)

def _build_user_text_invisible(
    target_label: str,
    occluders: list[dict[str, Any]],
    prompt_mode: str = "original",
) -> str:
    """Build the user prompt for a fully hidden target."""
    lines = [
        f"Target (HIDDEN, not in image): {target_label}",
        "",
        "Visible candidate occluders (one of them likely hides the target):",
    ]
    for o in occluders:
        part_text = ""
        if prompt_mode == "graspability":
            part_ids = o.get("part_ids") or []
            part_text = f", sam2_part_ids={part_ids}"
        lines.append(f"  - mid={o['mid']}, label={o['label']}{part_text}")
    lines.append("")
    mids = [o["mid"] for o in occluders]
    if prompt_mode == "graspability":
        lines.append(
            "Also estimate one integrated graspability score for each visible "
            "top-layer candidate in [0, 1] for the next removal grasp. Use the "
            "labeled scene image and the SAM2 part contact sheet when available to "
            "identify the best feasible visible part or region, but return only one "
            "object-level graspability score. Assume the robot uses a parallel "
            "gripper. A high score means there is an exposed, reachable, stable, "
            "sufficiently thick part/region with enough clearance, and grasping it "
            "can move and remove the whole object safely. A low score means the "
            "visible parts are small, thin, hidden, unstable, slippery-looking, "
            "merged with neighbors, hard to approach, likely to collide, or "
            "unlikely to move the whole object stably. Mention the best graspable "
            "part/region and any object-level penalty in the reason."
        )
        lines.append("")
        lines.append(
            f'Reply ONLY with JSON: {{"scores": {{"<mid>": <0..1>, ...}}, '
            f'"graspability": {{"<mid>": <0..1>, ...}}, '
            f'"reason": "<brief reason for the scores and graspability>"}}. '
            f'Mids must be exactly: {mids}. '
            f'The scores represent mutually-exclusive hypotheses and should roughly sum to 1.0. '
            f'NO markdown, NO code fences, NO prose.'
        )
    else:
        lines.append(
            f'Reply ONLY with JSON: {{"scores": {{"<mid>": <0..1>, ...}}, '
            f'"reason": "<brief reason for the scores>"}}. '
            f'Mids must be exactly: {mids}. '
            f'These represent mutually-exclusive hypotheses; they should roughly sum to 1.0. '
            f'NO markdown, NO code fences, NO prose.'
        )
    return "\n".join(lines)


def _encode_image_b64(image: np.ndarray) -> str:
    """Encode an RGB numpy image as base64 JPEG."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_scores_independent(
    text: str,
    occluder_mids: list[int],
) -> dict[int, float]:
    """Parse independent scores without normalizing across candidates."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        data = json.loads(cleaned)
        raw_scores = data.get("scores", {})
        out: dict[int, float] = {}
        for mid in occluder_mids:
            v = raw_scores.get(str(mid), raw_scores.get(mid, 0.5))
            out[mid] = min(1.0, max(0.0, float(v)))
        return out
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return {mid: 0.5 for mid in occluder_mids}


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_score_payload_independent(
    text: str,
    occluder_mids: list[int],
) -> dict[str, Any]:
    """Parse independent semantic scores plus optional graspability."""
    try:
        data = json.loads(_clean_json_text(text))
        return {
            "scores": _parse_value_map(data.get("scores", {}), occluder_mids, 0.5),
            "graspability": _parse_object_graspability(
                data.get("graspability", {}),
                data.get("graspability_parts", {}),
                occluder_mids,
            ),
            "graspability_part_id": _parse_graspability_part_id(
                data.get("graspability_parts", {}), occluder_mids
            ),
            "graspability_parts": _normalize_graspability_parts(
                data.get("graspability_parts", {}), occluder_mids
            ),
            "reason": _parse_reason(data),
        }
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return {
            "scores": {mid: 0.5 for mid in occluder_mids},
            "graspability": {mid: 1.0 for mid in occluder_mids},
            "graspability_part_id": {mid: None for mid in occluder_mids},
            "graspability_parts": {mid: {} for mid in occluder_mids},
            "reason": "VLM response could not be parsed; used fallback scores.",
        }


def _parse_score_payload_normalized(
    text: str,
    occluder_mids: list[int],
) -> dict[str, Any]:
    """Parse normalized hidden-target probabilities plus optional graspability."""
    try:
        data = json.loads(_clean_json_text(text))
        raw_scores = _parse_value_map(
            data.get("scores", {}), occluder_mids, 1.0 / len(occluder_mids)
        )
        total = sum(raw_scores.values())
        if total <= 0:
            n = len(raw_scores)
            scores = {mid: 1.0 / n for mid in raw_scores} if n > 0 else {}
        else:
            scores = {mid: value / total for mid, value in raw_scores.items()}
        return {
            "scores": scores,
            "graspability": _parse_object_graspability(
                data.get("graspability", {}),
                data.get("graspability_parts", {}),
                occluder_mids,
            ),
            "graspability_part_id": _parse_graspability_part_id(
                data.get("graspability_parts", {}), occluder_mids
            ),
            "graspability_parts": _normalize_graspability_parts(
                data.get("graspability_parts", {}), occluder_mids
            ),
            "reason": _parse_reason(data),
        }
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        n = len(occluder_mids)
        scores = {mid: 1.0 / n for mid in occluder_mids} if n > 0 else {}
        return {
            "scores": scores,
            "graspability": {mid: 1.0 for mid in occluder_mids},
            "graspability_part_id": {mid: None for mid in occluder_mids},
            "graspability_parts": {mid: {} for mid in occluder_mids},
            "reason": "VLM response could not be parsed; used fallback scores.",
        }


def _parse_value_map(
    raw_values: Any,
    mids: list[int],
    default: float,
) -> dict[int, float]:
    out: dict[int, float] = {}
    raw_values = raw_values if isinstance(raw_values, dict) else {}
    for mid in mids:
        value = raw_values.get(str(mid), raw_values.get(mid, default))
        out[mid] = min(1.0, max(0.0, float(value)))
    return out


def _normalize_graspability_parts(
    raw_values: Any,
    mids: list[int],
) -> dict[int, dict[int, float]]:
    raw_values = raw_values if isinstance(raw_values, dict) else {}
    out: dict[int, dict[int, float]] = {}
    for mid in mids:
        part_map = raw_values.get(str(mid), raw_values.get(mid, {}))
        if not isinstance(part_map, dict):
            part_map = {}
        normalized: dict[int, float] = {}
        for part_id, value in part_map.items():
            try:
                pid = int(part_id)
                normalized[pid] = min(1.0, max(0.0, float(value)))
            except (TypeError, ValueError):
                continue
        out[mid] = normalized
    return out


def _parse_graspability_parts(
    raw_values: Any,
    mids: list[int],
) -> dict[int, float]:
    parts = _normalize_graspability_parts(raw_values, mids)
    out: dict[int, float] = {}
    for mid in mids:
        part_scores = parts.get(mid, {})
        out[mid] = max(part_scores.values()) if part_scores else 1.0
    return out


def _parse_object_graspability(
    raw_object_values: Any,
    raw_part_values: Any,
    mids: list[int],
) -> dict[int, float]:
    """Parse object-level graspability, falling back to max part score."""
    fallback = _parse_graspability_parts(raw_part_values, mids)
    raw_object_values = raw_object_values if isinstance(raw_object_values, dict) else {}
    out: dict[int, float] = {}
    for mid in mids:
        if str(mid) in raw_object_values or mid in raw_object_values:
            value = raw_object_values.get(str(mid), raw_object_values.get(mid))
            try:
                out[mid] = min(1.0, max(0.0, float(value)))
            except (TypeError, ValueError):
                out[mid] = fallback[mid]
        else:
            out[mid] = fallback[mid]
    return out


def _parse_graspability_part_id(
    raw_values: Any,
    mids: list[int],
) -> dict[int, int | None]:
    parts = _normalize_graspability_parts(raw_values, mids)
    out: dict[int, int | None] = {}
    for mid in mids:
        part_scores = parts.get(mid, {})
        if not part_scores:
            out[mid] = None
            continue
        best_part = max(part_scores, key=lambda pid: (part_scores[pid], -pid))
        out[mid] = best_part
    return out


def _parse_reason(data: dict[str, Any]) -> str:
    reason = data.get("reason", "")
    if isinstance(reason, str):
        return reason.strip()
    return ""
