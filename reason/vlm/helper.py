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
) -> str:
    """Build the user prompt for a partially visible target."""
    lines = [
        f"Target: mid={target_mid}, label={target_label}",
        "",
        "Candidate occluders:",
    ]
    for o in occluders:
        lines.append(f"  - mid={o['mid']}, label={o['label']}")
    lines.append("")
    lines.append("Occlusion relations (a -> b means a covers b):")
    if relations:
        for a, b in relations:
            lines.append(f"  - {a} -> {b}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(
        f'Reply as JSON: {{"scores": {{"<mid>": <0..1>, ...}}}}. '
        f'Mids must be exactly: {[o["mid"] for o in occluders]}.'
    )
    return "\n".join(lines)

def _build_user_text_invisible(
    target_label: str,
    occluders: list[dict[str, Any]],
) -> str:
    """Build the user prompt for a fully hidden target."""
    lines = [
        f"Target (HIDDEN, not in image): {target_label}",
        "",
        "Visible candidate occluders (one of them likely hides the target):",
    ]
    for o in occluders:
        lines.append(f"  - mid={o['mid']}, label={o['label']}")
    lines.append("")
    lines.append(
        f'Reply ONLY with JSON: {{"scores": {{"<mid>": <0..1>, ...}}}}. '
        f'Mids must be exactly: {[o["mid"] for o in occluders]}. '
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
