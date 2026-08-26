"""VLM client used by semantic prior modules in ``reason``."""
from __future__ import annotations

import time
import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from . import config as vlm_config
from .helper import (
    _build_user_text_partial,
    _build_user_text_invisible,
    _build_user_text_graspability,
    _encode_image_b64,
    _parse_graspability_payload,
    _parse_score_payload_independent,
    _parse_score_payload_normalized,
    _parse_scores_independent,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


_REASON_LLM_CALL_TIMINGS: list[dict[str, Any]] = []
_ALLOWED_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}


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


def reset_llm_call_timings() -> None:
    """Reset process-local Reason VLM request timings for one target."""
    _REASON_LLM_CALL_TIMINGS.clear()


def get_llm_call_timings() -> list[dict[str, Any]]:
    """Return a copy of all Reason VLM requests since the last reset."""
    return [dict(item) for item in _REASON_LLM_CALL_TIMINGS]


def _allowed_part_ids_by_mid(
    objects: list[dict[str, Any]],
) -> dict[int, set[int]]:
    """Return the only part IDs each object may receive scores for."""
    allowed: dict[int, set[int]] = {}
    for obj in objects:
        mid = int(obj["mid"])
        allowed[mid] = {
            int(part_id)
            for part_id in (obj.get("part_ids") or [])
        }
    return allowed


_SYSTEM_PROMPT_PARTIAL = """You are a vision/spatial reasoning expert helping a robot
decide which object is most relevant to "uncover" a partially visible target.

You will see:
- A labeled scene image where each object is outlined and tagged with its id.
- An occlusion graph image. An arrow A -> B means Object A significantly
  covers Object B; the arrow points to the covered object B. Do not reverse
  the direction.
- The target object id and label.
- A list of candidate objects (object ids + labels). These are ALL ancestors
  of the target in the occlusion graph — they may be top-layer (directly
  graspable now) OR lower-layer (pressed by other objects above them).
- The occlusion relations (a -> b means a is on top of / in front of b).

For EACH candidate, output an INDEPENDENT score in [0, 1] reflecting its
IMPORTANCE in the occlusion chain leading to the target. Do NOT filter by
whether the candidate can be grasped right now; we account for that separately.

Use the FINE-GRAINED scale:
  - 0.90 - 1.00 = directly covers/blocks the target (immediate occluder).
  - 0.70 - 0.89 = strongly contributes to occlusion (e.g., presses a direct
                  occluder onto the target, or partially overlaps target).
  - 0.50 - 0.69 = indirect occluder (in the chain to target), OR a lower-layer
                  object that becomes the key bottleneck once layers above
                  it are removed.
  - 0.30 - 0.49 = weakly relevant (adjacent or contributes minimally).
  - 0.10 - 0.29 = mostly unrelated to the target.
  - 0.00 - 0.09 = essentially no relevance to the target.

IMPORTANT RULES:
1. Use the FULL [0, 1] range. Do NOT only output 0 or 1.
2. CHAIN OCCLUSION COUNTS: if A blocks B and B blocks the target, A is still
   important — give A a moderate-to-good score (0.5 - 0.8), NOT 0.
3. NON-TOP-LAYER CANDIDATES are fine. Score them based on their role in the
   occlusion chain, not whether they can be grasped first.
4. Scores are INDEPENDENT; they do NOT need to sum to 1. Judge each candidate
   on its own.
5. Consider the labeled scene image, the occlusion graph image, and the
   textual relations together. When judging underneath/covering relations,
   use the graph direction: the source covers the destination.

Output strictly as JSON, no prose, no markdown:
{"scores": {"<mid>": <0..1>, ...}, "reason": "<brief reason for the scores>"}
Include exactly the requested mids."""


_SYSTEM_PROMPT_PARTIAL_GRASPABILITY = _SYSTEM_PROMPT_PARTIAL + """

In addition to occlusion-chain importance, estimate:
1. ONE integrated object-level graspability coefficient in [0, 1] for each
   candidate object.
2. A part-level graspability coefficient in [0, 1] for every listed validated
   part id of each candidate object.

Additional visual references may be supplied:
- An object-ID sheet. Each cell isolates one complete assembled object from
  the scene and labels it as Object <mid>. Use it to understand the whole
  object's identity, shape, extent, and which visible regions belong together.
- A validated part-ID sheet. Each cell isolates one visible part and labels it
  with its part id. Use the candidate list's part_ids to determine which parts
  belong to each Object <mid>.

Use the labeled scene RGB and occlusion graph for spatial layout, overlap,
covering hierarchy, surrounding clearance, and collision risk; use the
object-ID sheet for whole-object appearance and extent; and use the part-ID
sheet for exact grasp contacts. Object ids and validated part ids are
different namespaces and must not be confused.

This is NOT a second semantic relevance score. It should measure whether a
parallel gripper can safely and stably remove the candidate now.

The object-level graspability score is used for choosing which object to
remove. It must jointly consider:
- The best feasible visible part or region to grasp.
- Exposed and reachable surface area of that part/region.
- Stable antipodal/contact geometry for a parallel gripper.
- Enough thickness/width for the gripper to close without slipping.
- Clearance from neighboring objects and low collision risk.
- Whether grasping that part/region can actually move the whole object.
- Whether lifting/removing the whole object would be stable.
- Prefer handles, rims, flat broad surfaces, stems, or rigid protrusions when
  they are exposed and reachable.
- Penalize tiny, thin, buried, merged, slippery-looking, occluded, or unstable
  parts, and also penalize whole-object collision or removal instability.

The part-level graspability score measures how suitable that specific
validated
part is as a visible grasp contact/region. Score every provided part id for
each candidate. If a candidate has no listed part ids, return an empty
object for that candidate in graspability_parts.

Mention the best graspable part/region and any whole-object penalty in the
reason.

Output strictly as JSON, no prose, no markdown:
{"scores": {"<mid>": <0..1>, ...}, "graspability": {"<mid>": <0..1>, ...}, "graspability_parts": {"<mid>": {"<part_id>": <0..1>, ...}, ...}, "reason": "<brief reason for the scores and graspability>"}
Include exactly the requested mids in scores, graspability, and graspability_parts."""


_SYSTEM_PROMPT_INVISIBLE = """You are a vision/spatial reasoning expert helping a robot
find a HIDDEN target object that is fully invisible in the current scene.

You will see:
- A labeled scene image where each visible object is outlined and tagged with its id.
- An occlusion graph image. An arrow A -> B means visible Object A
  significantly covers Object B; the arrow points to the covered object B.
  Use it to understand the visible covering hierarchy and never reverse it.
- The target object label (the target itself is NOT in the image).
- A list of candidate occluders (each could be hiding the target underneath/behind/inside).

For EACH candidate, output a probability in [0, 1] using a FINE-GRAINED scale:

  - 0.50 - 0.80 = strongly likely (size/shape/category match, e.g., bowl over spoon).
  - 0.20 - 0.49 = plausible (could hide it but not the most natural match).
  - 0.05 - 0.19 = unlikely (size/category mismatch but not impossible).
  - 0.00 - 0.04 = very unlikely (clearly cannot hide the target).

Use common sense about object semantics:
  - A bowl, cup, or box can hide small objects inside or beneath.
  - A book or flat plate can hide thin objects underneath.
  - A large object can hide more than a small one.
  - Match the size: a target of size X is unlikely to be hidden by a much smaller object.

IMPORTANT RULES:
1. Use the FULL probability range; avoid extreme 0 or 1.
2. The most likely candidate should usually be 0.4 - 0.7, NOT 0.95+,
   because we cannot SEE the target — there is real uncertainty.
3. These probabilities represent MUTUALLY EXCLUSIVE hypotheses (the target
   is hidden under exactly ONE candidate), so they should approximately
   sum to 1.0 (downstream code will normalize if needed).

Output strictly as JSON, no prose, no markdown, no code fences:
{"scores": {"<mid>": <0..1>, ...}, "reason": "<brief reason for the scores>"}
Include exactly the requested mids."""


_SYSTEM_PROMPT_INVISIBLE_GRASPABILITY = _SYSTEM_PROMPT_INVISIBLE + """

In addition to hidden-target probability, estimate:
1. ONE integrated object-level graspability coefficient in [0, 1] for each
   visible top-layer candidate.
2. A part-level graspability coefficient in [0, 1] for every listed validated
   part id of each candidate object.

Additional visual references may be supplied:
- An object-ID sheet. Each cell isolates one complete assembled object from
  the scene and labels it as Object <mid>. Use it to understand the whole
  object's identity, shape, extent, and which visible regions belong together.
- A validated part-ID sheet. Each cell isolates one visible part and labels it
  with its part id. Use the candidate list's part_ids to determine which parts
  belong to each Object <mid>.

Use the labeled scene RGB and occlusion graph for spatial layout, overlap,
covering hierarchy, surrounding clearance, and collision risk; use the
object-ID sheet for whole-object appearance and extent; and use the part-ID
sheet for exact grasp contacts. Object ids and validated part ids are
different namespaces and must not be confused.

This is NOT another hidden-target probability. It should measure whether a
parallel gripper can safely and stably remove the candidate now.

The object-level graspability score is used for choosing which object to
remove. It must jointly consider:
- The best feasible visible part or region to grasp.
- Exposed and reachable surface area of that part/region.
- Stable antipodal/contact geometry for a parallel gripper.
- Enough thickness/width for the gripper to close without slipping.
- Clearance from neighboring objects and low collision risk.
- Whether grasping that part/region can actually move the whole object.
- Whether lifting/removing the whole object would be stable.
- Prefer handles, rims, flat broad surfaces, stems, or rigid protrusions when
  they are exposed and reachable.
- Penalize tiny, thin, buried, merged, slippery-looking, occluded, or unstable
  parts, and also penalize whole-object collision or removal instability.

The part-level graspability score measures how suitable that specific
validated
part is as a visible grasp contact/region. Score every provided part id for
each candidate. If a candidate has no listed part ids, return an empty
object for that candidate in graspability_parts.

Mention the best graspable part/region and any whole-object penalty in the
reason.

Output strictly as JSON, no prose, no markdown, no code fences:
{"scores": {"<mid>": <0..1>, ...}, "graspability": {"<mid>": <0..1>, ...}, "graspability_parts": {"<mid>": {"<part_id>": <0..1>, ...}, ...}, "reason": "<brief reason for the scores and graspability>"}
Include exactly the requested mids in scores, graspability, and graspability_parts."""


_SYSTEM_PROMPT_GRASPABILITY = """You are a robot graspability evaluator.

You will see:
- A labeled scene image where objects are outlined and tagged with ids.
- An occlusion graph image. An arrow A -> B means Object A significantly
  covers Object B; the arrow points to the covered object B.
- Optionally, an object-ID sheet where each cell isolates one complete
  assembled object and labels it as Object <mid>.
- Optionally, a validated part contact sheet showing candidate part ids.
- A list of current object ids that the robot may grasp now.

Use the labeled scene RGB and occlusion graph for spatial layout, overlap,
covering hierarchy, surrounding clearance, and collision risk. Use the
object-ID sheet for whole-object identity, shape,
extent, and grouping. Use the validated part-ID sheet for exact visible grasp
contacts, following the supplied mapping from Object <mid> to part ids.
Object ids and part ids are different namespaces and must not be confused.

For EACH listed object, estimate:
1. ONE integrated object-level graspability coefficient in [0, 1].
2. A part-level graspability coefficient in [0, 1] for every listed validated
   part id of that object.

The object-level score should measure whether a parallel gripper can grasp the
best feasible visible part/region and remove the whole object safely. Consider
exposed area, stable antipodal/contact geometry, thickness, clearance from
neighbors, collision risk, and whether the grasped part can move the whole
object. Part-level scores should judge that exact validated part as a visible grasp
contact/region.

Output strictly as JSON, no prose, no markdown, no code fences:
{"graspability": {"<mid>": <0..1>, ...}, "graspability_parts": {"<mid>": {"<part_id>": <0..1>, ...}, ...}, "reason": "<brief reason>"}
Include exactly the requested mids and every listed part id."""



def _parse_scores_normalized(
    text: str,
    occluder_mids: list[int],
) -> dict[int, float]:
    """Parse JSON scores and normalize them to a probability distribution."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        raw_scores = data.get("scores", {})
        out: dict[int, float] = {}
        for mid in occluder_mids:
            v = raw_scores.get(str(mid), raw_scores.get(mid, 1.0 / len(occluder_mids)))
            out[mid] = max(0.0, float(v))

        total = sum(out.values())
        if total <= 0:
            n = len(out)
            return {mid: 1.0 / n for mid in out} if n > 0 else {}
        return {mid: v / total for mid, v in out.items()}
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
        print(f"[VLM] parse failed: {e}; raw text was: {text[:300]}")
        n = len(occluder_mids)
        return {mid: 1.0 / n for mid in occluder_mids} if n > 0 else {}



class VLMClient(ABC):
    @abstractmethod
    def score_occluders_partial(
        self,
        target_mid: int,
        target_label: str,
        occluders: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
        occlusion_relations: list[tuple[int, int]],
        parts_sheet_rgb: np.ndarray | None = None,
        prompt_mode: str = "original",
        object_sheet_rgb: np.ndarray | None = None,
        occlusion_graph_rgb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return semantic scores and optional graspability per occluder."""
        ...

    @abstractmethod
    def score_occluders_invisible(
        self,
        target_label: str,
        occluders: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
        parts_sheet_rgb: np.ndarray | None = None,
        prompt_mode: str = "original",
        object_sheet_rgb: np.ndarray | None = None,
        occlusion_graph_rgb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return hidden-target probabilities and optional graspability."""
        ...

    @abstractmethod
    def score_graspability_objects(
        self,
        objects: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
        parts_sheet_rgb: np.ndarray | None = None,
        object_sheet_rgb: np.ndarray | None = None,
        occlusion_graph_rgb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return object-level and part-level graspability for objects."""
        ...


class OpenAIVisionClient(VLMClient):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from e

        # Priority: explicit args > reason.vlm.config defaults.
        self.model = model or vlm_config.VLM_MODEL
        self.temperature = (
            temperature if temperature is not None
            else float(vlm_config.VLM_TEMPERATURE)
        )
        self.reasoning_effort = _configured_reasoning_effort()

        client_kwargs: dict = {
            "api_key": api_key or os.environ.get(vlm_config.VLM_API_KEY_ENV),
            "timeout": float(vlm_config.VLM_TIMEOUT),
            # A query gets one ten-minute attempt. SDK retries are disabled so
            # this timeout is not multiplied by hidden retry attempts.
            "max_retries": int(vlm_config.VLM_MAX_RETRIES),
        }
        base = base_url or vlm_config.VLM_BASE_URL
        if base:
            client_kwargs["base_url"] = base
        self.client = OpenAI(**client_kwargs)

    def _chat_completion(self, call_type: str, **kwargs: Any) -> Any:
        """Call the LLM and record wall time for this exact API request."""
        started_at = time.monotonic()
        succeeded = False
        if self.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        try:
            response = self.client.chat.completions.create(**kwargs)
            succeeded = True
            return response
        finally:
            _REASON_LLM_CALL_TIMINGS.append(
                {
                    "call_type": str(call_type),
                    "seconds": time.monotonic() - started_at,
                    "success": succeeded,
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort or "model_default",
                }
            )


    def score_occluders_partial(
        self,
        target_mid: int,
        target_label: str,
        occluders: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
        occlusion_relations: list[tuple[int, int]],
        parts_sheet_rgb: np.ndarray | None = None,
        prompt_mode: str = "original",
        object_sheet_rgb: np.ndarray | None = None,
        occlusion_graph_rgb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        mids = [o["mid"] for o in occluders]
        allowed_part_ids = _allowed_part_ids_by_mid(occluders)
        print(
            f"[VLM] calling {self.model}, target={target_mid}, "
            f"occluders={mids}, prompt_mode={prompt_mode}"
        )

        user_text = _build_user_text_partial(
            target_mid,
            target_label,
            occluders,
            occlusion_relations,
            prompt_mode=prompt_mode,
        )

        b64 = _encode_image_b64(labeled_rgb)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "text",
                "text": (
                    "Labeled scene RGB: use the full spatial layout, object "
                    "outlines, and numeric object IDs. These labels are object "
                    "mids, not validated part IDs."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
        if occlusion_graph_rgb is not None:
            graph_b64 = _encode_image_b64(occlusion_graph_rgb)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "Occlusion graph: an arrow A -> B means Object A "
                            "significantly covers Object B; the arrow points "
                            "to the covered object B."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{graph_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )
        if prompt_mode == "graspability" and object_sheet_rgb is not None:
            object_b64 = _encode_image_b64(object_sheet_rgb)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "Object-ID sheet: complete assembled objects labeled Object <mid>.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{object_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )
        if prompt_mode == "graspability" and parts_sheet_rgb is not None:
            parts_b64 = _encode_image_b64(parts_sheet_rgb)
            content.extend(
                [
                    {"type": "text", "text": "Validated part-ID sheet: object-owned visible parts."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{parts_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )
        max_retries = 5
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                resp = self._chat_completion(
                    "score_occluders_partial",
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                _SYSTEM_PROMPT_PARTIAL_GRASPABILITY
                                if prompt_mode == "graspability"
                                else _SYSTEM_PROMPT_PARTIAL
                            ),
                        },
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or ""
                if prompt_mode == "graspability":
                    payload = _parse_score_payload_independent(
                        text,
                        mids,
                        allowed_part_ids,
                    )
                else:
                    payload = {
                        "scores": _parse_scores_independent(text, mids),
                        "graspability": {mid: 1.0 for mid in mids},
                        "graspability_part_id": {mid: None for mid in mids},
                        "graspability_parts": {mid: {} for mid in mids},
                        "reason": _parse_score_payload_independent(text, mids).get("reason", ""),
                    }
                print(
                    f"[VLM] got scores: {payload['scores']}; "
                    f"graspability: {payload['graspability']}; "
                    f"reason: {payload.get('reason', '')}"
                )
                return payload
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str 
                    or "rate" in err_str.lower() 
                    or "饱和" in err_str
                    or "RateLimitError" in type(e).__name__
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait = backoff * (2 ** attempt)  # 2, 4, 8, 16, ...
                    print(f"[VLM] rate limited, retry {attempt+1}/{max_retries} after {wait:.1f}s")
                    time.sleep(wait)
                    continue
                import traceback
                print(f"[VLM] failed with {type(e).__name__}: {e}")
                traceback.print_exc()
                print(f"[VLM] fallback to scores=0.5, graspability=1.0")
                return {
                    "scores": {mid: 0.5 for mid in mids},
                    "graspability": {mid: 1.0 for mid in mids},
                    "graspability_part_id": {mid: None for mid in mids},
                    "graspability_parts": {mid: {} for mid in mids},
                    "reason": f"VLM request failed with {type(e).__name__}; used fallback scores.",
                }


    def score_occluders_invisible(
        self,
        target_label: str,
        occluders: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
        parts_sheet_rgb: np.ndarray | None = None,
        prompt_mode: str = "original",
        object_sheet_rgb: np.ndarray | None = None,
        occlusion_graph_rgb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        mids = [o["mid"] for o in occluders]
        allowed_part_ids = _allowed_part_ids_by_mid(occluders)
        print(f"[VLM-INV] calling {self.model}, "
              f"target_label={target_label!r}, occluders={mids}, "
              f"prompt_mode={prompt_mode}")

        user_text = _build_user_text_invisible(
            target_label, occluders, prompt_mode=prompt_mode
        )

        b64 = _encode_image_b64(labeled_rgb)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "text",
                "text": (
                    "Labeled scene RGB: use the full spatial layout, object "
                    "outlines, and numeric object IDs. These labels are object "
                    "mids, not validated part IDs."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
        if occlusion_graph_rgb is not None:
            graph_b64 = _encode_image_b64(occlusion_graph_rgb)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "Occlusion graph: an arrow A -> B means Object A "
                            "significantly covers Object B; the arrow points "
                            "to the covered object B."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{graph_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )
        if prompt_mode == "graspability" and object_sheet_rgb is not None:
            object_b64 = _encode_image_b64(object_sheet_rgb)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "Object-ID sheet: complete assembled objects labeled Object <mid>.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{object_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )
        if prompt_mode == "graspability" and parts_sheet_rgb is not None:
            parts_b64 = _encode_image_b64(parts_sheet_rgb)
            content.extend(
                [
                    {"type": "text", "text": "Validated part-ID sheet: object-owned visible parts."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{parts_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )

        try:
            resp = self._chat_completion(
                "score_occluders_invisible",
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            _SYSTEM_PROMPT_INVISIBLE_GRASPABILITY
                            if prompt_mode == "graspability"
                            else _SYSTEM_PROMPT_INVISIBLE
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            if prompt_mode == "graspability":
                payload = _parse_score_payload_normalized(
                    text,
                    mids,
                    allowed_part_ids,
                )
            else:
                scores = _parse_scores_normalized(text, mids)
                payload = {
                    "scores": scores,
                    "graspability": {mid: 1.0 for mid in mids},
                    "graspability_part_id": {mid: None for mid in mids},
                    "graspability_parts": {mid: {} for mid in mids},
                    "reason": _parse_score_payload_normalized(text, mids).get("reason", ""),
                }
            print(
                f"[VLM-INV] got scores: {payload['scores']}; "
                f"graspability: {payload['graspability']}; "
                f"reason: {payload.get('reason', '')}"
            )
            return payload
        except Exception as e:
            import traceback
            print(f"[VLM-INV] failed with {type(e).__name__}: {e}")
            print(f"[VLM-INV] full traceback:")
            traceback.print_exc()
            n = len(mids)
            uniform = {mid: 1.0 / n for mid in mids} if n > 0 else {}
            print(f"[VLM-INV] fallback to uniform: {uniform}, graspability=1.0")
            return {
                "scores": uniform,
                "graspability": {mid: 1.0 for mid in mids},
                "graspability_part_id": {mid: None for mid in mids},
                "graspability_parts": {mid: {} for mid in mids},
                "reason": f"VLM request failed with {type(e).__name__}; used fallback scores.",
            }


    def score_graspability_objects(
        self,
        objects: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
        parts_sheet_rgb: np.ndarray | None = None,
        object_sheet_rgb: np.ndarray | None = None,
        occlusion_graph_rgb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        mids = [obj["mid"] for obj in objects]
        allowed_part_ids = _allowed_part_ids_by_mid(objects)
        print(f"[VLM-GRASP] calling {self.model}, objects={mids}")

        user_text = _build_user_text_graspability(objects)
        b64 = _encode_image_b64(labeled_rgb)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "text", "text": "Labeled scene RGB: spatial layout and object labels."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
        if occlusion_graph_rgb is not None:
            graph_b64 = _encode_image_b64(occlusion_graph_rgb)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "Occlusion graph: an arrow A -> B means Object A "
                            "significantly covers Object B; the arrow points "
                            "to the covered object B."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{graph_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )
        if object_sheet_rgb is not None:
            object_b64 = _encode_image_b64(object_sheet_rgb)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "Object-ID sheet: complete assembled objects labeled Object <mid>.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{object_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )
        if parts_sheet_rgb is not None:
            parts_b64 = _encode_image_b64(parts_sheet_rgb)
            content.extend(
                [
                    {"type": "text", "text": "Validated part-ID sheet: object-owned visible parts."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{parts_b64}",
                            "detail": "high",
                        },
                    },
                ]
            )

        try:
            resp = self._chat_completion(
                "score_graspability_objects",
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT_GRASPABILITY},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            payload = _parse_graspability_payload(
                text,
                mids,
                allowed_part_ids,
            )
            print(
                f"[VLM-GRASP] got graspability: {payload['graspability']}; "
                f"reason: {payload.get('reason', '')}"
            )
            return payload
        except Exception as e:
            import traceback
            print(f"[VLM-GRASP] failed with {type(e).__name__}: {e}")
            traceback.print_exc()
            print("[VLM-GRASP] fallback to graspability=1.0")
            return {
                "graspability": {mid: 1.0 for mid in mids},
                "graspability_part_id": {mid: None for mid in mids},
                "graspability_parts": {mid: {} for mid in mids},
                "reason": f"VLM request failed with {type(e).__name__}; used fallback graspability.",
            }


def get_default_client() -> VLMClient:
    """Return the default VLM backend used by the reasoning code."""
    return OpenAIVisionClient()
