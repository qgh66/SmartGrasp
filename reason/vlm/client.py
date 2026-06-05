"""VLM client used by semantic prior modules in ``reason``."""
from __future__ import annotations

import time
import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from .helper import _build_user_text_partial, _build_user_text_invisible, _encode_image_b64, _parse_scores_independent


_SYSTEM_PROMPT_PARTIAL = """You are a vision/spatial reasoning expert helping a robot
decide which object is most relevant to "uncover" a partially visible target.

You will see:
- A labeled scene image where each object is outlined and tagged with its id.
- The target object id and label.
- A list of candidate occluders (object ids + labels).
- The occlusion relations (a -> b means a is on top of / in front of b).

For EACH candidate, output an INDEPENDENT score in [0, 1] using the
following FINE-GRAINED scale:

  - 0.90 - 1.00 = directly covers/blocks the target (clear top-layer occluder).
  - 0.70 - 0.89 = strongly contributes to occlusion (e.g., presses a directly-
                  covering object onto the target, or partially overlaps target).
  - 0.50 - 0.69 = indirect occluder (blocks a direct occluder; removing it is a
                  necessary intermediate step toward reaching the target).
  - 0.30 - 0.49 = weakly relevant (adjacent or contributes minimally).
  - 0.10 - 0.29 = mostly unrelated to the target.
  - 0.00 - 0.09 = removing this candidate has essentially no effect on the target.

IMPORTANT RULES:
1. Use the FULL [0, 1] range. Do NOT only output 0 or 1.
2. CHAIN OCCLUSION COUNTS: if A blocks B and B blocks the target, removing A
   is still useful — give A a moderate-to-good score (0.5 - 0.8), NOT 0.
3. Scores are INDEPENDENT; they do NOT need to sum to 1. Judge each candidate
   on its own.
4. Consider BOTH the image (spatial layout) and the relations (graph chain).

Output strictly as JSON, no prose, no markdown:
{"scores": {"<mid>": <0..1>, ...}}
Include exactly the requested mids."""



_SYSTEM_PROMPT_INVISIBLE = """You are a vision/spatial reasoning expert helping a robot
find a HIDDEN target object that is fully invisible in the current scene.

You will see:
- A labeled scene image where each visible object is outlined and tagged with its id.
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
{"scores": {"<mid>": <0..1>, ...}}
Include exactly the requested mids."""



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
    ) -> dict[int, float]:
        """Return independent [0,1] scores per occluder (partially-visible target)."""
        ...

    @abstractmethod
    def score_occluders_invisible(
        self,
        target_label: str,
        occluders: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
    ) -> dict[int, float]:
        """Return mutually-exclusive probabilities (fully-invisible target)."""
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

        # Priority: explicit args > environment > defaults.
        self.model = model or os.environ.get("VLM_MODEL", "gpt-4o-mini")
        self.temperature = (
            temperature if temperature is not None
            else float(os.environ.get("VLM_TEMPERATURE", "0.0"))
        )

        client_kwargs: dict = {
            "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
        }
        base = base_url or os.environ.get("OPENAI_BASE_URL")
        if base:
            client_kwargs["base_url"] = base
        self.client = OpenAI(**client_kwargs)


    def score_occluders_partial(
        self,
        target_mid: int,
        target_label: str,
        occluders: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
        occlusion_relations: list[tuple[int, int]],
    ) -> dict[int, float]:
        mids = [o["mid"] for o in occluders]
        print(f"[VLM] calling {self.model}, target={target_mid}, occluders={mids}")

        user_text = _build_user_text_partial(
            target_mid, target_label, occluders, occlusion_relations
        )

        b64 = _encode_image_b64(labeled_rgb)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
        max_retries = 5
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT_PARTIAL},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or ""
                scores = _parse_scores_independent(text, mids)
                print(f"[VLM] got scores: {scores}")
                return scores
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
                print(f"[VLM] fallback to 0.5")
                return {mid: 0.5 for mid in mids}


    def score_occluders_invisible(
        self,
        target_label: str,
        occluders: list[dict[str, Any]],
        labeled_rgb: np.ndarray,
    ) -> dict[int, float]:
        mids = [o["mid"] for o in occluders]
        print(f"[VLM-INV] calling {self.model}, "
              f"target_label={target_label!r}, occluders={mids}")

        user_text = _build_user_text_invisible(target_label, occluders)

        b64 = _encode_image_b64(labeled_rgb)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT_INVISIBLE},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            scores = _parse_scores_normalized(text, mids)
            print(f"[VLM-INV] got scores: {scores}")
            return scores
        except Exception as e:
            import traceback
            print(f"[VLM-INV] failed with {type(e).__name__}: {e}")
            print(f"[VLM-INV] full traceback:")
            traceback.print_exc()
            n = len(mids)
            uniform = {mid: 1.0 / n for mid in mids} if n > 0 else {}
            print(f"[VLM-INV] fallback to uniform: {uniform}")
            return uniform


def get_default_client() -> VLMClient:
    """Return the default VLM backend used by the reasoning code."""
    return OpenAIVisionClient()
