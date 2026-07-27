"""Resolve a natural-language instruction with a VLM.

The VLM chooses which object(s) in the scene can solve the user's task. This
module validates the returned object id against ``summary.json`` and returns
predicted object ids only. Visibility branch classification lives in
``reason.branch_judge``.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_TIMEOUT = 600.0
HIDDEN_TARGET_OCCLUDER_MODE = "visible_occluder_for_hidden_target"

_SIDE_WORDS = {
    "left",
    "right",
    "top",
    "bottom",
    "upper",
    "lower",
    "front",
    "back",
    "middle",
    "center",
    "centre",
}

_STOPWORDS = {
    "a",
    "an",
    "the",
    "that",
    "this",
    "one",
    "thing",
    "object",
    "item",
    "brand",
    "text",
    "with",
    "and",
    "on",
    "in",
    "of",
    "for",
    "to",
}


class IntentHandleError(RuntimeError):
    """Raised when intent handling cannot complete."""


class VLMClient(Protocol):
    """Client interface used by tests and the real VLM client."""

    def choose_target(
        self,
        instruction: str,
        scene_context: dict[str, Any],
        image_paths: Iterable[Path],
    ) -> dict[str, Any]:
        """Return a JSON-like VLM decision."""


@dataclass(frozen=True)
class SceneObject:
    """Object from a perception summary."""

    object_id: int
    category: str
    raw_label: str
    x: int | None = None
    y: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.object_id,
            "category": self.category,
            "label": self.raw_label,
            "x": self.x,
            "y": self.y,
        }


@dataclass(frozen=True)
class IntentResult:
    """Resolved target object decision."""

    target_object: SceneObject | None
    candidates: tuple[SceneObject, ...]
    reason: str
    vlm_decision: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "target_object": self.target_object.to_json() if self.target_object else None,
            "candidates": [obj.to_json() for obj in self.candidates],
            "reason": self.reason,
            "vlm_decision": self.vlm_decision,
        }


class ResponsesVLMClient:
    """OpenAI-compatible chat-completions client for intent recognition."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise IntentHandleError(
                f"Missing API key. Set {DEFAULT_API_KEY_ENV} or pass --api-key-env."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise IntentHandleError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> "ResponsesVLMClient":
        api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY", "")
        return cls(
            api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            model=model or DEFAULT_MODEL,
            timeout=timeout,
        )

    def choose_target(
        self,
        instruction: str,
        scene_context: dict[str, Any],
        image_paths: Iterable[Path],
    ) -> dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the intent recognition module for a "
                            "robotic grasping system. Return only one JSON "
                            "object with no markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._content_parts(
                            instruction, scene_context, image_paths
                        ),
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except TimeoutError as exc:
            raise IntentHandleError(
                f"VLM request timed out after {self.timeout}s: {exc}"
            ) from exc
        except Exception as exc:
            raise IntentHandleError(
                f"VLM chat-completions request failed: {exc}"
            ) from exc

        response_text = resp.choices[0].message.content or ""
        return _parse_json_object(response_text)

    def _content_parts(
        self,
        instruction: str,
        scene_context: dict[str, Any],
        image_paths: Iterable[Path],
    ) -> list[dict[str, Any]]:
        prompt = _build_prompt(instruction, scene_context)
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            parts.append(
                {"type": "text", "text": _image_caption(image_path)}
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path), "detail": "high"},
                }
            )
        return parts


def resolve_intent(
    instruction: str,
    summary_path: str | Path,
    occlusion_graph_path: str | Path | None = None,
    *,
    image_paths: Iterable[str | Path] | None = None,
    client: VLMClient | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> IntentResult:
    """Resolve ``instruction`` by calling a VLM and validating its answer."""

    summary_file = Path(summary_path)
    summary = _load_json(summary_file)
    graph_file = Path(occlusion_graph_path) if occlusion_graph_path else _default_graph_path(summary_file, summary)
    graph = _load_json(graph_file) if graph_file and graph_file.exists() else None

    objects = _objects_from_summary(summary)
    occlusion = _occlusion_from_scene(summary, graph)
    scene_context = _scene_context(summary, objects, occlusion)
    resolved_images = _resolve_image_paths(summary_file, summary, image_paths)
    vlm_client = client or ResponsesVLMClient.from_env(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )
    print(
        "[INTENT] calling VLM "
        f"model={getattr(vlm_client, 'model', model)}, "
        f"instruction={instruction!r}, "
        f"summary={summary_file}, "
        f"graph={graph_file}, "
        f"images={[str(path) for path in resolved_images]}"
    )

    decision = vlm_client.choose_target(instruction, scene_context, resolved_images)
    return _result_from_vlm_decision(decision, objects, occlusion)


def _result_from_vlm_decision(
    decision: dict[str, Any],
    objects: list[SceneObject],
    occlusion: dict[int, set[int]],
) -> IntentResult:
    object_by_id = {obj.object_id: obj for obj in objects}
    if not _as_bool(decision.get("target_present", False)):
        selection_mode = str(decision.get("selection_mode") or "").strip().lower()
        if selection_mode == HIDDEN_TARGET_OCCLUDER_MODE:
            candidate_ids = _int_list(decision.get("candidate_object_ids"))
            candidates = tuple(
                object_by_id[obj_id]
                for obj_id in candidate_ids
                if obj_id in object_by_id
            )
            if candidates:
                return IntentResult(
                    target_object=None,
                    candidates=candidates,
                    reason=str(
                        decision.get("reason")
                        or "The target is hidden; selected visible occluder candidates."
                    ),
                    vlm_decision=decision,
                )
        return IntentResult(
            target_object=None,
            candidates=(),
            reason=str(decision.get("reason") or "VLM found no suitable scene object."),
            vlm_decision=decision,
        )

    candidate_ids = _int_list(decision.get("candidate_object_ids"))
    target_id = _optional_int(decision.get("target_object_id"))
    if target_id is not None and target_id not in candidate_ids:
        candidate_ids.append(target_id)

    candidates = [object_by_id[obj_id] for obj_id in candidate_ids if obj_id in object_by_id]
    if not candidates:
        return IntentResult(
            target_object=None,
            candidates=(),
            reason="VLM did not return a valid object id from summary.json.",
            vlm_decision=decision,
        )

    target = object_by_id.get(target_id) if target_id in object_by_id else candidates[0]
    return IntentResult(
        target_object=target,
        candidates=tuple(candidates),
        reason=str(
            decision.get("reason")
            or f"Selected object {target.object_id} from VLM candidates."
        ),
        vlm_decision=decision,
    )


def _build_prompt(instruction: str, scene_context: dict[str, Any]) -> str:
    return (
        "You are a highly capable robotic assistant designed to support grasping "
        "tasks in real-world environments. Your role here is intent recognition: "
        "analyze the user instruction, identify the task-relevant scene object, "
        "and return only the selected object information.\n\n"

        "Attached visual references are introduced by a caption immediately "
        "before each image. Use them as follows:\n"
        "- Labeled scene RGB: understand the complete spatial layout and map "
        "visible labels to object ids.\n"
        "- Occlusion graph: determine covering relations; source -> target means "
        "the source object covers/occludes the target object.\n"
        "- Final-object ID sheet: inspect each complete assembled object isolated "
        "on white and labeled with its object id.\n"
        "The numeric labels in these three references are object ids from "
        "summary.json, not SAM2 part ids. Cross-check the references instead of "
        "inferring an id from only one image.\n\n"

        "Follow these reasoning steps internally:\n"
        "Step 1. Task analysis: understand the user's underlying intention and "
        "any implicit task requirements.\n"
        "Step 2. Relevant object identification: from the listed objects and "
        "attached images, select the object most relevant to the task. The scene "
        "image shows spatial context; the final-object sheet shows each assembled "
        "object isolated on white and labeled with its final object id. Use the "
        "sheet to inspect object appearance and map it to summary.json ids.\n"
        "If the requested target itself is completely invisible, but the instruction "
        "explicitly identifies a visible object that covers or contains it, treat that "
        "visible object as an occluder candidate. In that case set target_present=false, "
        f"selection_mode={HIDDEN_TARGET_OCCLUDER_MODE!r}, target_object_id=null, "
        "and put only the visible covering object's id(s) in candidate_object_ids.\n"
        "IMPORTANT: If the user's description is ambiguous or incomplete, combine "
        "the scene object information and object list, and reason from object shape, "
        "color, function, and spatial position. Except for the completely hidden "
        "target rule above, you must still select the single most likely object as "
        "target_object_id.\n"
        "Step 3. Spatial reasoning: if the instruction uses words such as top, "
        "bottom, upper, lower, front, or back, interpret them from 2D image position. "
        "If the instruction says that the target is underneath, below, or covered by "
        "another object, use both the attached occlusion graph and the occluded_by "
        "relations in Scene context to determine the covering relationship. In the "
        "occlusion graph, an arrow source -> target means that the source object has "
        "a clear covering/occluding relationship over the target object.\n\n"

        "Rules:\n"
        "- The target object must be one object id from summary.json.\n"
        "- Do not invent objects or ids that are not in summary.json.\n"
        "- Do not select object parts.\n"
        "- Do not generate grasp poses.\n"
        "- Do not decide visibility branches such as fully_visible or partially_visible.\n"
        "- selection_mode must be target when selecting a visible target, "
        f"{HIDDEN_TARGET_OCCLUDER_MODE} when returning visible occluders for a "
        "completely hidden target, or none when neither can be identified.\n"
        "- If no listed target or instruction-grounded visible occluder can satisfy "
        "the instruction, set target_present=false, selection_mode=none, and return "
        "an empty candidate_object_ids list.\n"
        "- If multiple objects are plausible, include them in candidate_object_ids "
        "but still choose the best target_object_id.\n"
        "- Return a concise reason explaining the task interpretation and object choice.\n\n"

        "Required JSON shape:\n"
        "{\n"
        '  "target_present": true or false,\n'
        '  "selection_mode": "target | visible_occluder_for_hidden_target | none",\n'
        '  "inferred_task": "short phrase or null",\n'
        '  "target_object_id": integer or null,\n'
        '  "target_category": string or null,\n'
        '  "candidate_object_ids": [integers],\n'
        '  "reason": "short explanation"\n'
        "}\n\n"

        f"User instruction: {instruction}\n\n"
        "Scene context:\n"
        f"{json.dumps(scene_context, ensure_ascii=False, indent=2)}"
    )


def _scene_context(
    summary: dict[str, Any],
    objects: list[SceneObject],
    occlusion: dict[int, set[int]],
) -> dict[str, Any]:
    """Build the compact object-level context sent to the Intent VLM.

    The full numeric occlusion matrix stays in ``summary.json`` for Reason and
    evaluation.  Intent receives object ids plus the derived ``occluded_by``
    relations, while the attached occlusion graph provides the visual form.
    """
    return {
        "scene_id": summary.get("scene_id"),
        "objects": [obj.to_json() for obj in objects],
        "occlusion_direction": "source object occludes target object",
        "occluded_by": {
            str(object_id): sorted(blockers) for object_id, blockers in sorted(occlusion.items())
        },
        "matrix_labels": summary.get("matrix_labels", []),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _objects_from_summary(summary: dict[str, Any]) -> list[SceneObject]:
    labels = _matrix_labels(summary.get("matrix_labels", []))
    points = _summary_points(summary)

    object_ids = sorted(labels) if labels else sorted(points)
    objects = []
    for object_id in object_ids:
        point = points.get(object_id, {})
        raw_label = labels.get(object_id) or str(point.get("label") or f"object_{object_id}")
        objects.append(
            SceneObject(
                object_id=object_id,
                category=_canonical_category(raw_label),
                raw_label=raw_label,
                x=_optional_int(point.get("x")),
                y=_optional_int(point.get("y")),
            )
        )
    return objects


def _summary_points(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    points: dict[int, dict[str, Any]] = {}
    for point in summary.get("molmo_points", []) or []:
        if "molmo_id" in point:
            points[int(point["molmo_id"])] = point
    for point in summary.get("object_points", []) or []:
        if "object_id" in point:
            points[int(point["object_id"])] = point
    return points


def _matrix_labels(matrix_labels: Iterable[Any]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for item in matrix_labels:
        match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", str(item))
        if match:
            labels[int(match.group(1))] = match.group(2)
    return labels


def _canonical_category(label: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", label.lower())
        if token not in _SIDE_WORDS and token not in _STOPWORDS
    ]
    return " ".join(tokens) if tokens else label.strip().lower()


def _occlusion_from_scene(
    summary: dict[str, Any],
    graph: dict[str, Any] | None,
) -> dict[int, set[int]]:
    occluded_by: dict[int, set[int]] = {}
    matrix_labels = _matrix_labels(summary.get("matrix_labels", []))
    matrix_ids = sorted(matrix_labels)
    matrix = summary.get("occlusion_matrix")

    if isinstance(matrix, list) and matrix_ids:
        for source_index, row in enumerate(matrix):
            if source_index >= len(matrix_ids) or not isinstance(row, list):
                continue
            source_id = matrix_ids[source_index]
            for target_index, value in enumerate(row):
                if target_index >= len(matrix_ids):
                    continue
                if _positive(value):
                    target_id = matrix_ids[target_index]
                    occluded_by.setdefault(target_id, set()).add(source_id)

    graph_payload = graph.get("graph", graph) if graph else {}
    for edge in graph_payload.get("edges", []) if isinstance(graph_payload, dict) else []:
        source_id = _edge_object_id(edge, "source_molmo_id", "source")
        target_id = _edge_object_id(edge, "target_molmo_id", "target")
        if source_id is not None and target_id is not None:
            occluded_by.setdefault(target_id, set()).add(source_id)

    return occluded_by


def _resolve_image_paths(
    summary_path: Path,
    summary: dict[str, Any],
    explicit_paths: Iterable[str | Path] | None,
) -> list[Path]:
    summary_dir = summary_path.parent
    object_sheet_candidates = [
        summary.get("final_objects_sheet_png"),
        summary_dir / "final_objects_sheet.png",
    ]
    if explicit_paths:
        return _existing_unique_paths(summary_dir, explicit_paths) + _existing_unique_paths(
            summary_dir,
            object_sheet_candidates,
        )

    primary_candidates = [
        summary.get("perception_label_png"),
        summary.get("graph_png"),
        summary_dir / "label_2_vlm.png",
        summary_dir / "label_3_final.png",
        summary_dir / "scene_labeled.png",
        summary_dir / "occlusion_graph.png",
    ]
    fallback_candidates = [
        summary.get("image_path"),
        summary_dir / "scene_image.png",
        summary_dir / "scene.png",
    ]

    resolved = _existing_unique_paths(summary_dir, primary_candidates)
    if resolved:
        return resolved + _existing_unique_paths(summary_dir, object_sheet_candidates)
    return (
        _existing_unique_paths(summary_dir, fallback_candidates)
        + _existing_unique_paths(summary_dir, object_sheet_candidates)
    )


def _existing_unique_paths(
    summary_dir: Path,
    candidates: Iterable[Any],
) -> list[Path]:
    resolved: list[Path] = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists() and not path.is_absolute():
            path = summary_dir / path
        if not path.exists():
            continue
        key = path.resolve()
        if key not in seen:
            resolved.append(path)
            seen.add(key)
    return resolved


def _default_graph_path(summary_path: Path, summary: dict[str, Any]) -> Path | None:
    candidates = [
        summary.get("graph_json"),
        summary.get("occlusion_graph_json"),
        summary_path.with_name("occlusion_graph.json"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = summary_path.parent / path
        if path.exists():
            return path
    return None


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise IntentHandleError(f"VLM did not return JSON: {stripped}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise IntentHandleError("VLM JSON response must be an object.")
    return parsed


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_caption(path: Path) -> str:
    """Describe an attached image so the VLM knows its role and ID namespace."""
    name = path.name.lower()
    if name in {"label_2_vlm.png", "label_3_final.png", "scene_labeled.png"}:
        return (
            "Labeled scene RGB: use the full scene layout, object outlines, and "
            "numeric object IDs for spatial reasoning."
        )
    if "occlusion_graph" in name:
        return (
            "Occlusion graph: an arrow source -> target means the source object "
            "significantly covers/occludes the target object."
        )
    if name in {"final_objects_sheet.png", "vlm_rgb_objects_sheet.png"}:
        return (
            "Final-object ID sheet: each cell shows one complete assembled object "
            "isolated on white and labeled with its object ID."
        )
    if "sam2_rgb_parts_sheet" in name:
        return (
            "SAM2 part-ID sheet: these numbers are SAM2 part IDs, not object IDs."
        )
    return f"Additional scene visual reference: {path.name}."


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return bool(value)


def _edge_object_id(edge: dict[str, Any], molmo_key: str, node_key: str) -> int | None:
    value = edge.get(molmo_key, edge.get(node_key))
    object_id = _optional_int(value)
    if object_id is None:
        return None
    if molmo_key in edge:
        return object_id
    return object_id + 1


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    ids = []
    for item in value:
        item_id = _optional_int(item)
        if item_id is not None and item_id not in ids:
            ids.append(item_id)
    return ids


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1"}
    return bool(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a natural-language instruction to a target object using a VLM.",
    )
    parser.add_argument("instruction", help="Natural-language user instruction.")
    parser.add_argument("--summary", required=True, type=Path, help="Path to summary.json.")
    parser.add_argument("--occlusion-graph", type=Path, default=None)
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        default=None,
        help="Optional image path. Can be passed multiple times.",
    )
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    result = resolve_intent(
        args.instruction,
        args.summary,
        args.occlusion_graph,
        image_paths=args.images,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
    )
    payload = result.to_json()

    if not args.json_only:
        if result.target_object:
            print(
                "target_object: "
                f"id={result.target_object.object_id}, "
                f"category={result.target_object.category}"
            )
        else:
            print("target_object: null")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
