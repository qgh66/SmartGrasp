"""Intent handler that supplies both part-ID and whole-object-ID sheets.

This opt-in handler reuses the standard intent result validation and transport,
while giving the intent VLM both complementary visual representations produced
by the same frozen perception output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import intent_handler as original


IntentHandleError = original.IntentHandleError
IntentResult = original.IntentResult
SceneObject = original.SceneObject
VLMClient = original.VLMClient


class CombinationResponsesVLMClient(original.ResponsesVLMClient):
    """Intent client whose prompt explains both complementary ID sheets."""

    def _content_parts(
        self,
        instruction: str,
        scene_context: dict[str, Any],
        image_paths: Iterable[Path],
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": _build_combination_prompt(instruction, scene_context)}
        ]
        for image_path in image_paths:
            parts.append(
                {
                    "type": "text",
                    "text": f"Attached image file: {image_path.name}",
                }
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": original._image_data_url(image_path),
                        "detail": "high",
                    },
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
    api_key_env: str = original.DEFAULT_API_KEY_ENV,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = original.DEFAULT_TIMEOUT,
) -> IntentResult:
    """Resolve intent using both part-ID and final whole-object-ID cutouts."""

    summary_file = Path(summary_path)
    summary = original._load_json(summary_file)
    graph_file = (
        Path(occlusion_graph_path)
        if occlusion_graph_path
        else original._default_graph_path(summary_file, summary)
    )
    graph = original._load_json(graph_file) if graph_file and graph_file.exists() else None

    objects = original._objects_from_summary(summary)
    occlusion = original._occlusion_from_scene(summary, graph)
    scene_context = original._scene_context(
        summary,
        objects,
        occlusion,
        occlusion_graph_available=graph is not None,
    )
    scene_context["combined_sheet_semantics"] = {
        "sam2_rgb_parts_sheet.png": (
            "isolated SAM2 parts labeled with part IDs; map every part ID through "
            "sam2_part_id_to_object_id before selecting an object"
        ),
        "vlm_rgb_objects_sheet.png": (
            "isolated whole-object RGB cutouts labeled with final selectable object IDs"
        ),
    }

    resolved_images = _resolve_combination_image_paths(summary_file, summary, image_paths)
    vlm_client = client or CombinationResponsesVLMClient.from_env(
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )
    print(
        "[INTENT combination] calling VLM "
        f"model={getattr(vlm_client, 'model', model)}, "
        f"instruction={instruction!r}, "
        f"summary={summary_file}, "
        f"graph={graph_file}, "
        f"images={[str(path) for path in resolved_images]}"
    )

    decision = vlm_client.choose_target(instruction, scene_context, resolved_images)
    return original._result_from_vlm_decision(decision, objects, occlusion)


def _resolve_combination_image_paths(
    summary_path: Path,
    summary: dict[str, Any],
    explicit_paths: Iterable[str | Path] | None,
) -> list[Path]:
    summary_dir = summary_path.parent
    if explicit_paths:
        return original._existing_unique_paths(
            summary_dir,
            [Path(path) for path in explicit_paths],
        )

    primary_candidates = [
        summary.get("perception_label_png"),
        summary.get("sam2_rgb_parts_sheet_png"),
        summary.get("vlm_rgb_objects_sheet_png"),
        summary.get("graph_png"),
        summary_dir / "label_2_vlm.png",
        summary_dir / "sam2_rgb_parts_sheet.png",
        summary_dir / "vlm_rgb_objects_sheet.png",
        summary_dir / "occlusion_graph.png",
    ]
    resolved = original._existing_unique_paths(summary_dir, primary_candidates)
    if resolved:
        return resolved
    return original._existing_unique_paths(
        summary_dir,
        [summary.get("image_path"), summary_dir / "scene_image.png", summary_dir / "scene.png"],
    )


def _build_combination_prompt(instruction: str, scene_context: dict[str, Any]) -> str:
    return (
        "You are a highly capable robotic assistant designed to support grasping "
        "tasks in real-world environments. Your role here is intent recognition: "
        "analyze the user instruction, identify the task-relevant scene object, "
        "and return only the selected object information.\n\n"
        "Follow these reasoning steps internally:\n"
        "Step 1. Understand the user's underlying intention and implicit task requirements.\n"
        "Step 2. Compare the object list and all attached images. Use "
        "vlm_rgb_objects_sheet.png to inspect each complete physical object. Also use "
        "sam2_rgb_parts_sheet.png to inspect small or partially visible components that "
        "may reveal an object's shape, color, material, or function. The labels on the "
        "part sheet are SAM2 part IDs, not selectable object IDs. Resolve each part ID "
        "through sam2_part_id_to_object_id before drawing an object-level conclusion. "
        "The labels on the whole-object sheet are final selectable object IDs.\n"
        "Step 3. For top, bottom, upper, lower, left, and right, use the labeled scene "
        "image as the default 2D location reference. If the instruction says an object "
        "is underneath another object, use the occlusion graph to determine stacking "
        "or depth order instead of relying only on 2D position.\n\n"
        "Attached image guide:\n"
        "- label_2_vlm.png shows final object IDs in their original scene locations.\n"
        "- sam2_rgb_parts_sheet.png shows isolated segmented parts labeled with SAM2 "
        "part IDs. Convert them using sam2_part_id_to_object_id.\n"
        "- vlm_rgb_objects_sheet.png shows isolated complete-object RGB cutouts labeled "
        "with final selectable object IDs.\n"
        "- In occlusion_graph.png, an arrow points from the object that visibly "
        "covers/occludes another object toward the covered/occluded object.\n\n"
        "Rules:\n"
        "- The target must be one object ID from summary.json.\n"
        "- Do not invent objects or IDs.\n"
        "- Never return a SAM2 part ID as target_object_id.\n"
        "- Do not generate grasp poses or decide visibility branches.\n"
        "- If no listed object can satisfy the instruction, set target_present=false.\n"
        "- If multiple objects are plausible, include them in candidate_object_ids but "
        "still choose the single most likely target_object_id.\n"
        "- IMPORTANT: If the user's description is ambiguous, combine the scene object "
        "information with the object list. Compare shape, color, function, and spatial "
        "position, then select the single object most likely to satisfy the need.\n"
        "- Return a concise reason.\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "target_present": true or false,\n'
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


__all__ = ["IntentResult", "SceneObject", "resolve_intent"]
