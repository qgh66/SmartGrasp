#!/usr/bin/env python
"""Unified JSON entrypoint for the SmartGrasp execution layer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    from .reveal_api import execute_reveal_action
except ImportError:
    from reveal_api import execute_reveal_action

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT / "graspnet-workspace"
sys.path.insert(0, str(WORKSPACE_ROOT))

from simulation.reveal_push import run_reveal_push_scene


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    raw = Path(os.path.expanduser(str(path)))
    if raw.is_absolute():
        return raw
    return (REPO_ROOT / raw).resolve()


def workspace_arg(path: str | os.PathLike[str]) -> str:
    """Return a path suitable for scripts that run from graspnet-workspace."""
    raw = Path(os.path.expanduser(str(path)))
    if raw.is_absolute():
        try:
            return str(raw.resolve().relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(raw)

    parts = raw.parts
    if parts and parts[0] == "graspnet-workspace":
        return str(Path(*parts[1:]))
    return str(raw)


def default_raw_output(request: dict[str, Any], action_type: str) -> str:
    request_id = request.get("request_id") or "execution_request"
    return f"results/{request_id}_{action_type}.json"


def load_reason_request(request: dict[str, Any]) -> dict[str, Any]:
    """Populate an execution request from one Reason scene summary."""
    reason_config = request.get("reason") or {}
    summary_value = (
        request.get("reason_summary_path")
        or reason_config.get("summary_path")
    )
    if not summary_value:
        return request

    summary_path = resolve_repo_path(summary_value)
    summary = load_json(summary_path)
    grasp_object = summary.get("grasp_object") or {}
    if grasp_object.get("id") is None:
        raise ValueError(
            f"Reason summary has no grasp_object.id: {summary_path}"
        )

    hydrated = dict(request)
    hydrated["reason_summary_path"] = str(summary_path)
    hydrated["branch"] = hydrated.get("branch") or summary.get("branch")
    if not hydrated.get("task_type"):
        hydrated["task_type"] = (
            "grasp"
            if hydrated.get("branch") == "fully_visible"
            else "reveal"
        )

    obj = dict(hydrated.get("object") or {})
    if obj.get("id") is None:
        obj["id"] = grasp_object.get("id")
    if not obj.get("category"):
        obj["category"] = grasp_object.get("label")
    obj.setdefault(
        "role",
        "target" if hydrated["task_type"] == "grasp" else "occluder",
    )
    hydrated["object"] = obj

    scene = dict(hydrated.get("scene") or {})
    if not scene.get("occlusion_graph_path"):
        default_graph = (
            summary_path.parent.parent
            / "perception"
            / "occlusion_graph.json"
        )
        scene["occlusion_graph_path"] = str(default_graph)
    hydrated["scene"] = scene
    return hydrated


def resolve_reason_object_mask(request: dict[str, Any]) -> Path | None:
    """Resolve object.id to its whole-object Perception mask."""
    obj = request.get("object") or {}
    scene = request.get("scene") or {}
    object_id = obj.get("id")

    direct_mask = scene.get("object_mask_path") or scene.get("mask_path")
    if direct_mask:
        path = resolve_repo_path(direct_mask)
        if not path.exists():
            raise FileNotFoundError(f"Execution object mask not found: {path}")
        return path

    graph_value = (
        scene.get("occlusion_graph_path")
        or scene.get("perception_graph_path")
    )
    if graph_value is None or object_id is None:
        return None

    graph_path = resolve_repo_path(graph_value)
    graph_data = load_json(graph_path)
    nodes = (graph_data.get("graph") or {}).get("nodes") or []
    matching_node = next(
        (
            node
            for node in nodes
            if int(node.get("object_id", node.get("node_id", -1)))
            == int(object_id)
        ),
        None,
    )
    if matching_node is None:
        raise ValueError(
            f"Reason object_id={object_id} is absent from {graph_path}"
        )

    mask_value = matching_node.get("mask_path") or matching_node.get("mask_file")
    if not mask_value:
        raise ValueError(
            f"Perception node object_id={object_id} has no mask path"
        )

    raw_mask = Path(os.path.expanduser(str(mask_value)))
    candidates = (
        [raw_mask]
        if raw_mask.is_absolute()
        else [
            graph_path.parent / raw_mask,
            graph_path.parent.parent / raw_mask,
            REPO_ROOT / raw_mask,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Mask for Reason object_id={object_id} not found; "
        f"graph={graph_path}, mask_path={mask_value!r}"
    )


def normalize_result(
    request: dict[str, Any],
    *,
    status: str,
    success: bool,
    action_type: str,
    result: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    request_reloop: bool = True,
) -> dict[str, Any]:
    obj = request.get("object", {})
    return {
        "request_id": request.get("request_id"),
        "status": status,
        "success": bool(success),
        "branch": request.get("branch"),
        "action_type": action_type,
        "object": {
            "id": obj.get("id"),
            "name": obj.get("name"),
            "category": obj.get("category"),
            "role": obj.get("role"),
        },
        "result": result,
        "artifacts": artifacts or {},
        "request_reloop": bool(request_reloop),
        "diagnostics": diagnostics or {},
    }


def run_grasp(request: dict[str, Any]) -> dict[str, Any]:
    obj = request.get("object", {})
    scene = request.get("scene", {})
    execution = request.get("execution", {})

    target_name = obj.get("name")
    target_mask = resolve_reason_object_mask(request)
    if not target_name and target_mask is None:
        raise ValueError(
            "fully_visible/grasp request requires object.name or "
            "object.id plus scene.occlusion_graph_path"
        )

    scene_config = scene.get("scene_config") or "graspnet-workspace/config/industrial_scene.json"
    raw_output = execution.get("output") or default_raw_output(request, "grasp")
    top_k = str(execution.get("top_k", 5))
    device = str(execution.get("device", "cuda:0"))

    cmd = [
        "bash",
        str(REPO_ROOT / "run_grasp_simulation.sh"),
        "--scene-config",
        workspace_arg(scene_config),
        "--top_k",
        top_k,
        "--device",
        device,
        "--output",
        workspace_arg(raw_output),
    ]
    if target_mask is not None:
        cmd.extend(["--target-mask", str(target_mask)])
        cmd.extend(
            [
                "--target-mask-min-iou",
                str(execution.get("target_mask_min_iou", 0.01)),
            ]
        )
    elif target_name:
        cmd.extend(["--target-object", str(target_name)])
    env = os.environ.copy()
    if execution.get("gui"):
        env["PYBULLET_GUI"] = "1"

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    raw_output_path = resolve_repo_path(raw_output)
    if not raw_output_path.exists() and not Path(raw_output).is_absolute():
        raw_output_path = WORKSPACE_ROOT / workspace_arg(raw_output)

    raw_result: dict[str, Any] = {}
    if raw_output_path.exists():
        raw_result = load_json(raw_output_path)
    mapped_name = raw_result.get("target_object_name")
    if mapped_name:
        request.setdefault("object", {})["name"] = mapped_name

    success_count = int(raw_result.get("success", 0) or 0)
    total = int(raw_result.get("total", 0) or 0)
    grasps = raw_result.get("grasps") or []
    best_grasp = grasps[0] if grasps else {}

    status = "finished" if completed.returncode == 0 else "failed"
    diagnostics = {
        "returncode": completed.returncode,
        "command": cmd,
        "log_tail": completed.stdout.splitlines()[-80:],
        "grasp_filter": raw_result.get("grasp_filter"),
        "object_point_counts": raw_result.get("object_point_counts"),
        "reason_object_id": obj.get("id"),
        "target_selection": raw_result.get("target_selection"),
    }
    artifacts = {
        "result_json": str(raw_output_path),
        "viz_data_pkl": str(raw_output_path.with_name(raw_output_path.stem + "_viz_data.pkl")),
    }
    result = {
        "total": total,
        "success_count": success_count,
        "best_index": 0 if best_grasp else None,
        "failure_reason": best_grasp.get("failure_reason"),
        "executed_pose": {
            "translation": best_grasp.get("translation"),
            "rotation": best_grasp.get("rotation"),
            "width": best_grasp.get("width"),
        },
        "frame_log": "available in raw result grasps[i].frame_log when generated",
    }
    return normalize_result(
        request,
        status=status,
        success=success_count > 0,
        action_type="grasp",
        result=result,
        artifacts=artifacts,
        diagnostics=diagnostics,
        request_reloop=True,
    )


def run_reveal(request: dict[str, Any]) -> dict[str, Any]:
    obj = request.get("object", {})
    scene = request.get("scene", {})
    execution = request.get("execution", {})
    reveal = request.get("reveal", {})
    action_type = reveal.get("action_type", "push")
    if action_type != "push":
        center_point = reveal.get("center_point")
        if center_point is None:
            raise ValueError("reveal request requires reveal.center_point")
        plan = execute_reveal_action(
            occluder_id=obj.get("id"),
            center_point=center_point,
            action_type=action_type,
            move_distance=float(reveal.get("move_distance", 0.05)),
        )
        diagnostics = {
            "physical_simulation": "not_implemented",
            "note": "Only push is physically implemented in PyBullet for now.",
        }
        return normalize_result(
            request,
            status="not_implemented",
            success=False,
            action_type="reveal",
            result=plan,
            artifacts={},
            diagnostics=diagnostics,
            request_reloop=bool(plan.get("request_reloop", True)),
        )

    scene_config = scene.get("scene_config") or "graspnet-workspace/config/industrial_scene.json"
    raw_output = execution.get("output") or default_raw_output(request, "reveal")
    target_mask = resolve_reason_object_mask(request)
    if not obj.get("name") and target_mask is None:
        raise ValueError(
            "reveal request requires object.name or "
            "object.id plus scene.occlusion_graph_path"
        )
    run_data = run_reveal_push_scene(
        scene_config=scene_config,
        object_name=obj.get("name"),
        object_mask_path=target_mask,
        target_mask_min_iou=float(
            execution.get("target_mask_min_iou", 0.01)
        ),
        center_point=reveal.get("center_point"),
        direction=reveal.get("direction", [1.0, 0.0, 0.0]),
        move_distance=float(reveal.get("move_distance", 0.05)),
        output=raw_output,
        gui=bool(execution.get("gui", False)),
    )
    raw_result = run_data["result"]
    mapped_name = raw_result.get("target_object_name")
    if mapped_name:
        request.setdefault("object", {})["name"] = mapped_name
    push_result = (raw_result.get("grasps") or [{}])[0]
    diagnostics = {
        "physical_simulation": "pybullet_jaka_push",
        "signed_displacement": push_result.get("signed_displacement"),
        "success_threshold": push_result.get("success_threshold"),
        "object_point_counts_before": raw_result.get("object_point_counts_before"),
        "object_point_counts_after": raw_result.get("object_point_counts_after"),
        "reason_object_id": obj.get("id"),
        "target_selection": raw_result.get("target_selection"),
    }
    artifacts = {
        "result_json": run_data["result_json"],
        "viz_data_pkl": run_data["viz_data_pkl"],
    }
    result = {
        "total": raw_result.get("total", 1),
        "success_count": raw_result.get("success", 0),
        "best_index": 0,
        "failure_reason": None if push_result.get("success") else "push_displacement_below_threshold",
        "executed_pose": {
            "translation": push_result.get("translation"),
            "rotation": push_result.get("rotation"),
            "width": push_result.get("width"),
        },
        "push_direction": push_result.get("push_direction"),
        "requested_distance": push_result.get("requested_distance"),
        "actual_displacement": push_result.get("actual_displacement"),
        "signed_displacement": push_result.get("signed_displacement"),
        "start_position": push_result.get("start_position"),
        "final_position": push_result.get("final_position"),
        "frame_log": "available in raw result grasps[0].frame_log",
    }
    return normalize_result(
        request,
        status="finished",
        success=bool(push_result.get("success", False)),
        action_type="reveal",
        result=result,
        artifacts=artifacts,
        diagnostics=diagnostics,
        request_reloop=True,
    )


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    branch = request.get("branch")
    task_type = request.get("task_type")
    if branch == "fully_visible" and task_type == "grasp":
        return run_grasp(request)
    if branch in {"partially_occluded", "fully_occluded"} and task_type == "reveal":
        return run_reveal(request)
    raise ValueError(f"Unsupported execution request: branch={branch!r}, task_type={task_type!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmartGrasp execution request JSON")
    parser.add_argument("--input", required=True, help="Execution request JSON")
    parser.add_argument("--output", required=True, help="Normalized execution result JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = load_reason_request(load_json(resolve_repo_path(args.input)))
    try:
        response = run_request(request)
    except Exception as exc:
        response = normalize_result(
            request,
            status="failed",
            success=False,
            action_type=str(request.get("task_type")),
            result={},
            diagnostics={"error": str(exc)},
            request_reloop=False,
        )
    write_json(resolve_repo_path(args.output), response)


if __name__ == "__main__":
    main()
