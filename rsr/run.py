from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    ALGORITHMS,
    ALGORITHM_BY_SLUG,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    PROJECT_ROOT,
    REASON_MODELS,
    TEST_CASES,
    safe_model_name,
)
from .evaluate import evaluate_all


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_timeout(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("timed out", "timeout", "readtimeout", "apitimeouterror"))


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def build_annotation_view(
    shared_perception: Path,
    run_perception_dir: Path,
    instruction: str,
) -> None:
    run_perception_dir.mkdir(parents=True, exist_ok=True)
    for artifact in shared_perception.iterdir():
        if artifact.name == "summary.json":
            continue
        _relative_symlink(artifact, run_perception_dir / artifact.name)

    summary = _load_json(shared_perception / "summary.json")
    summary["annotation"] = instruction
    summary["instruction"] = instruction
    summary["shared_perception_dir"] = str(shared_perception.resolve())
    _write_json(run_perception_dir / "summary.json", summary)


def _label_for_id(perception_summary: dict[str, Any], object_id: int | None) -> str | None:
    if object_id is None:
        return None
    for item in perception_summary.get("molmo_points", []) or perception_summary.get("object_points", []):
        raw_id = item.get("molmo_id", item.get("object_id"))
        try:
            if int(raw_id) == int(object_id):
                label = item.get("label")
                return str(label) if label is not None else f"object_{object_id}"
        except (TypeError, ValueError):
            continue
    return f"object_{object_id}"


def write_selection_result(
    split_root: Path,
    run_scene: Path,
    shared_perception: Path,
    *,
    scene_id: int,
    split: int,
    model: str,
    algorithm: str,
    outcome_status: str | None = None,
) -> dict[str, Any]:
    intent_path = run_scene / "intent" / "intent_result.json"
    reason_path = run_scene / "reason" / "summary.json"
    perception_summary = _load_json(shared_perception / "summary.json")
    intent_payload = _load_json(intent_path) if intent_path.exists() else {}
    reason_payload = _load_json(reason_path) if reason_path.exists() else {}

    intent_id = intent_payload.get("selected_object_id")
    reason_target = reason_payload.get("target_object", {}) or {}
    intent_label = reason_target.get("label") or _label_for_id(perception_summary, intent_id)
    grasp_payload = reason_payload.get("grasp_object", {}) or {}
    grasp_id = grasp_payload.get("id")
    grasp_label = grasp_payload.get("label") or _label_for_id(perception_summary, grasp_id)
    reason_status = str(reason_payload.get("status") or "")
    completed = intent_path.exists() and reason_path.exists()
    if outcome_status is not None:
        selection_status = outcome_status
    elif completed and (not reason_status or reason_status in {"ok", "no_item_found", "selection_no_found"}):
        selection_status = "completed"
    else:
        selection_status = "failed"
    result = {
        "scene_id": scene_id,
        "split": split,
        "model": model,
        "algorithm": algorithm,
        "status": selection_status,
        "reason_status": reason_status or None,
        "intent": {
            "selected_id": intent_id,
            "selected_object": intent_label,
        },
        "reason": {
            "grasp_id": grasp_id,
            "grasp_object": grasp_label,
        },
    }
    _write_json(split_root / "selection.json", result)
    lines = [
        f"scene_id: {scene_id}",
        f"split: {split}",
        f"model: {model}",
        f"algorithm: {algorithm}",
        f"status: {result['status']}",
        f"intent_selected_id: {intent_id}",
        f"intent_selected_object: {intent_label}",
        f"reason_grasp_id: {grasp_id}",
        f"reason_grasp_object: {grasp_label}",
    ]
    (split_root / "selection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run_perception(input_scene: Path, output_category: Path, args: argparse.Namespace) -> Path:
    metadata = _load_json(input_scene / "metadata.json")
    scene_id = int(metadata["scene_id"])
    output_scene = output_category / f"scene_{scene_id}"
    shared_perception = output_scene / "perception"
    if (shared_perception / "summary.json").exists() and not args.force_perception:
        print(f"[perception cached] scene={scene_id} -> {shared_perception}", flush=True)
        return shared_perception

    print(f"[perception start] scene={scene_id} mode={args.perception_mode}", flush=True)

    from perception import run_perception as perception_runner

    parser = perception_runner.build_arg_parser()
    perception_args = parser.parse_args([])
    perception_args.scene_id = scene_id
    perception_args.scene_ids = None
    perception_args.query_obj_id = int(metadata["query_obj_id"])
    perception_args.mode = args.perception_mode
    perception_args.review_model_id = args.perception_review_model
    perception_args.review_base_url = args.perception_review_base_url
    perception_args.review_timeout = args.perception_review_timeout
    perception_args.device = args.device

    output_category.mkdir(parents=True, exist_ok=True)
    perception_runner.OUT_ROOT = output_category
    perception_runner.INPUT_ROOT = output_category / "__priority_input_disabled__"
    source_npz = input_scene / "source.npz"
    original_find_npz_source = perception_runner.find_npz_source
    perception_runner.find_npz_source = lambda requested_scene_id: (source_npz, None)
    frame = pd.DataFrame([
        {
            "sceneId": scene_id,
            "queryObjId": int(metadata["query_obj_id"]),
            "annotation": metadata["annotations"][0]["instruction"],
            "image": {"path": str((input_scene / "scene_image.png").resolve())},
        }
    ])
    try:
        perception_runner.run_pipeline(perception_args, df=frame)
        try:
            from perception.sam2auto import clear_sam2_image_state
            clear_sam2_image_state()
        except Exception:
            pass
    finally:
        perception_runner.find_npz_source = original_find_npz_source

    if args.perception_mode == "gt" and not (shared_perception / "summary.json").exists():
        gt_perception = output_scene / "gt"
        if (gt_perception / "summary.json").exists():
            gt_summary = _load_json(gt_perception / "summary.json")
            build_annotation_view(
                gt_perception,
                shared_perception,
                str(gt_summary.get("annotation", "")),
            )

    if not (shared_perception / "summary.json").exists():
        raise RuntimeError(f"Perception did not create {shared_perception / 'summary.json'}")
    print(f"[perception done] scene={scene_id} -> {shared_perception}", flush=True)
    return shared_perception


def run_reason(
    input_scene: Path,
    result_scene: Path,
    shared_perception: Path,
    annotation: dict[str, Any],
    model: str,
    algorithm_slug: str,
    args: argparse.Namespace,
) -> None:
    scene_id = int(_load_json(input_scene / "metadata.json")["scene_id"])
    split = int(annotation["split"])
    instruction = str(annotation["instruction"])
    algorithm = ALGORITHM_BY_SLUG[algorithm_slug]
    split_root = result_scene / "annotations" / f"split_{split}"
    run_scene = split_root / f"scene_{scene_id}"
    reason_summary = run_scene / "reason" / "summary.json"
    timeout_path = split_root / "timeout.json"
    if timeout_path.exists() and not args.force_reason:
        write_selection_result(
            split_root,
            run_scene,
            shared_perception,
            scene_id=scene_id,
            split=split,
            model=model,
            algorithm=algorithm_slug,
            outcome_status="timeout",
        )
        print(
            f"[reason timeout cached] scene={scene_id} split={split} model={model} algorithm={algorithm_slug}",
            flush=True,
        )
        return
    if reason_summary.exists() and not args.force_reason:
        write_selection_result(
            split_root,
            run_scene,
            shared_perception,
            scene_id=scene_id,
            split=split,
            model=model,
            algorithm=algorithm_slug,
        )
        print(
            f"[reason cached] scene={scene_id} split={split} model={model} algorithm={algorithm_slug}",
            flush=True,
        )
        return

    build_annotation_view(shared_perception, run_scene / "perception", instruction)
    (split_root / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        "-m",
        "reason.run_reason",
        "--root",
        str(split_root),
        "--scene-id",
        str(scene_id),
        "--target-source",
        "intent",
        "--instruction",
        instruction,
        "--model",
        model,
        "--intent-model",
        model,
        "--prior-prompt",
        algorithm.prior_prompt,
        "--ranking-score",
        algorithm.ranking_score,
        "--out-root",
        str(split_root / "run_detail"),
        "--scene-root",
        str(split_root),
        "--quiet",
    ]
    log_path = split_root / "run.log"
    started = time.time()
    print(
        f"[reason start] scene={scene_id} split={split} model={model} algorithm={algorithm_slug}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    reason_payload = _load_json(reason_summary) if reason_summary.exists() else {}
    reason_status = str(reason_payload.get("status") or "")
    timed_out = _is_timeout(log_text) or _is_timeout(reason_status)
    selection = write_selection_result(
        split_root,
        run_scene,
        shared_perception,
        scene_id=scene_id,
        split=split,
        model=model,
        algorithm=algorithm_slug,
        outcome_status="timeout" if timed_out else None,
    )
    elapsed_seconds = round(time.time() - started, 3)
    timeout_record = {
        "scene_id": scene_id,
        "split": split,
        "model": model,
        "algorithm": algorithm_slug,
        "prior_prompt": algorithm.prior_prompt,
        "ranking_score": algorithm.ranking_score,
        "instruction": instruction,
        "timeout_seconds_per_query": 600.0,
        "elapsed_seconds": elapsed_seconds,
        "reason_status": reason_status or None,
        "log": str(log_path),
    }
    if timed_out:
        _write_json(timeout_path, timeout_record)
    _write_json(
        split_root / "run_status.json",
        {
            "scene_id": scene_id,
            "split": split,
            "model": model,
            "algorithm": algorithm_slug,
            "prior_prompt": algorithm.prior_prompt,
            "ranking_score": algorithm.ranking_score,
            "instruction": instruction,
            "command": command,
            "returncode": completed.returncode,
            "valid_output": selection["status"] == "completed",
            "timed_out": timed_out,
            "failure_type": "timeout" if timed_out else (None if selection["status"] == "completed" else "invalid_output"),
            "reason_status": reason_status or None,
            "elapsed_seconds": elapsed_seconds,
            "reason_summary": str(reason_summary),
        },
    )
    if timed_out:
        print(
            f"[reason timeout] scene={scene_id} split={split} model={model} algorithm={algorithm_slug} "
            f"elapsed={elapsed_seconds:.1f}s -> skipped",
            flush=True,
        )
    if completed.returncode != 0 or selection["status"] != "completed":
        raise RuntimeError(f"Reason failed for scene={scene_id} split={split}; see {log_path}")
    print(
        f"[reason done] scene={scene_id} split={split} model={model} algorithm={algorithm_slug} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )


def selected_testcases(names: list[str] | None):
    if not names:
        return TEST_CASES
    wanted = set(names)
    selected = [case for case in TEST_CASES if case.slug in wanted or case.directory_name in wanted]
    missing = wanted - {case.slug for case in selected} - {case.directory_name for case in selected}
    if missing:
        raise ValueError(f"Unknown testcase names: {sorted(missing)}")
    return selected


def selected_models(names: list[str] | None) -> tuple[str, ...]:
    return tuple(names or REASON_MODELS)


def selected_algorithms(names: list[str] | None):
    if not names:
        return ALGORITHMS
    return tuple(ALGORITHM_BY_SLUG[name] for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared perception once and three independent reason calls per scene.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--testcase", action="append", default=None, help="Slug or numbered testcase directory; repeatable.")
    parser.add_argument("--scene-id", type=int, action="append", default=None)
    parser.add_argument("--split", type=int, action="append", choices=[0, 1, 2], default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--perception-only", action="store_true")
    parser.add_argument("--reason-only", action="store_true")
    parser.add_argument("--force-perception", action="store_true")
    parser.add_argument("--force-reason", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--perception-mode", choices=["vlm", "gt"], default="vlm")
    parser.add_argument("--perception-review-model", default="gpt-5.5")
    parser.add_argument("--perception-review-base-url", default=None)
    parser.add_argument("--perception-review-timeout", type=float, default=120.0)
    parser.add_argument(
        "--reason-model",
        action="append",
        default=None,
        help="Reason and intent model. Repeatable; defaults to gpt-5.5 and gpt-4o.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=sorted(ALGORITHM_BY_SLUG),
        default=None,
        help="Repeatable; defaults to information_gain and theory. Both use the graspability prior.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.perception_only and args.reason_only:
        parser.error("--perception-only and --reason-only are mutually exclusive")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    allowed_scene_ids = set(args.scene_id or [])
    allowed_splits = set(args.split or [0, 1, 2])
    models = selected_models(args.reason_model)
    algorithms = selected_algorithms(args.algorithm)
    failures = []

    for testcase in selected_testcases(args.testcase):
        input_category = input_root / testcase.directory_name
        perception_category = output_root / "perception" / testcase.directory_name
        scene_dirs = sorted(input_category.glob("scene_*"))
        if allowed_scene_ids:
            scene_dirs = [path for path in scene_dirs if int(path.name.removeprefix("scene_")) in allowed_scene_ids]
        if args.limit_scenes is not None:
            scene_dirs = scene_dirs[: args.limit_scenes]

        for input_scene in scene_dirs:
            metadata = _load_json(input_scene / "metadata.json")
            perception_scene = perception_category / input_scene.name
            try:
                if args.reason_only:
                    shared_perception = perception_scene / "perception"
                    if not (shared_perception / "summary.json").exists():
                        raise FileNotFoundError(f"Missing shared perception: {shared_perception}")
                else:
                    shared_perception = run_perception(input_scene, perception_category, args)
            except Exception as exc:
                failure = {
                    "testcase": testcase.directory_name,
                    "scene": input_scene.name,
                    "stage": "perception",
                    "error": str(exc),
                }
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
                continue

            if args.perception_only:
                continue

            for model in models:
                for algorithm in algorithms:
                    result_scene = (
                        output_root
                        / "results"
                        / safe_model_name(model)
                        / algorithm.slug
                        / testcase.directory_name
                        / input_scene.name
                    )
                    for annotation in metadata["annotations"]:
                        if int(annotation["split"]) not in allowed_splits:
                            continue
                        try:
                            run_reason(
                                input_scene,
                                result_scene,
                                shared_perception,
                                annotation,
                                model,
                                algorithm.slug,
                                args,
                            )
                        except Exception as exc:
                            failure = {
                                "testcase": testcase.directory_name,
                                "scene": input_scene.name,
                                "stage": "reason",
                                "split": int(annotation["split"]),
                                "model": model,
                                "algorithm": algorithm.slug,
                                "error": str(exc),
                            }
                            failures.append(failure)
                            print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
                            if args.fail_fast:
                                raise

    _write_json(output_root / "run_failures.json", {"failures": failures})
    if not args.perception_only and not args.scene_id and not args.testcase and not args.split and args.limit_scenes is None:
        print(json.dumps(
            evaluate_all(
                input_root,
                output_root,
                models=list(models),
                algorithms=[algorithm.slug for algorithm in algorithms],
            ),
            ensure_ascii=False,
            indent=2,
        ))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
