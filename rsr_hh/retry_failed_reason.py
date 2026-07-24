from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import (
    ALGORITHM_BY_SLUG,
    PROJECT_ROOT,
    safe_model_name,
)
from .evaluate import evaluate_all
from .run import run_perception, run_reason, selected_testcases


DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "rsr_hh" / "data" / "gpt4o_first10_four_categories"
)
DEFAULT_TESTCASES = (
    "hard_ambiguous",
    "medium_ambiguous",
    "medium_unambiguous",
    "hard_unambiguous",
)
DEFAULT_ALGORITHMS = (
    "information_gain_original",
    "information_gain_graspability",
    "theory_original",
    "theory_graspability",
)
COMPLETED_REASON_STATUSES = {
    "",
    "ok",
    "no_item_found",
    "selection_no_found",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        return _load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class RetryTask:
    testcase: str
    input_scene: Path
    scene_id: int
    query_obj_id: int
    split: int
    model: str
    algorithm: str
    annotation: dict[str, Any]
    shared_perception: Path
    result_scene: Path

    @property
    def split_root(self) -> Path:
        return self.result_scene / "annotations" / f"split_{self.split}"

    @property
    def run_scene(self) -> Path:
        return self.split_root / f"scene_{self.scene_id}"

    @property
    def key(self) -> str:
        return (
            f"{self.testcase}/{self.input_scene.name}/split_{self.split}/"
            f"{self.model}/{self.algorithm}"
        )


@dataclass(frozen=True)
class PerceptionRetryTask:
    testcase: str
    input_scene: Path
    scene_id: int
    query_obj_id: int
    perception_category: Path
    shared_perception: Path

    @property
    def key(self) -> str:
        return f"{self.testcase}/{self.input_scene.name}/perception"


def _is_complete(task: RetryTask) -> bool:
    timeout_path = task.split_root / "timeout.json"
    run_status_path = task.split_root / "run_status.json"
    selection_path = task.split_root / "selection.json"
    reason_path = task.run_scene / "reason" / "summary.json"
    intent_path = task.run_scene / "intent" / "intent_result.json"
    if timeout_path.exists():
        return False
    if not reason_path.exists() or not intent_path.exists():
        return False

    run_status = _load_json_or_empty(run_status_path)
    if run_status and (
        bool(run_status.get("timed_out"))
        or run_status.get("valid_output") is False
    ):
        return False
    selection = _load_json_or_empty(selection_path)
    if selection and selection.get("status") != "completed":
        return False

    reason_payload = _load_json_or_empty(reason_path)
    reason_status = str(reason_payload.get("status") or "")
    return reason_status in COMPLETED_REASON_STATUSES


def _failure_detail(task: RetryTask) -> dict[str, Any]:
    run_status_path = task.split_root / "run_status.json"
    selection_path = task.split_root / "selection.json"
    timeout_path = task.split_root / "timeout.json"
    run_log_path = task.split_root / "run.log"
    run_status = _load_json_or_empty(run_status_path)
    selection = _load_json_or_empty(selection_path)

    error_line = None
    if run_log_path.exists():
        text = run_log_path.read_text(encoding="utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            lowered = line.lower()
            if any(
                token in lowered
                for token in (
                    "connection error",
                    "timed out",
                    "timeout",
                    "[error]",
                    "traceback",
                )
            ):
                error_line = line.strip()
                break

    return {
        "key": task.key,
        "testcase": task.testcase,
        "case_directory": task.input_scene.name,
        "scene_id": task.scene_id,
        "query_obj_id": task.query_obj_id,
        "split": task.split,
        "model": task.model,
        "algorithm": task.algorithm,
        "selection_status": selection.get("status"),
        "failure_type": run_status.get("failure_type"),
        "timed_out": bool(run_status.get("timed_out")) or timeout_path.exists(),
        "returncode": run_status.get("returncode"),
        "error": error_line,
        "run_log": str(run_log_path),
    }


def _collect_tasks(
    args: argparse.Namespace,
) -> tuple[list[RetryTask], list[PerceptionRetryTask]]:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    allowed_scene_ids = set(args.scene_id or [])
    allowed_query_ids = set(args.query_obj_id or [])
    allowed_case_directories = set(args.case_directory or [])
    allowed_splits = set(args.split or (0, 1, 2))
    testcases = selected_testcases(args.testcase or list(DEFAULT_TESTCASES))

    reason_tasks: list[RetryTask] = []
    perception_tasks: list[PerceptionRetryTask] = []
    for testcase in testcases:
        input_category = input_root / testcase.directory_name
        perception_category = (
            output_root / "perception" / testcase.directory_name
        )
        for input_scene in sorted(input_category.glob("scene_*")):
            metadata = _load_json(input_scene / "metadata.json")
            scene_id = int(metadata["scene_id"])
            query_obj_id = int(metadata["query_obj_id"])
            if allowed_scene_ids and scene_id not in allowed_scene_ids:
                continue
            if allowed_query_ids and query_obj_id not in allowed_query_ids:
                continue
            if (
                allowed_case_directories
                and input_scene.name not in allowed_case_directories
            ):
                continue

            shared_perception = (
                perception_category / input_scene.name / "perception"
            )
            annotations = {
                int(item["split"]): item
                for item in metadata["annotations"]
            }
            case_reason_tasks: list[RetryTask] = []
            for model in args.model:
                for algorithm in args.algorithm:
                    result_scene = (
                        output_root
                        / "results"
                        / safe_model_name(model)
                        / algorithm
                        / testcase.directory_name
                        / input_scene.name
                    )
                    for split in sorted(allowed_splits):
                        task = RetryTask(
                            testcase=testcase.directory_name,
                            input_scene=input_scene,
                            scene_id=scene_id,
                            query_obj_id=query_obj_id,
                            split=split,
                            model=model,
                            algorithm=algorithm,
                            annotation=annotations[split],
                            shared_perception=shared_perception,
                            result_scene=result_scene,
                        )
                        if _is_complete(task):
                            continue
                        case_reason_tasks.append(task)

            if not case_reason_tasks:
                continue
            if (shared_perception / "summary.json").exists():
                reason_tasks.extend(case_reason_tasks)
            else:
                perception_tasks.append(
                    PerceptionRetryTask(
                        testcase=testcase.directory_name,
                        input_scene=input_scene,
                        scene_id=scene_id,
                        query_obj_id=query_obj_id,
                        perception_category=perception_category,
                        shared_perception=shared_perception,
                    )
                )
    return reason_tasks, perception_tasks


def _perception_record(task: PerceptionRetryTask) -> dict[str, Any]:
    return {
        "key": task.key,
        "testcase": task.testcase,
        "case_directory": task.input_scene.name,
        "scene_id": task.scene_id,
        "query_obj_id": task.query_obj_id,
        "shared_perception": str(task.shared_perception),
    }


def _perception_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        force_perception=True,
        perception_mode="vlm",
        perception_review_model=args.perception_review_model,
        perception_review_base_url=args.perception_review_base_url,
        perception_review_timeout=args.perception_review_timeout,
        device=args.device,
        sam2_points_per_side=24,
        sam2_pred_iou_thresh=0.68,
        sam2_stability_score_thresh=0.83,
        sam2_crop_n_layers=0,
        depth_sam2_crop_n_layers=1,
        depth_sam2_pred_iou_thresh=0.58,
        depth_sam2_stability_score_thresh=0.73,
        kernel_size=11,
        min_contact_pixels=50,
        min_contact_ratio=0.002,
        mask_clean_kernel=3,
        proposal_min_area_ratio=0.006,
        proposal_max_area_ratio=0.11,
        proposal_border_fraction_threshold=0.18,
    )


def _print_inventory(
    reason_tasks: list[RetryTask],
    perception_tasks: list[PerceptionRetryTask],
) -> None:
    print(
        json.dumps(
            {
                "reason_failures": len(reason_tasks),
                "perception_failures": len(perception_tasks),
                "reason_tasks": [
                    _failure_detail(task) for task in reason_tasks
                ],
                "perception_tasks": [
                    _perception_record(task) for task in perception_tasks
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def retry(args: argparse.Namespace) -> int:
    status_path = (
        args.output_root.resolve() / "retry_failed_reason_status.json"
    )
    attempts: dict[str, int] = {}
    last_errors: dict[str, str] = {}
    round_number = 0

    while True:
        reason_tasks, perception_tasks = _collect_tasks(args)
        if round_number == 0:
            _print_inventory(reason_tasks, perception_tasks)
        if args.dry_run:
            return 0
        if not reason_tasks and not perception_tasks:
            payload = {
                "status": "complete",
                "rounds": round_number,
                "remaining_reason": 0,
                "remaining_perception": 0,
                "attempts_per_task": attempts,
                "last_errors": last_errors,
            }
            _write_json(status_path, payload)
            print(
                f"[retry complete] all selected Perception/Reason tasks are "
                f"valid after "
                f"{round_number} round(s)",
                flush=True,
            )
            if args.evaluate:
                evaluate_all(
                    args.input_root.resolve(),
                    args.output_root.resolve(),
                    models=args.model,
                    algorithms=list(DEFAULT_ALGORITHMS),
                    testcases=selected_testcases(list(DEFAULT_TESTCASES)),
                )
            return 0

        if args.max_rounds > 0 and round_number >= args.max_rounds:
            payload = {
                "status": "max_rounds_reached",
                "rounds": round_number,
                "remaining_reason": [
                    _failure_detail(task) for task in reason_tasks
                ],
                "remaining_perception": [
                    _perception_record(task) for task in perception_tasks
                ],
                "attempts_per_task": attempts,
                "last_errors": last_errors,
            }
            _write_json(status_path, payload)
            print(
                f"[retry stopped] reached max_rounds={args.max_rounds}; "
                f"{len(perception_tasks)} Perception and "
                f"{len(reason_tasks)} Reason task(s) remain",
                flush=True,
            )
            return 1

        round_number += 1
        print(
            f"[retry round {round_number}] starting with "
            f"{len(perception_tasks)} Perception and "
            f"{len(reason_tasks)} Reason task(s)",
            flush=True,
        )
        perception_completed = 0
        for index, task in enumerate(perception_tasks, start=1):
            attempts[task.key] = attempts.get(task.key, 0) + 1
            print(
                f"[perception retry {index}/{len(perception_tasks)}] "
                f"{task.key} attempt={attempts[task.key]}",
                flush=True,
            )
            try:
                run_perception(
                    task.input_scene,
                    task.perception_category,
                    _perception_args(args),
                )
            except Exception as exc:
                last_errors[task.key] = str(exc)

            if (task.shared_perception / "summary.json").exists():
                perception_completed += 1
                last_errors.pop(task.key, None)
                print(f"[perception retry success] {task.key}", flush=True)
            else:
                print(
                    f"[perception retry still failed] {task.key}: "
                    f"{last_errors.get(task.key, 'missing summary.json')}",
                    flush=True,
                )

            if (
                args.request_delay > 0
                and index < len(perception_tasks)
            ):
                time.sleep(args.request_delay)

        # A newly repaired Perception immediately unlocks its exact failed
        # split/algorithm tasks in the same retry round.
        reason_tasks, perception_still_missing = _collect_tasks(args)
        reason_completed = 0
        for index, task in enumerate(reason_tasks, start=1):
            attempts[task.key] = attempts.get(task.key, 0) + 1
            print(
                f"[reason retry {index}/{len(reason_tasks)}] {task.key} "
                f"attempt={attempts[task.key]}",
                flush=True,
            )
            # Force-retry a timeout instead of letting the normal runner treat
            # its marker as cached.  A new timeout will recreate this file.
            timeout_path = task.split_root / "timeout.json"
            if timeout_path.exists():
                timeout_path.unlink()
            try:
                run_reason(
                    task.input_scene,
                    task.result_scene,
                    task.shared_perception,
                    task.annotation,
                    task.model,
                    task.algorithm,
                    SimpleNamespace(force_reason=True),
                )
            except Exception as exc:
                last_errors[task.key] = str(exc)

            if _is_complete(task):
                reason_completed += 1
                last_errors.pop(task.key, None)
                print(f"[reason retry success] {task.key}", flush=True)
            else:
                print(
                    f"[reason retry still failed] {task.key}: "
                    f"{last_errors.get(task.key, 'invalid output')}",
                    flush=True,
                )

            if args.request_delay > 0 and index < len(reason_tasks):
                time.sleep(args.request_delay)

        remaining_reason, remaining_perception = _collect_tasks(args)
        payload = {
            "status": (
                "running"
                if remaining_reason or remaining_perception
                else "complete"
            ),
            "round": round_number,
            "perception_attempted_this_round": len(perception_tasks),
            "perception_completed_this_round": perception_completed,
            "reason_attempted_this_round": len(reason_tasks),
            "reason_completed_this_round": reason_completed,
            "perception_still_missing_before_reason": [
                _perception_record(task)
                for task in perception_still_missing
            ],
            "remaining_reason": [
                _failure_detail(task) for task in remaining_reason
            ],
            "remaining_perception": [
                _perception_record(task) for task in remaining_perception
            ],
            "attempts_per_task": attempts,
            "last_errors": last_errors,
        }
        _write_json(status_path, payload)

        remaining_count = len(remaining_reason) + len(remaining_perception)
        if remaining_count and args.round_delay > 0:
            print(
                f"[retry wait] {len(remaining_perception)} Perception and "
                f"{len(remaining_reason)} Reason task(s) remain; "
                f"sleeping {args.round_delay:.1f}s",
                flush=True,
            )
            time.sleep(args.round_delay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retry missing/failed Perception once per scene-query and retry "
            "only technically failed Reason tasks, preserving every valid "
            "output and RSR=0-but-completed prediction."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "input",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "output",
    )
    parser.add_argument(
        "--testcase",
        action="append",
        default=None,
        help="Repeatable testcase slug; defaults to the four GPT-4o groups.",
    )
    parser.add_argument("--scene-id", type=int, action="append", default=None)
    parser.add_argument(
        "--query-obj-id",
        type=int,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--case-directory",
        action="append",
        default=None,
        help="Exact scene_<scene>[_query_<query>] directory; repeatable.",
    )
    parser.add_argument(
        "--split",
        type=int,
        action="append",
        choices=(0, 1, 2),
        default=None,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Repeatable; defaults to gpt-4o.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=sorted(ALGORITHM_BY_SLUG),
        default=None,
        help="Repeatable; defaults to the four comparison algorithms.",
    )
    parser.add_argument(
        "--perception-review-model",
        default="gpt-4o",
    )
    parser.add_argument(
        "--perception-review-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"),
    )
    parser.add_argument(
        "--perception-review-timeout",
        type=float,
        default=300.0,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="0 means keep retrying until every selected task succeeds.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=2.0,
        help="Seconds between Reason requests in one retry round.",
    )
    parser.add_argument(
        "--round-delay",
        type=float,
        default=20.0,
        help="Seconds between retry rounds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected failures without sending API requests.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "After retries, refresh all four algorithms across all four "
            "GPT-4o categories."
        ),
    )
    args = parser.parse_args()
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "OPENAI_API_KEY is not exported; refusing to enter a retry loop"
        )
    args.model = args.model or ["gpt-4o"]
    args.algorithm = args.algorithm or list(DEFAULT_ALGORITHMS)
    raise SystemExit(retry(args))


if __name__ == "__main__":
    main()
