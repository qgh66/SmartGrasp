#!/usr/bin/env python
"""
SmartGrasp full pipeline: capture → perception → intent → reason → grasp.

Usage:
    python run_pipeline.py --instruction "grasp the leftmost apple"

This script orchestrates the end-to-end flow:
  1. Capture RGB-D from RealSense → saved to data_realworld/<start_timestamp>/<round>/
  2. Run perception (Molmo + SAM2) → object masks, occlusion graph
  3. Run intent (VLM) → select target object from instruction
  4. Run reason (VLM) → decide best object/part to grasp
  5. Look up SAM2 part mask → run GraspNet + JAKA execution
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

SMARTGRASP_ROOT = Path(__file__).resolve().parent
DATA_REALWORLD_ROOT = SMARTGRASP_ROOT / "data_realworld"
LOGS_DIR = SMARTGRASP_ROOT / "logs"

# Default RealSense serial number suffix for new captures. It can also be
# overridden per run with ``--camera-serial``.
DEFAULT_CAMERA_SERIAL_SUFFIX = "72659"
# Stop repeated physical actions if perception never converges on the target.
MAX_GRASP_ROUNDS = 10
DEFAULT_GRASP_EXTRA_DEPTH_MM = 0.0

# Occlusion contact detection only: 9x9 expands each object mask by 4 px per
# side, twice the previous 5x5 kernel's 2 px expansion.
OCCLUSION_DILATION_KERNEL_SIZE = 9

# Edit SAM2 settings here. These values are used by both the full Perception
# pipeline and ``--debug sam2``. A ``None`` depth value inherits the matching
# RGB SAM2 value inside perception/run_perception.py.
SAM2_PARAMETERS: dict[str, int | float | None] = {
    "--mask-clean-kernel": 3,
    "--proposal-min-area-ratio": 0.006,
    "--proposal-max-area-ratio": 0.11,
    "--proposal-border-fraction-threshold": 0.18,
    "--sam2-points-per-side": 24,
    "--sam2-crop-n-layers": 1,
    "--sam2-pred-iou-thresh": 0.65,
    "--sam2-stability-score-thresh": 0.78,
    "--depth-sam2-points-per-side": 24,
    "--depth-sam2-crop-n-layers": 1,
    "--depth-sam2-pred-iou-thresh": 0.58,
    "--depth-sam2-stability-score-thresh": 0.73,
}
SAVE_SAM2_CANDIDATES = True

# Global log file handle – opened once at pipeline start and closed at exit.
_log_file: TextIO | None = None
_log_path: Path | None = None


class RoundOutcome(str, Enum):
    """How the outer pipeline should proceed after one round."""

    STAGE_COMPLETE = "stage_complete"
    FINAL_TARGET_GRASPED = "final_target_grasped"
    NON_TARGET_GRASPED = "non_target_grasped"
    GRASP_FAILED = "grasp_failed"


def _log(msg: str = "", *, flush: bool = True) -> None:
    """Write to both stdout and the global log file."""
    if _log_file is not None:
        _log_file.write(msg + "\n")
        if flush:
            _log_file.flush()
    print(msg, flush=flush)


def _run(cmd: list[str], desc: str = "", env=None) -> subprocess.CompletedProcess:
    """Run a command, tee its output to the global log file and stdout."""
    header = f"===== {desc} =====" if desc else ""
    if header:
        _log(f"\n{header}")
    _log(f"  CMD: {' '.join(cmd)}")
    
    started = time.time()
    if _log_file is not None:
        result = subprocess.run(
            cmd, cwd=str(SMARTGRASP_ROOT), env=env,
            stdout=_log_file, stderr=subprocess.STDOUT,
        )
    else:
        result = subprocess.run(cmd, cwd=str(SMARTGRASP_ROOT), env=env)
    
    elapsed = time.time() - started
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    _log(f"  [{elapsed:.1f}s] {status}")
    return result


def find_latest_scene() -> Path | None:
    """Return the most recently created data_realworld scene directory."""
    if not DATA_REALWORLD_ROOT.exists():
        return None
    dirs = sorted(
        [d for d in DATA_REALWORLD_ROOT.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs[0] if dirs else None


def _init_logging(log_path: Path) -> Path:
    """Open the pipeline log file at the requested session location."""
    global _log_file, _log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_path = log_path
    _log_file = open(str(_log_path), "w", encoding="utf-8")
    _log(f"{'='*60}")
    _log(f"SmartGrasp Pipeline Log — {datetime.now().isoformat()}")
    _log(f"{'='*60}")
    return _log_path


def _scene_key(scene_dir: Path) -> str:
    """Return the data_realworld-relative scene id used by every stage."""
    try:
        return scene_dir.resolve().relative_to(DATA_REALWORLD_ROOT.resolve()).as_posix()
    except ValueError:
        return scene_dir.name


def _close_logging() -> None:
    """Flush and close the global log file."""
    global _log_file
    if _log_file is not None:
        _log(f"\n{'='*60}")
        _log(f"Pipeline finished — {datetime.now().isoformat()}")
        _log(f"{'='*60}")
        _log_file.close()
        _log_file = None


def run_perception(scene_dir: Path, args: argparse.Namespace) -> bool:
    """Run perception pipeline on the given scene directory."""
    scene_id = _scene_key(scene_dir)
    cmd = [
        sys.executable, "-u",
        str(SMARTGRASP_ROOT / "perception" / "run_perception.py"),
        "--scene-id", scene_id,
        "--mode", "vlm",
        "--kernel-size", str(OCCLUSION_DILATION_KERNEL_SIZE),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.debug:
        cmd.extend(["--debug", args.debug])
    if getattr(args, "disable_depth_proposals", False):
        cmd.append("--disable-depth-proposals")
    for option, value in SAM2_PARAMETERS.items():
        if value is not None:
            cmd.extend([option, str(value)])
    if SAVE_SAM2_CANDIDATES:
        cmd.append("--save-candidates")
    result = _run(cmd, f"Perception on {scene_id}")
    return result.returncode == 0


def run_intent(scene_dir: Path, instruction: str, args: argparse.Namespace) -> bool:
    """Run intent resolution on the scene."""
    scene_id = _scene_key(scene_dir)
    cmd = [
        sys.executable, "-u",
        str(SMARTGRASP_ROOT / "intent" / "run_intent.py"),
        "--scene-id", scene_id,
        "--instruction", instruction,
        "--mode", "data",
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    result = _run(cmd, f"Intent on {scene_id}")
    return result.returncode == 0


def run_reason(scene_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    """Run reason on the scene, return the decision."""
    scene_id = _scene_key(scene_dir)
    intent_id_path = scene_dir / "intent" / "id.txt"
    if not intent_id_path.is_file():
        _log(f"  ERROR: intent result not found at {intent_id_path}")
        return None
    intent_id_text = intent_id_path.read_text(encoding="utf-8").strip()
    intent_id: int | None
    if intent_id_text.lower() in {"none", "null"}:
        intent_id = None
    else:
        try:
            intent_id = int(intent_id_text)
        except ValueError:
            _log(f"  ERROR: invalid intent object id in {intent_id_path}: {intent_id_text!r}")
            return None

    reason_script = str(SMARTGRASP_ROOT / "reason" / "run_reason.py")
    cmd = [
        sys.executable, "-u",
        reason_script,
        "--root", str(DATA_REALWORLD_ROOT),
        "--scene-root", str(DATA_REALWORLD_ROOT),
        "--scene-id", scene_id,
        "--out-root", str(scene_dir / "reason_runs"),
        "--prior-prompt", "graspability",
        "--ranking-score", "ig_graspability",
        "--closed-loop",
    ]
    if intent_id is None:
        # A fully hidden target has no current perception object id. Preserve
        # the first Intent result instead of making a second, potentially
        # inconsistent VLM call inside Reason.
        cmd.extend([
            "--target-source", "missing",
            "--instruction", args.instruction,
        ])
    else:
        cmd.extend([
            "--target-source", "id",
            "--target-id", str(intent_id),
        ])
    if args.model:
        cmd.extend(["--model", args.model])

    result = _run(cmd, f"Reason on {scene_id}")
    if result.returncode != 0:
        return None

    reason_results = scene_dir / "reason" / "results.csv"
    if not reason_results.exists():
        _log(f"  WARNING: reason output not found at {reason_results}")
        return None

    import pandas as pd
    df = pd.read_csv(reason_results)
    if df.empty:
        _log(f"  ERROR: reason output is empty: {reason_results}")
        return None
    row = df.iloc[0].to_dict()
    status = str(row.get("status") or "").strip().lower()
    if status != "ok":
        _log(f"  ERROR: reason returned status={row.get('status')!r}")
        return None
    selected_object_id = row.get("selected_object_id")
    if selected_object_id is None or pd.isna(selected_object_id):
        _log("  ERROR: reason did not select an object")
        return None
    return row


def _parse_reason_object_id(reason_result: dict[str, Any], field: str) -> int:
    """Read an integer object id from a pandas-derived Reason result row."""
    raw_value = reason_result.get(field)
    try:
        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError
        return int(numeric_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Reason returned an invalid {field}: {raw_value!r}") from exc


def _parse_optional_reason_object_id(
    reason_result: dict[str, Any],
    field: str,
) -> int | None:
    """Read a nullable integer object id from a pandas-derived Reason row."""
    raw_value = reason_result.get(field)
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip().lower() in {
        "", "none", "null", "nan",
    }:
        return None
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Reason returned an invalid {field}: {raw_value!r}") from exc
    if math.isnan(numeric_value):
        return None
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"Reason returned an invalid {field}: {raw_value!r}")
    return int(numeric_value)


def _write_round_summary(
    round_dir: Path,
    round_index: int,
    reason_result: dict[str, Any],
    *,
    grasp_attempted: bool,
    grasp_succeeded: bool,
) -> Path:
    """Persist the target-vs-selected decision and physical grasp status."""
    target_object_id = _parse_optional_reason_object_id(reason_result, "target_id")
    selected_object_id = _parse_reason_object_id(reason_result, "selected_object_id")
    summary = {
        "round": round_index,
        "scene_id": _scene_key(round_dir),
        "target_object_id": target_object_id,
        "selected_object_id": selected_object_id,
        "selected_object_is_final_target": (
            target_object_id is not None and selected_object_id == target_object_id
        ),
        "branch": reason_result.get("branch"),
        "selected_part_id": _parse_optional_reason_object_id(
            reason_result,
            "selected_object_graspability_part_id",
        ),
        "grasp_attempted": grasp_attempted,
        "grasp_succeeded": grasp_succeeded,
    }
    summary_path = round_dir / "pipeline_round.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def get_part_mask_path(scene_dir: Path, reason_result: dict[str, Any] | None) -> Path | None:
    """Return the binary SAM2 part mask selected by Reason."""
    if reason_result is None:
        return None
    part_id = reason_result.get("selected_object_graspability_part_id")
    try:
        part_id_value = float(part_id)
        if not math.isfinite(part_id_value) or not part_id_value.is_integer():
            raise ValueError
        part_id_int = int(part_id_value)
    except (TypeError, ValueError, OverflowError):
        _log("  ERROR: Reason did not return a valid SAM2 part id")
        return None

    part_mask_path = scene_dir / "perception" / "mask_sam2" / f"part_{part_id_int:03d}.png"
    if not part_mask_path.is_file():
        _log(f"  ERROR: Reason-selected binary SAM2 mask not found: {part_mask_path}")
        return None
    _log(f"  Using Reason-selected SAM2 binary mask: {part_mask_path}")
    return part_mask_path


def run_grasp(scene_dir: Path, perception_mask: Path, args: argparse.Namespace) -> bool:
    """Run GraspNet + JAKA execution on the scene."""
    scene_id = _scene_key(scene_dir)
    input_dir = scene_dir / "input"
    if not input_dir.exists():
        input_dir = scene_dir  # legacy layout fallback
    cmd = [
        "bash",
        str(SMARTGRASP_ROOT / "run_realworld_grasp.sh"),
        "--output-dir", str(input_dir),
        "--camera-serial", args.camera_serial,
        "--calibration-mode", args.calibration_mode,
        "--reuse-capture",
        "--execute",
        "--perception-mask", str(perception_mask),
        "--use-sam-mask",
        "--no-trial-log",
        "--grasp-extra-depth-mm", str(args.grasp_extra_depth_mm),
    ]

    if args.top_k:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.candidate_index is not None:
        cmd.extend(["--candidate-index", str(args.candidate_index)])

    candidates_path = input_dir / "grasp_candidates.json"
    for attempt in (1, 2):
        attempt_started = time.time()
        result = _run(cmd, f"Grasp on {scene_id} (attempt {attempt}/2)")
        if result.returncode == 0:
            return True
        if attempt == 2 or not _failed_before_execution_with_no_candidates(
            candidates_path,
            attempt_started,
        ):
            return False
        _log(
            "  GraspNet produced no candidate after safety filters; "
            "retrying inference once without changing the capture or mask."
        )
    return False


def _failed_before_execution_with_no_candidates(
    candidates_path: Path,
    attempt_started: float,
) -> bool:
    """Allow a retry only when this attempt wrote zero filtered candidates."""
    try:
        if candidates_path.stat().st_mtime < attempt_started - 1.0:
            return False
        payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("num_candidates_after_filter") == 0


def main():
    program_started_at = datetime.now()
    run_timestamp = program_started_at.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="SmartGrasp full pipeline")
    parser.add_argument("--instruction", required=True, help="Natural language grasp instruction")
    parser.add_argument("--model", default=None, help="VLM model name for intent/reason")
    parser.add_argument("--device", default=None, help="Device for perception (cuda/cpu)")
    parser.add_argument(
        "--camera-serial",
        default=DEFAULT_CAMERA_SERIAL_SUFFIX,
        help=(
            "RealSense full serial number or unique suffix used for capture "
            f"(default: {DEFAULT_CAMERA_SERIAL_SUFFIX})"
        ),
    )
    parser.add_argument("--top-k", type=int, default=100, help="Top K grasp candidates")
    parser.add_argument("--candidate-index", type=int, default=0, help="Grasp candidate index to execute")
    parser.add_argument(
        "--grasp-extra-depth-mm",
        type=float,
        default=DEFAULT_GRASP_EXTRA_DEPTH_MM,
        help=(
            "Signed final grasp offset along TCP local Z in millimeters: positive moves along +Z, "
            f"negative along -Z (default: {DEFAULT_GRASP_EXTRA_DEPTH_MM:g})"
        ),
    )
    parser.add_argument("--calibration-mode", default="hand_eye", choices=["legacy_plate", "hand_eye"],
                        help="Calibration mode for robot-camera transform")
    parser.add_argument("--capture-only", action="store_true", help="Only capture RGB-D, skip perception+grasp")
    parser.add_argument("--perception-only", action="store_true", help="Stop after perception")
    parser.add_argument("--reason-only", action="store_true", help="Stop after reason")
    parser.add_argument("--no-grasp", action="store_true", help="Skip grasp execution")
    parser.add_argument(
        "--disable-depth-proposals",
        "--disable-depth-sam2-proposals",
        dest="disable_depth_proposals",
        action="store_true",
        help=(
            "Disable proposals generated by running SAM2 on the depth image; "
            "depth remains enabled for background filtering and geometry"
        ),
    )
    parser.add_argument(
        "--debug",
        choices=["sam2"],
        default=None,
        help=(
            "sam2: keep the normal RGB-D capture, run Perception through SAM2 "
            "candidate generation, and stop before VLM review/assembly"
        ),
    )
    parser.add_argument("--scene-dir", default=None, help="Use existing scene dir instead of new capture")
    args = parser.parse_args()

    if args.scene_dir:
        session_dir = None
        log_path = _init_logging(LOGS_DIR / f"realworld_{run_timestamp}.log")
    else:
        session_dir = DATA_REALWORLD_ROOT / run_timestamp
        session_dir.mkdir(parents=True, exist_ok=False)
        log_path = _init_logging(session_dir / "pipeline.log")

    _log(f"Instruction: {args.instruction}")
    _log(f"Camera serial/suffix: {args.camera_serial}")
    _log(f"Calibration: {args.calibration_mode}")
    _log(f"Debug: {args.debug or 'disabled'}")
    _log(f"Depth SAM2 proposals: {'disabled' if args.disable_depth_proposals else 'enabled'}")
    _log(f"Session: {session_dir or 'existing scene'}")
    _log(f"Log file: {log_path}")

    try:
        _main_impl(args, session_dir)
    except KeyboardInterrupt:
        _log("\n[pipeline] interrupted by user")
        sys.exit(130)
    except Exception as exc:
        _log(f"\n[pipeline] FATAL: {exc}")
        raise
    finally:
        _close_logging()


def _run_pipeline_round(
    args: argparse.Namespace,
    scene_dir: Path,
    round_index: int,
    *,
    capture_new_scene: bool,
) -> RoundOutcome:
    """Run one physical observation/action round.

    Grasp execution failures are returned to the outer loop so the next round
    can recapture and recompute the complete pipeline instead of terminating.
    """
    scene_id = _scene_key(scene_dir)
    _log(f"\n[pipeline] ===== round {round_index} ({scene_id}) =====")

    if capture_new_scene:
        input_dir = scene_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        _log(f"[pipeline] capturing to {input_dir}")

        capture_cmd = [
            "bash",
            str(SMARTGRASP_ROOT / "run_realworld_grasp.sh"),
            "--output-dir", str(input_dir),
            "--instruction", args.instruction,
            "--camera-serial", args.camera_serial,
            "--calibration-mode", args.calibration_mode,
            "--no-trial-log",
            "--no-data-realworld",
            "--capture-only",
            "--num-cycles", "1",
        ]
        result = _run(capture_cmd, f"Round {round_index} Step 1: Capture RGB-D")
        if result.returncode != 0:
            _log("ERROR: capture failed")
            sys.exit(1)
    else:
        _log(f"[pipeline] using existing scene: {scene_dir}")

    if args.capture_only:
        _log(f"[pipeline] capture done. scene at {scene_dir}")
        return RoundOutcome.STAGE_COMPLETE

    # === Step 1: Perception ===
    if not run_perception(scene_dir, args):
        _log("ERROR: perception failed")
        sys.exit(1)

    if args.debug == "sam2":
        debug_output = scene_dir / "perception" / "debug_sam2.json"
        if not debug_output.is_file():
            _log(f"ERROR: SAM2 debug output not found: {debug_output}")
            sys.exit(1)
        _log(
            "[pipeline] SAM2 debug done before VLM review/assembly. "
            f"outputs in {scene_dir / 'perception'}"
        )
        return RoundOutcome.STAGE_COMPLETE

    if args.perception_only:
        _log(f"[pipeline] perception done. outputs in {scene_dir / 'perception'}")
        return RoundOutcome.STAGE_COMPLETE

    # === Step 2: Intent ===
    if not run_intent(scene_dir, args.instruction, args):
        _log("ERROR: intent failed")
        sys.exit(1)

    # === Step 3: Reason ===
    reason_result = run_reason(scene_dir, args)
    if reason_result is None:
        _log("ERROR: reason failed")
        sys.exit(1)

    try:
        target_object_id = _parse_optional_reason_object_id(reason_result, "target_id")
        selected_object_id = _parse_reason_object_id(reason_result, "selected_object_id")
    except ValueError as exc:
        _log(f"ERROR: {exc}")
        sys.exit(1)
    selected_final_target = (
        target_object_id is not None and selected_object_id == target_object_id
    )
    _log(
        f"[pipeline] reason target_id={target_object_id} "
        f"selected_object_id={selected_object_id} "
        f"part_id={reason_result.get('selected_object_graspability_part_id')} "
        f"final_target={selected_final_target}"
    )
    round_summary_path = _write_round_summary(
        scene_dir,
        round_index,
        reason_result,
        grasp_attempted=False,
        grasp_succeeded=False,
    )
    _log(f"[pipeline] round decision saved to {round_summary_path}")

    if args.reason_only:
        _log("[pipeline] reason done.")
        return RoundOutcome.STAGE_COMPLETE

    # === Step 4: Find part mask ===
    perception_mask = get_part_mask_path(scene_dir, reason_result)

    if args.no_grasp:
        _log(f"[pipeline] skipping grasp. perception_mask={perception_mask}")
        return RoundOutcome.STAGE_COMPLETE

    if perception_mask is None:
        _log("ERROR: cannot run grasp without the binary SAM2 part mask selected by Reason")
        sys.exit(1)

    # === Step 5: Grasp ===
    if not run_grasp(scene_dir, perception_mask, args):
        _write_round_summary(
            scene_dir,
            round_index,
            reason_result,
            grasp_attempted=True,
            grasp_succeeded=False,
        )
        _log(
            f"[pipeline] grasp failed in round {round_index}; "
            "the next round will recapture and rerun the full pipeline."
        )
        return RoundOutcome.GRASP_FAILED

    _write_round_summary(
        scene_dir,
        round_index,
        reason_result,
        grasp_attempted=True,
        grasp_succeeded=True,
    )
    if selected_final_target:
        return RoundOutcome.FINAL_TARGET_GRASPED
    return RoundOutcome.NON_TARGET_GRASPED


def _main_impl(args: argparse.Namespace, session_dir: Path | None) -> None:
    if args.scene_dir:
        scene_dir = Path(args.scene_dir).expanduser().resolve()
        if not scene_dir.exists():
            _log(f"ERROR: scene directory not found: {scene_dir}")
            sys.exit(1)
        outcome = _run_pipeline_round(
            args,
            scene_dir,
            1,
            capture_new_scene=False,
        )
        if outcome == RoundOutcome.NON_TARGET_GRASPED:
            _log(
                "[pipeline] existing-scene mode removed a non-target object; "
                "automatic recapture is disabled for --scene-dir."
            )
        elif outcome == RoundOutcome.GRASP_FAILED:
            _log(
                "[pipeline] grasp failed in existing-scene mode; automatic "
                "recapture is disabled for --scene-dir."
            )
        return

    if session_dir is None:
        raise RuntimeError("A new pipeline run requires a session directory")

    for round_index in range(1, MAX_GRASP_ROUNDS + 1):
        round_dir = session_dir / str(round_index)
        round_dir.mkdir(parents=True, exist_ok=False)
        outcome = _run_pipeline_round(
            args,
            round_dir,
            round_index,
            capture_new_scene=True,
        )
        if outcome == RoundOutcome.STAGE_COMPLETE:
            return
        if outcome == RoundOutcome.FINAL_TARGET_GRASPED:
            _log(
                f"[pipeline] ALL DONE: final target grasped in round {round_index}. "
                f"session={session_dir}"
            )
            _log(f"[pipeline] log saved to {_log_path}")
            return
        if round_index < MAX_GRASP_ROUNDS:
            if outcome == RoundOutcome.NON_TARGET_GRASPED:
                _log(
                    f"[pipeline] round {round_index} removed a non-target object; "
                    f"recapturing the changed scene for round {round_index + 1}."
                )
            else:
                _log(
                    f"[pipeline] round {round_index} grasp execution failed; "
                    f"restarting the full pipeline in round {round_index + 1}."
                )

    _log(
        f"ERROR: final target was not grasped within the safety limit of "
        f"{MAX_GRASP_ROUNDS} rounds"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
