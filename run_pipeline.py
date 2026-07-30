#!/usr/bin/env python
"""
SmartGrasp full pipeline: capture → perception → intent → reason → grasp.

Usage:
    python run_pipeline.py --instruction "grasp the leftmost apple"

This script orchestrates the end-to-end flow:
  1. Capture RGB-D from RealSense → saved to data_realworld/<timestamp>/
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
from pathlib import Path
from typing import Any, TextIO

SMARTGRASP_ROOT = Path(__file__).resolve().parent
DATA_REALWORLD_ROOT = SMARTGRASP_ROOT / "data_realworld"
LOGS_DIR = SMARTGRASP_ROOT / "logs"

# Edit SAM2 settings here. These values are used by both the full Perception
# pipeline and ``--debug sam2``. A ``None`` depth value inherits the matching
# RGB SAM2 value inside perception/run_perception.py.
SAM2_PARAMETERS: dict[str, int | float | None] = {
    "--mask-clean-kernel": 3,
    "--proposal-min-area-ratio": 0.006,
    "--proposal-max-area-ratio": 0.11,
    "--proposal-border-fraction-threshold": 0.18,
    "--sam2-points-per-side": 24,
    "--sam2-crop-n-layers": 0,
    "--sam2-pred-iou-thresh": 0.68,
    "--sam2-stability-score-thresh": 0.83,
    "--depth-sam2-points-per-side": 24,
    "--depth-sam2-crop-n-layers": 1,
    "--depth-sam2-pred-iou-thresh": 0.58,
    "--depth-sam2-stability-score-thresh": 0.73,
}
SAVE_SAM2_CANDIDATES = True

# Global log file handle – opened once at pipeline start and closed at exit.
_log_file: TextIO | None = None
_log_path: Path | None = None


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


def _init_logging() -> Path:
    """Create logs/ directory and open the pipeline log file."""
    global _log_file, _log_path
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _log_path = LOGS_DIR / f"realworld_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_file = open(str(_log_path), "w", encoding="utf-8")
    _log(f"{'='*60}")
    _log(f"SmartGrasp Pipeline Log — {datetime.now().isoformat()}")
    _log(f"{'='*60}")
    return _log_path


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
    scene_id = scene_dir.name  # timestamp string
    cmd = [
        sys.executable, "-u",
        str(SMARTGRASP_ROOT / "perception" / "run_perception.py"),
        "--scene-id", scene_id,
        "--mode", "vlm",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.debug:
        cmd.extend(["--debug", args.debug])
    for option, value in SAM2_PARAMETERS.items():
        if value is not None:
            cmd.extend([option, str(value)])
    if SAVE_SAM2_CANDIDATES:
        cmd.append("--save-candidates")
    result = _run(cmd, f"Perception on {scene_id}")
    return result.returncode == 0


def run_intent(scene_dir: Path, instruction: str, args: argparse.Namespace) -> bool:
    """Run intent resolution on the scene."""
    scene_id = scene_dir.name
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
    scene_id = scene_dir.name
    intent_id_path = scene_dir / "intent" / "id.txt"
    if not intent_id_path.is_file():
        _log(f"  ERROR: intent result not found at {intent_id_path}")
        return None
    intent_id_text = intent_id_path.read_text(encoding="utf-8").strip()
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
        "--target-source", "id",
        "--target-id", str(intent_id),
        "--prior-prompt", "graspability",
        "--ranking-score", "ig_graspability",
        "--closed-loop",
    ]
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
    scene_id = scene_dir.name
    input_dir = scene_dir / "input"
    if not input_dir.exists():
        input_dir = scene_dir  # legacy layout fallback
    cmd = [
        "bash",
        str(SMARTGRASP_ROOT / "run_realworld_grasp.sh"),
        "--output-dir", str(input_dir),
        "--calibration-mode", args.calibration_mode,
        "--reuse-capture",
        "--execute",
        "--perception-mask", str(perception_mask),
        "--use-sam-mask",
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
    parser = argparse.ArgumentParser(description="SmartGrasp full pipeline")
    parser.add_argument("--instruction", required=True, help="Natural language grasp instruction")
    parser.add_argument("--model", default=None, help="VLM model name for intent/reason")
    parser.add_argument("--device", default=None, help="Device for perception (cuda/cpu)")
    parser.add_argument("--top-k", type=int, default=50, help="Top K grasp candidates")
    parser.add_argument("--candidate-index", type=int, default=0, help="Grasp candidate index to execute")
    parser.add_argument("--calibration-mode", default="hand_eye", choices=["legacy_plate", "hand_eye"],
                        help="Calibration mode for robot-camera transform")
    parser.add_argument("--capture-only", action="store_true", help="Only capture RGB-D, skip perception+grasp")
    parser.add_argument("--perception-only", action="store_true", help="Stop after perception")
    parser.add_argument("--reason-only", action="store_true", help="Stop after reason")
    parser.add_argument("--no-grasp", action="store_true", help="Skip grasp execution")
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

    # --- init logging ---
    log_path = _init_logging()
    _log(f"Instruction: {args.instruction}")
    _log(f"Calibration: {args.calibration_mode}")
    _log(f"Debug: {args.debug or 'disabled'}")
    _log(f"Log file: {log_path}")

    try:
        _main_impl(args)
    except KeyboardInterrupt:
        _log("\n[pipeline] interrupted by user")
        sys.exit(130)
    except Exception as exc:
        _log(f"\n[pipeline] FATAL: {exc}")
        raise
    finally:
        _close_logging()


def _main_impl(args: argparse.Namespace) -> None:
    # === Step 0: Capture RGB-D (if not reusing scene) ===
    if args.scene_dir:
        scene_dir = Path(args.scene_dir).expanduser().resolve()
        if not scene_dir.exists():
            _log(f"ERROR: scene directory not found: {scene_dir}")
            sys.exit(1)
        _log(f"[pipeline] using existing scene: {scene_dir}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scene_dir = DATA_REALWORLD_ROOT / timestamp
        input_dir = scene_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        _log(f"[pipeline] capturing to {input_dir}")

        capture_cmd = [
            "bash",
            str(SMARTGRASP_ROOT / "run_realworld_grasp.sh"),
            "--output-dir", str(input_dir),
            "--instruction", args.instruction,
            "--calibration-mode", args.calibration_mode,
            "--no-trial-log",
            "--no-data-realworld",
            "--capture-only",
            "--num-cycles", "1",
        ]
        result = _run(capture_cmd, "Step 1: Capture RGB-D")
        if result.returncode != 0:
            _log("ERROR: capture failed")
            sys.exit(1)

    if args.capture_only:
        _log(f"[pipeline] capture done. scene at {scene_dir}")
        return

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
        return

    if args.perception_only:
        _log(f"[pipeline] perception done. outputs in {scene_dir / 'perception'}")
        return

    # === Step 2: Intent ===
    if not run_intent(scene_dir, args.instruction, args):
        _log("ERROR: intent failed")
        sys.exit(1)

    # === Step 3: Reason ===
    reason_result = run_reason(scene_dir, args)
    if reason_result is None:
        _log("ERROR: reason failed")
        sys.exit(1)

    _log(f"[pipeline] reason selected object_id={reason_result.get('selected_object_id')} "
         f"part_id={reason_result.get('selected_object_graspability_part_id')}")

    if args.reason_only:
        _log(f"[pipeline] reason done.")
        return

    # === Step 4: Find part mask ===
    perception_mask = get_part_mask_path(scene_dir, reason_result)

    if args.no_grasp:
        _log(f"[pipeline] skipping grasp. perception_mask={perception_mask}")
        return

    if perception_mask is None:
        _log("ERROR: cannot run grasp without the binary SAM2 part mask selected by Reason")
        sys.exit(1)

    # === Step 5: Grasp ===
    if not run_grasp(scene_dir, perception_mask, args):
        _log("ERROR: grasp failed")
        sys.exit(1)

    _log(f"[pipeline] ALL DONE. scene={scene_dir}")
    _log(f"[pipeline] log saved to {_log_path}")


if __name__ == "__main__":
    main()
