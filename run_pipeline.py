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
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

SMARTGRASP_ROOT = Path(__file__).resolve().parent
DATA_REALWORLD_ROOT = SMARTGRASP_ROOT / "data_realworld"
LOGS_DIR = SMARTGRASP_ROOT / "logs"

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
    reason_script = str(SMARTGRASP_ROOT / "reason" / "run_reason.py")
    cmd = [
        sys.executable, "-u",
        reason_script,
        "--root", str(DATA_REALWORLD_ROOT),
        "--scene-root", str(DATA_REALWORLD_ROOT),
        "--scene-id", scene_id,
        "--target-source", "intent",
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
        return None
    row = df.iloc[0].to_dict()
    return row


def get_part_mask_path(scene_dir: Path, reason_result: dict[str, Any] | None) -> Path | None:
    """Determine SAM2 part mask path from reason output."""
    if reason_result is None:
        return None

    summary_path = scene_dir / "perception" / "summary.json"
    if not summary_path.exists():
        _log(f"  WARNING: perception summary not found")
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    part_id = reason_result.get("selected_object_graspability_part_id")
    if part_id is not None:
        part_file = f"mask_sam2/part_{int(part_id):03d}.png"
        part_mask_path = scene_dir / "perception" / part_file
        if part_mask_path.exists():
            _log(f"  Using SAM2 binary mask: {part_mask_path}")
            return part_mask_path
        part_file = f"sam2_rgb_parts/part_{int(part_id):03d}.png"
        part_mask_path = scene_dir / "perception" / part_file
        if part_mask_path.exists():
            _log(f"  Using SAM2 cropped part image: {part_mask_path}")
            return part_mask_path

    selected_id = reason_result.get("selected_object_id")
    if selected_id is not None:
        object_id_to_part_files = summary.get("object_id_to_sam2_part_files", {})
        part_files = object_id_to_part_files.get(str(int(selected_id)), [])
        if part_files:
            first_part = part_files[0]
            part_mask_path = scene_dir / "perception" / first_part
            if part_mask_path.exists():
                _log(f"  Using first SAM2 part mask for object {selected_id}: {part_mask_path}")
                return part_mask_path

    mask_dir = scene_dir / "perception" / "mask"
    if mask_dir.exists():
        anchor_masks = sorted(mask_dir.glob("*_anchor_*.png"))
        if anchor_masks:
            _log(f"  Using anchor mask: {anchor_masks[0]}")
            return anchor_masks[0]

    _log(f"  WARNING: no SAM2 part mask found for selected object")
    return None


def run_grasp(scene_dir: Path, perception_mask: Path | None, args: argparse.Namespace) -> bool:
    """Run GraspNet + JAKA execution on the scene."""
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
    ]
    if perception_mask is not None:
        cmd.extend(["--perception-mask", str(perception_mask)])
        cmd.extend(["--use-sam-mask"])  # enable mask-based cropping
    else:
        cmd.extend(["--no-use-sam-mask"])  # use full point cloud, no interactive SAM

    if args.top_k:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.candidate_index is not None:
        cmd.extend(["--candidate-index", str(args.candidate_index)])

    result = _run(cmd, f"Grasp on {scene_id}")
    return result.returncode == 0


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
    parser.add_argument("--scene-dir", default=None, help="Use existing scene dir instead of new capture")
    args = parser.parse_args()

    # --- init logging ---
    log_path = _init_logging()
    _log(f"Instruction: {args.instruction}")
    _log(f"Calibration: {args.calibration_mode}")
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

    # === Step 5: Grasp ===
    if not run_grasp(scene_dir, perception_mask, args):
        _log("ERROR: grasp failed")
        sys.exit(1)

    _log(f"[pipeline] ALL DONE. scene={scene_dir}")
    _log(f"[pipeline] log saved to {_log_path}")


if __name__ == "__main__":
    main()
