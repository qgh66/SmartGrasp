#!/usr/bin/env python
"""Print the current JAKA TCP/gripper pose in the robot base frame."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
JAKA_WORKER = WORKSPACE_ROOT / "scripts" / "jaka_motion_worker.py"
DEFAULT_JAKA_IP = "192.168.1.199"
DEFAULT_JAKA_PYTHON = "/home/admin128/anaconda3/envs/smartgrasp310/bin/python"
DEFAULT_JKRC_DIR = WORKSPACE_ROOT / "jkrc"


def read_tcp_pose(args: argparse.Namespace) -> list[float]:
    command = [
        str(Path(args.jaka_python).expanduser()),
        str(JAKA_WORKER),
        "--print-tcp-pose",
        "--json-only",
        "--jaka-ip",
        args.jaka_ip,
        "--jkrc-dir",
        str(Path(args.jkrc_dir).expanduser()),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        tcp_pose = payload.get("tcp_pose")
        if isinstance(tcp_pose, list) and len(tcp_pose) == 6:
            return [float(value) for value in tcp_pose]
    raise RuntimeError(f"Could not parse TCP pose from worker output: {result.stdout!r}")


def format_tcp_pose(pose: list[float]) -> str:
    return (
        f"x={pose[0]:.3f} mm\n"
        f"y={pose[1]:.3f} mm\n"
        f"z={pose[2]:.3f} mm\n"
        f"rx={pose[3]:.6f} rad\n"
        f"ry={pose[4]:.6f} rad\n"
        f"rz={pose[5]:.6f} rad"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print current JAKA TCP/gripper pose in robot base frame.")
    parser.add_argument("--jaka-ip", default=DEFAULT_JAKA_IP, help="JAKA controller IP.")
    parser.add_argument("--jaka-python", default=DEFAULT_JAKA_PYTHON, help="Python executable that can import jkrc.")
    parser.add_argument("--jkrc-dir", default=str(DEFAULT_JKRC_DIR), help="Directory containing jkrc.so and libjakaAPI.so.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    tcp_pose = read_tcp_pose(args)
    if args.json:
        print(json.dumps({"frame": "jaka_base", "unit": "mm_rad", "tcp_pose": tcp_pose}, indent=2))
        return
    print("[tcp-base] current JAKA TCP/gripper pose")
    print(format_tcp_pose(tcp_pose))


if __name__ == "__main__":
    main()
