#!/usr/bin/env python
"""Small JAKA/Robotiq executor intended to run in a jkrc-compatible Python.

The main real-world GraspNet pipeline can stay in the smartgrasp environment,
while this worker is launched with a separate Python 3.10 environment that can
load the controller-compatible JAKA SDK.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JKRC_DIR = WORKSPACE_ROOT / "jkrc"
VENDOR_DIR = WORKSPACE_ROOT / "vendor"


def import_jkrc_backend(jkrc_dir: Path, quiet: bool = False):
    jkrc_dir = jkrc_dir.expanduser().resolve()
    if str(jkrc_dir) not in sys.path:
        sys.path.insert(0, str(jkrc_dir))

    local_jaka_api = jkrc_dir / "libjakaAPI.so"
    if local_jaka_api.exists():
        ctypes.CDLL(str(local_jaka_api), mode=ctypes.RTLD_GLOBAL)

    try:
        import jkrc
    except Exception as exc:
        raise RuntimeError(f"Failed to import jkrc from {jkrc_dir}: {exc!r}") from exc

    if not quiet:
        print(f"[jkrc-worker] loaded from {getattr(jkrc, '__file__', 'unknown')}")
    return jkrc


def import_robotiq_backend():
    try:
        from robotiq_gripper_python import RobotiqGripper

        return "robotiq_gripper_python", RobotiqGripper
    except Exception as first_error:
        vendor_path = str(VENDOR_DIR)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        try:
            from pyrobotiqgripper import RobotiqGripper

            return "pyrobotiqgripper", RobotiqGripper
        except Exception as second_error:
            raise RuntimeError(
                "Failed to import a Robotiq backend. Tried robotiq_gripper_python "
                f"and local vendor/pyrobotiqgripper. First error: {first_error}; "
                f"second error: {second_error}"
            ) from second_error


def return_code(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, (list, tuple)) and raw:
        first = raw[0]
        if isinstance(first, int):
            return int(first)
    return None


def check_call(name: str, raw: Any, allow_none: bool = True) -> None:
    print(f"[jkrc-worker] {name} returned: {raw!r}")
    code = return_code(raw)
    if code is None:
        if allow_none:
            return
        raise RuntimeError(f"{name} returned unexpected format: {raw!r}")
    if code != 0:
        raise RuntimeError(f"{name} failed: ret={code}, raw={raw!r}")


def parse_tcp_position(raw: Any) -> list[float]:
    ret = None
    pos = None
    if isinstance(raw, (list, tuple)):
        raw_list = list(raw)
        if len(raw_list) == 2:
            ret, pos = raw_list[0], raw_list[1]
        elif len(raw_list) == 7:
            ret, pos = raw_list[0], raw_list[1:]
        elif len(raw_list) == 6:
            ret, pos = 0, raw_list
        elif len(raw_list) >= 2:
            ret, pos = raw_list[0], raw_list[1]
        elif len(raw_list) == 1:
            ret, pos = raw_list[0], None
    elif isinstance(raw, int):
        ret, pos = int(raw), None

    if ret is None or pos is None:
        raise RuntimeError(f"get_tcp_position returned unexpected format: {raw!r}")
    if int(ret) != 0:
        raise RuntimeError(f"get_tcp_position failed: ret={ret}, raw={raw!r}")
    if not isinstance(pos, (list, tuple)) or len(pos) != 6:
        raise RuntimeError(f"get_tcp_position pose must be 6D: pos={pos!r}, raw={raw!r}")
    return [float(value) for value in pos]


def format_tcp_pose(pose: list[float]) -> str:
    return (
        f"x={pose[0]:.3f} y={pose[1]:.3f} z={pose[2]:.3f} "
        f"rx={pose[3]:.6f} ry={pose[4]:.6f} rz={pose[5]:.6f}"
    )


def print_tcp_pose(robot: Any, label: str) -> None:
    pose = parse_tcp_position(robot.get_tcp_position())
    print(f"[tcp-base] {label}: {format_tcp_pose(pose)}", flush=True)


class TcpPoseMonitor:
    """Poll and print the current JAKA TCP pose in the robot base frame."""

    def __init__(self, robot: Any, label: str, interval_s: float):
        self.robot = robot
        self.label = label
        self.interval_s = max(0.05, float(interval_s))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 2.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                pose = parse_tcp_position(self.robot.get_tcp_position())
                print(f"[tcp-base] {self.label}: {format_tcp_pose(pose)}", flush=True)
            except Exception as exc:
                print(f"[tcp-base] {self.label}: failed to read TCP pose: {exc!r}", flush=True)
            self._stop.wait(self.interval_s)


def read_tcp_pose(args: argparse.Namespace) -> list[float]:
    jkrc = import_jkrc_backend(Path(args.jkrc_dir), quiet=args.json_only)
    robot = jkrc.RC(args.jaka_ip)
    logged_in = False
    try:
        raw_login = robot.login()
        code = return_code(raw_login)
        if code is not None and code != 0:
            raise RuntimeError(f"login failed: ret={code}, raw={raw_login!r}")
        logged_in = True
        return parse_tcp_position(robot.get_tcp_position())
    finally:
        if logged_in:
            robot.logout()


def print_sdk_info(robot: Any) -> None:
    for method_name in ("get_jaka_pymoudle_version", "get_sdk_version", "get_SDK_filepath"):
        method = getattr(robot, method_name, None)
        if method is None:
            continue
        try:
            print(f"[jkrc-worker] {method_name}: {method()!r}")
        except Exception as exc:
            print(f"[jkrc-worker] {method_name} failed: {exc!r}")


def connect_gripper(comport: str, slave_address: int = 9, activate: bool = True):
    backend, robotiq_cls = import_robotiq_backend()
    if backend == "robotiq_gripper_python":
        gripper = robotiq_cls(comport=comport)
        gripper.start()
    else:
        gripper = robotiq_cls(portname=comport, slaveAddress=int(slave_address))
        if activate and hasattr(gripper, "activate"):
            gripper.activate()
    print(f"[jkrc-worker] gripper connected via {backend} on {comport}")
    return backend, gripper


def move_gripper(gripper: Any, backend: str, command: str, velocity: int = 30, force: int = 30) -> None:
    if command == "open":
        position = 0
    elif command == "close":
        position = 255
    else:
        raise ValueError(f"Unsupported gripper command: {command}")

    if backend == "robotiq_gripper_python":
        gripper.move(pos=position, vel=velocity, force=force, block=True)
    elif backend == "pyrobotiqgripper":
        gripper.goTo(position=position, speed=velocity, force=force)
    else:
        raise RuntimeError(f"Unknown gripper backend: {backend}")


def shutdown_gripper(gripper: Any, backend: str) -> None:
    try:
        if backend == "robotiq_gripper_python" and hasattr(gripper, "shutdown"):
            gripper.shutdown()
        elif hasattr(gripper, "disconnect"):
            gripper.disconnect()
    except Exception:
        pass


def execute_sequence(args: argparse.Namespace) -> None:
    sequence = json.loads(args.sequence_json)
    if not isinstance(sequence, list):
        raise ValueError("--sequence-json must decode to a list")

    jkrc = import_jkrc_backend(Path(args.jkrc_dir))
    robot = jkrc.RC(args.jaka_ip)
    gripper = None
    gripper_backend = None
    logged_in = False
    try:
        print_sdk_info(robot)
        check_call("login", robot.login())
        logged_in = True
        check_call("power_on", robot.power_on())
        check_call("enable_robot", robot.enable_robot())
        if args.tcp_monitor:
            print_tcp_pose(robot, "after_enable")

        for index, step in enumerate(sequence):
            if not isinstance(step, dict):
                raise ValueError(f"Step {index} is not an object: {step!r}")
            step_type = step.get("type")
            print(f"[jkrc-worker] step {index}: {step}")
            if step_type == "move":
                pose = step.get("pose")
                if not isinstance(pose, list) or len(pose) != 6:
                    raise ValueError(f"Move step {index} must provide a 6D pose")
                target_pose = [float(value) for value in pose]
                if args.tcp_monitor:
                    print(f"[tcp-base] step={index} target: {format_tcp_pose(target_pose)}", flush=True)
                    print_tcp_pose(robot, f"step={index} before_move")
                monitor = TcpPoseMonitor(robot, f"step={index} moving", args.tcp_monitor_interval) if args.tcp_monitor else None
                if monitor is not None:
                    monitor.start()
                try:
                    ret = robot.linear_move_extend(
                        target_pose,
                        0,
                        True,
                        float(args.velocity),
                        float(args.acceleration),
                        1,
                    )
                finally:
                    if monitor is not None:
                        monitor.stop()
                check_call("linear_move_extend", ret)
                if args.tcp_monitor:
                    print_tcp_pose(robot, f"step={index} after_move")
            elif step_type == "joint_move":
                joints = step.get("joints_rad")
                if not isinstance(joints, list) or len(joints) != 6:
                    raise ValueError(f"Joint move step {index} must provide 6 joint values in radians")
                target_joints = [float(value) for value in joints]
                print(f"[joint-base] step={index} target_rad={target_joints}", flush=True)
                ret = robot.joint_move(target_joints, 0, True, float(args.joint_velocity_rad_s))
                check_call("joint_move", ret)
                if args.tcp_monitor:
                    print_tcp_pose(robot, f"step={index} after_joint_move")
            elif step_type == "gripper":
                if args.tcp_monitor:
                    print_tcp_pose(robot, f"step={index} before_gripper")
                if gripper is None:
                    gripper_backend, gripper = connect_gripper(args.robotiq_port)
                move_gripper(gripper, gripper_backend, str(step.get("command")))
                if args.tcp_monitor:
                    print_tcp_pose(robot, f"step={index} after_gripper")
            else:
                raise ValueError(f"Unsupported step type at {index}: {step_type!r}")
    finally:
        if gripper is not None and gripper_backend is not None:
            shutdown_gripper(gripper, gripper_backend)
        if logged_in:
            robot.logout()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute JAKA/Robotiq actions in a jkrc-compatible Python.")
    parser.add_argument("--sequence-json", help="JSON list of move/gripper steps.")
    parser.add_argument("--print-tcp-pose", action="store_true", help="Read current JAKA TCP pose and print it as JSON.")
    parser.add_argument("--json-only", action="store_true", help="Suppress informational logs for machine-readable output.")
    parser.add_argument("--jaka-ip", default="192.168.1.199", help="JAKA controller IP.")
    parser.add_argument("--robotiq-port", default="/dev/ttyUSB0", help="Robotiq serial port.")
    parser.add_argument("--velocity", type=float, default=60.0, help="JAKA linear_move_extend velocity.")
    parser.add_argument("--acceleration", type=float, default=60.0, help="JAKA linear_move_extend acceleration.")
    parser.add_argument("--joint-velocity-rad-s", type=float, default=0.5, help="JAKA joint_move velocity in rad/s.")
    parser.add_argument("--no-tcp-monitor", dest="tcp_monitor", action="store_false", help="Disable live TCP pose printing during execution.")
    parser.add_argument("--tcp-monitor-interval", type=float, default=0.25, help="Seconds between live TCP pose prints.")
    parser.add_argument("--jkrc-dir", default=str(DEFAULT_JKRC_DIR), help="Directory containing jkrc.so and libjakaAPI.so.")
    parser.set_defaults(tcp_monitor=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.print_tcp_pose:
        print(json.dumps({"tcp_pose": read_tcp_pose(args)}))
        return
    if not args.sequence_json:
        raise SystemExit("--sequence-json is required unless --print-tcp-pose is used.")
    execute_sequence(args)


if __name__ == "__main__":
    main()
