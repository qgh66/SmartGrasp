"""JAKA motion helpers and persistent worker client."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from utils.data_loader import json_safe
from utils.gripper import command_gripper


def import_jkrc_backend(jkrc_dir: Path):
    jkrc_path = str(jkrc_dir)
    if jkrc_dir.exists() and jkrc_path not in sys.path:
        sys.path.insert(0, jkrc_path)

    local_jaka_api = jkrc_dir / "libjakaAPI.so"
    if local_jaka_api.exists():
        import ctypes

        ctypes.CDLL(str(local_jaka_api), mode=ctypes.RTLD_GLOBAL)

    try:
        import jkrc
    except Exception as exc:
        raise RuntimeError(
            "JAKA 执行模式需要 jkrc。已优先尝试加载本项目下的 "
            f"{jkrc_dir / 'jkrc.so'}，但导入失败: {exc!r}。如果错误包含 "
            "Py_TPFLAGS_HAVE_GC，说明这个 jkrc.so 与当前 Python ABI 不兼容，"
            "需要换成当前 smartgrasp Python 版本匹配的 JAKA jkrc.so/wheel，"
            "或切到该 jkrc 编译时对应的 Python 环境。"
        ) from exc

    print(f"[jkrc] loaded from {getattr(jkrc, '__file__', 'unknown')}")
    return jkrc


def jaka_return_code(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
        first = raw[0]
        if isinstance(first, (int, np.integer)):
            return int(first)
    return None


def check_jaka_call(name: str, raw: Any, allow_none: bool = True) -> None:
    print(f"[jaka] {name} returned: {raw!r}")
    code = jaka_return_code(raw)
    if code is None:
        if allow_none:
            return
        raise RuntimeError(f"{name} returned unexpected format: {raw!r}")
    if code != 0:
        if name == "login" and code == -1:
            raise RuntimeError(
                "JAKA login failed with ret=-1. The loaded JAKA SDK V2.2.7 requires "
                "controller version 1.7.2_28 or newer. For controller 1.7.0_x or 1.5.x, "
                "use SDK v2.1.11 or earlier, or upgrade the robot controller firmware."
            )
        raise RuntimeError(f"{name} failed: ret={code}, raw={raw!r}")



def move_jaka_pose(target_pose: list[float], ip: str, velocity: float, acceleration: float, jkrc_dir: Path) -> None:
    jkrc = import_jkrc_backend(jkrc_dir)

    robot = jkrc.RC(ip)
    check_jaka_call("login", robot.login())
    try:
        check_jaka_call("power_on", robot.power_on())
        check_jaka_call("enable_robot", robot.enable_robot())
        ret = robot.linear_move_extend(target_pose, 0, True, velocity, acceleration, 1)
        check_jaka_call("linear_move_extend", ret)
    finally:
        robot.logout()


def move_jaka_joints(target_joints_rad: list[float], ip: str, velocity_rad_s: float, jkrc_dir: Path) -> None:
    jkrc = import_jkrc_backend(jkrc_dir)

    robot = jkrc.RC(ip)
    check_jaka_call("login", robot.login())
    try:
        check_jaka_call("power_on", robot.power_on())
        check_jaka_call("enable_robot", robot.enable_robot())
        ret = robot.joint_move(target_joints_rad, 0, True, velocity_rad_s)
        check_jaka_call("joint_move", ret)
    finally:
        robot.logout()


class PersistentJakaWorker:
    """Keep one JAKA subprocess alive across capture/grasp cycles."""

    def __init__(self, args: argparse.Namespace, jaka_worker: Path, worker_ready_prefix: str, worker_response_prefix: str):
        self.args = args
        self.jaka_worker = jaka_worker
        self.worker_ready_prefix = worker_ready_prefix
        self.worker_response_prefix = worker_response_prefix
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "PersistentJakaWorker":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return
        jaka_python = Path(self.args.jaka_python).expanduser()
        if not jaka_python.exists():
            raise FileNotFoundError(
                f"JAKA subprocess Python does not exist: {jaka_python}. "
                "Create a Python 3.10 env for jkrc or pass --jaka-python /path/to/python."
            )
        if not self.jaka_worker.exists():
            raise FileNotFoundError(f"JAKA worker script does not exist: {self.jaka_worker}")

        command = [
            str(jaka_python),
            str(self.jaka_worker),
            "--stdio-server",
            "--jaka-ip",
            self.args.jaka_ip,
            "--robotiq-port",
            self.args.robotiq_port,
            "--velocity",
            str(self.args.velocity),
            "--acceleration",
            str(self.args.acceleration),
            "--joint-velocity-rad-s",
            str(self.args.joint_velocity_rad_s),
            "--gripper-open-force",
            str(self.args.gripper_open_force),
            "--gripper-close-force",
            str(self.args.gripper_close_force),
            "--jkrc-dir",
            self.args.jkrc_dir,
        ]
        print(f"[jaka] starting persistent subprocess: {' '.join(command[:3])}", flush=True)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        ready = self._read_response(self.worker_ready_prefix)
        if not ready.get("ok"):
            raise RuntimeError(f"Persistent JAKA worker failed to start: {ready!r}")

    def execute_sequence(self, sequence: list[dict[str, Any]], label: str) -> None:
        response = self._request(
            {
                "command": "execute_sequence",
                "label": label,
                "sequence": json_safe(sequence),
            }
        )
        if not response.get("ok"):
            raise RuntimeError(f"Persistent JAKA sequence failed: {response!r}")

    def read_tcp_pose(self) -> list[float]:
        response = self._request({"command": "read_tcp_pose"})
        if not response.get("ok"):
            raise RuntimeError(f"Persistent JAKA TCP read failed: {response!r}")
        tcp_pose = response.get("tcp_pose")
        if not isinstance(tcp_pose, list) or len(tcp_pose) != 6:
            raise RuntimeError(f"Persistent JAKA TCP read returned invalid pose: {response!r}")
        return [float(value) for value in tcp_pose]

    def close(self) -> None:
        if self.process is None:
            return
        process = self.process
        if process.poll() is None:
            try:
                self._send({"command": "shutdown"})
                self._read_response(self.worker_response_prefix)
            except Exception as exc:
                print(f"[jaka] persistent worker shutdown request failed: {exc!r}", flush=True)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5.0)
        self.process = None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._send(payload)
        return self._read_response(self.worker_response_prefix)

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("Persistent JAKA worker is not running.")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _read_response(self, prefix: str) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Persistent JAKA worker is not running.")
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(f"Persistent JAKA worker exited with code {self.process.poll()}.")
            line = line.rstrip()
            if line.startswith(prefix):
                return json.loads(line[len(prefix) :])
            print(line, flush=True)


def run_jaka_sequence(sequence: list[dict[str, Any]], args: argparse.Namespace, label: str) -> None:
    persistent_worker = getattr(args, "_persistent_jaka_worker", None)
    if persistent_worker is not None:
        persistent_worker.execute_sequence(sequence, label)
        return

    if args.jaka_executor == "direct":
        for step in sequence:
            if step["type"] == "move":
                move_jaka_pose(step["pose"], args.jaka_ip, args.velocity, args.acceleration, Path(args.jkrc_dir))
            elif step["type"] == "joint_move":
                move_jaka_joints(step["joints_rad"], args.jaka_ip, args.joint_velocity_rad_s, Path(args.jkrc_dir))
            elif step["type"] == "gripper":
                command_gripper(
                    step["command"],
                    args.robotiq_port,
                    Path(args.vendor_dir),
                    args.gripper_open_force,
                    args.gripper_close_force,
                )
            else:
                raise ValueError(f"Unsupported JAKA step: {step}")
        return

    jaka_python = Path(args.jaka_python).expanduser()
    if not jaka_python.exists():
        raise FileNotFoundError(
            f"JAKA subprocess Python does not exist: {jaka_python}. "
            "Create a Python 3.10 env for jkrc or pass --jaka-python /path/to/python."
        )
    jaka_worker = Path(args.jaka_worker)
    if not jaka_worker.exists():
        raise FileNotFoundError(f"JAKA worker script does not exist: {jaka_worker}")

    command = [
        str(jaka_python),
        str(jaka_worker),
        "--sequence-json",
        json.dumps(json_safe(sequence)),
        "--jaka-ip",
        args.jaka_ip,
        "--robotiq-port",
        args.robotiq_port,
        "--velocity",
        str(args.velocity),
        "--acceleration",
        str(args.acceleration),
        "--joint-velocity-rad-s",
        str(args.joint_velocity_rad_s),
        "--gripper-open-force",
        str(args.gripper_open_force),
        "--gripper-close-force",
        str(args.gripper_close_force),
        "--jkrc-dir",
        args.jkrc_dir,
    ]
    print(f"[jaka] running {label} via subprocess: {' '.join(command[:2])}")
    subprocess.run(command, check=True)


def read_jaka_tcp_pose(args: argparse.Namespace) -> list[float]:
    persistent_worker = getattr(args, "_persistent_jaka_worker", None)
    if persistent_worker is not None:
        return persistent_worker.read_tcp_pose()

    if args.jaka_executor == "direct":
        jkrc = import_jkrc_backend(Path(args.jkrc_dir))
        robot = jkrc.RC(args.jaka_ip)
        check_jaka_call("login", robot.login())
        try:
            raw = robot.get_tcp_position()
        finally:
            robot.logout()
        return parse_jaka_tcp_pose(raw)

    jaka_python = Path(args.jaka_python).expanduser()
    jaka_worker = Path(args.jaka_worker)
    command = [
        str(jaka_python),
        str(jaka_worker),
        "--print-tcp-pose",
        "--json-only",
        "--jaka-ip",
        args.jaka_ip,
        "--jkrc-dir",
        args.jkrc_dir,
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
    raise RuntimeError(f"Could not parse TCP pose from JAKA worker output: {result.stdout!r}")


def parse_jaka_tcp_pose(raw: Any) -> list[float]:
    ret = None
    pos = None
    if isinstance(raw, (list, tuple, np.ndarray)):
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
    elif isinstance(raw, (int, np.integer)):
        ret, pos = int(raw), None
    if ret is None or pos is None:
        raise RuntimeError(f"get_tcp_position returned unexpected format: {raw!r}")
    if int(ret) != 0:
        raise RuntimeError(f"get_tcp_position failed: ret={ret}, raw={raw!r}")
    if not isinstance(pos, (list, tuple, np.ndarray)) or len(pos) != 6:
        raise RuntimeError(f"get_tcp_position pose must be 6D: pos={pos!r}, raw={raw!r}")
    return [float(value) for value in pos]
