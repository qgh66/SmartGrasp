"""Robotiq gripper helpers for the real-world grasp pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def import_robotiq_backend(vendor_dir: Path):
    try:
        from robotiq_gripper_python import RobotiqGripper

        return "robotiq_gripper_python", RobotiqGripper
    except Exception as first_error:
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        try:
            from pyrobotiqgripper import RobotiqGripper

            return "pyrobotiqgripper", RobotiqGripper
        except Exception as second_error:
            raise RuntimeError(
                "Failed to import a Robotiq gripper backend. Tried robotiq_gripper_python "
                f"and pyrobotiqgripper. First error: {first_error}; second error: {second_error}"
            ) from second_error


def command_gripper(
    opening: str,
    comport: str,
    vendor_dir: Path,
    open_force: int = 30,
    close_force: int = 30,
) -> None:
    backend, robotiq_cls = import_robotiq_backend(vendor_dir)
    if backend == "robotiq_gripper_python":
        gripper = robotiq_cls(comport=comport)
        gripper.start()
    else:
        gripper = robotiq_cls(portname=comport, slaveAddress=9)
        if hasattr(gripper, "activate"):
            gripper.activate()
    try:
        if opening == "open":
            if backend == "robotiq_gripper_python":
                gripper.move(pos=0, vel=30, force=open_force, block=True)
            else:
                gripper.goTo(position=0, speed=30, force=open_force)
        elif opening == "close":
            if backend == "robotiq_gripper_python":
                gripper.move(pos=255, vel=30, force=close_force, block=True)
            else:
                gripper.goTo(position=255, speed=30, force=close_force)
        else:
            raise ValueError(f"Unsupported gripper command: {opening}")
    finally:
        if backend == "robotiq_gripper_python" and hasattr(gripper, "shutdown"):
            gripper.shutdown()
        elif hasattr(gripper, "disconnect"):
            gripper.disconnect()

