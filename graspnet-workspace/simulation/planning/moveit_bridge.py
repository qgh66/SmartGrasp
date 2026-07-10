"""JSON bridge from SmartGrasp to a ROS 2 / MoveIt 2 planner."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
ROS_PLANNER = ROOT / "ros2_moveit" / "plan_jaka_zu3.py"
JAKA_OVERLAY = Path("/home/qiuguanhe/jaka_ros2/install/setup.bash")
ROS_SETUP = Path("/opt/ros/humble/setup.bash")


class MoveItPlanningError(RuntimeError):
    pass


class MoveItJakaPlanner:
    """Call MoveIt in a clean system ROS environment using JSON files."""

    def __init__(
        self,
        enabled: bool = False,
        start_move_group: bool = True,
        timeout: float = 60.0,
        ros_log_dir: str = "/tmp/smartgrasp-ros-log",
    ):
        self.enabled = bool(enabled)
        self.start_move_group = bool(start_move_group)
        self.timeout = float(timeout)
        self.ros_log_dir = ros_log_dir

    @classmethod
    def from_env(cls):
        enabled = os.environ.get("GRASP_USE_MOVEIT", "0").lower() in {"1", "true", "yes", "on"}
        start_move_group = os.environ.get("GRASP_MOVEIT_START_MOVE_GROUP", "1").lower() in {"1", "true", "yes", "on"}
        timeout = float(os.environ.get("GRASP_MOVEIT_TIMEOUT", "80"))
        ros_log_dir = os.environ.get("GRASP_MOVEIT_LOG_DIR", "/tmp/smartgrasp-ros-log")
        return cls(enabled=enabled, start_move_group=start_move_group, timeout=timeout, ros_log_dir=ros_log_dir)

    def available(self) -> bool:
        return ROS_SETUP.exists() and JAKA_OVERLAY.exists() and ROS_PLANNER.exists()

    def plan(self, start_joint_positions, link6_position, link6_orientation_xyzw) -> dict:
        if not self.enabled:
            raise MoveItPlanningError("MoveIt planner is disabled")
        if not self.available():
            raise MoveItPlanningError(
                f"MoveIt environment is incomplete: ROS_SETUP={ROS_SETUP.exists()} "
                f"JAKA_OVERLAY={JAKA_OVERLAY.exists()} ROS_PLANNER={ROS_PLANNER.exists()}"
            )

        request = {
            "group_name": "jaka_zu3",
            "link_name": "Link_6",
            "frame_id": "world",
            "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            "start_joint_positions": [float(v) for v in start_joint_positions],
            "target_pose": {
                "position": [float(v) for v in link6_position],
                "orientation_xyzw": [float(v) for v in link6_orientation_xyzw],
            },
            "position_tolerance": float(os.environ.get("GRASP_MOVEIT_POSITION_TOLERANCE", "0.003")),
            "orientation_tolerance": float(os.environ.get("GRASP_MOVEIT_ORIENTATION_TOLERANCE", "0.04")),
            "num_planning_attempts": int(os.environ.get("GRASP_MOVEIT_ATTEMPTS", "8")),
            "allowed_planning_time": float(os.environ.get("GRASP_MOVEIT_ALLOWED_TIME", "6.0")),
            "max_velocity_scaling_factor": float(os.environ.get("GRASP_MOVEIT_VEL_SCALE", "0.25")),
            "max_acceleration_scaling_factor": float(os.environ.get("GRASP_MOVEIT_ACC_SCALE", "0.25")),
            "move_group_warmup": float(os.environ.get("GRASP_MOVEIT_WARMUP", "4.0")),
        }

        with tempfile.TemporaryDirectory(prefix="smartgrasp_moveit_", dir="/tmp") as tmpdir:
            request_path = Path(tmpdir) / "request.json"
            output_path = Path(tmpdir) / "trajectory.json"
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
            cmd = self._command(request_path, output_path)
            proc = subprocess.run(
                cmd,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
                executable="/bin/bash",
            )
            output = {}
            if output_path.exists():
                output = json.loads(output_path.read_text(encoding="utf-8"))
            if proc.returncode != 0 or not output.get("success"):
                raise MoveItPlanningError(
                    f"MoveIt planning failed rc={proc.returncode}: "
                    f"{output.get('error') or output.get('error_code')}\n{proc.stdout[-4000:]}"
                )
            output["planner_stdout"] = proc.stdout[-4000:]
            return output

    def _command(self, request_path: Path, output_path: Path) -> str:
        start_flag = "--start-move-group" if self.start_move_group else ""
        return (
            "env -i "
            "HOME=$HOME USER=$USER PATH=/usr/bin:/bin:/usr/sbin:/sbin "
            f"ROS_LOG_DIR={self.ros_log_dir} "
            "bash -lc '"
            f"source {ROS_SETUP} && "
            f"source {JAKA_OVERLAY} && "
            f"python3 {ROS_PLANNER} --request {request_path} --output {output_path} {start_flag}"
            "'"
        )


def robotiq_base_pose_from_tcp(tcp_position, grasp_rotation_matrix, tcp_offset):
    grasp_pos = np.asarray(tcp_position, dtype=float)
    grasp_rot = np.asarray(grasp_rotation_matrix, dtype=float)
    approach_axis = grasp_rot[:, 0]
    opening_axis = grasp_rot[:, 1]
    approach_axis = approach_axis / max(np.linalg.norm(approach_axis), 1e-8)
    opening_axis = opening_axis / max(np.linalg.norm(opening_axis), 1e-8)

    robotiq_z = approach_axis
    robotiq_y = opening_axis - np.dot(opening_axis, robotiq_z) * robotiq_z
    robotiq_y = robotiq_y / max(np.linalg.norm(robotiq_y), 1e-8)
    robotiq_x = np.cross(robotiq_y, robotiq_z)
    robotiq_x = robotiq_x / max(np.linalg.norm(robotiq_x), 1e-8)
    robotiq_y = np.cross(robotiq_z, robotiq_x)
    robotiq_rot = np.column_stack([robotiq_x, robotiq_y, robotiq_z])
    robotiq_base_pos = grasp_pos - robotiq_rot @ np.asarray(tcp_offset, dtype=float)
    return robotiq_base_pos, Rotation.from_matrix(robotiq_rot).as_quat()


def now_stamp() -> float:
    return time.time()
