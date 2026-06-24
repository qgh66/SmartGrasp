#!/usr/bin/env python3
"""Plan JAKA Zu3 arm trajectories with MoveIt 2.

This script is intentionally ROS-only and is launched from a clean system
Python environment. SmartGrasp communicates with it through JSON files so the
conda environment never has to import ROS Python modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetMotionPlan
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def _load_request(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_output(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _pose_from_list(position: list[float], orientation_xyzw: list[float]) -> Pose:
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.x = float(orientation_xyzw[0])
    pose.orientation.y = float(orientation_xyzw[1])
    pose.orientation.z = float(orientation_xyzw[2])
    pose.orientation.w = float(orientation_xyzw[3])
    return pose


def _make_goal_constraints(data: dict) -> Constraints:
    goal = Constraints()
    frame_id = data.get("frame_id", "world")
    link_name = data.get("link_name", "Link_6")
    target = data["target_pose"]
    pose = _pose_from_list(target["position"], target["orientation_xyzw"])

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [float(data.get("position_tolerance", 0.01))]
    region = BoundingVolume()
    region.primitives.append(primitive)
    region.primitive_poses.append(pose)

    pc = PositionConstraint()
    pc.header.frame_id = frame_id
    pc.link_name = link_name
    pc.constraint_region = region
    pc.weight = 1.0

    oc = OrientationConstraint()
    oc.header.frame_id = frame_id
    oc.link_name = link_name
    oc.orientation = pose.orientation
    tol = float(data.get("orientation_tolerance", 0.08))
    oc.absolute_x_axis_tolerance = tol
    oc.absolute_y_axis_tolerance = tol
    oc.absolute_z_axis_tolerance = tol
    oc.weight = 1.0

    goal.position_constraints.append(pc)
    goal.orientation_constraints.append(oc)
    return goal


def _make_start_state(data: dict) -> RobotState:
    start = RobotState()
    joints = data.get("start_joint_positions") or [0.0] * len(JOINT_NAMES)
    names = data.get("joint_names") or JOINT_NAMES
    start.joint_state = JointState()
    start.joint_state.name = list(names)
    start.joint_state.position = [float(v) for v in joints]
    start.is_diff = False
    return start


def _trajectory_to_dict(response) -> dict:
    traj = response.motion_plan_response.trajectory.joint_trajectory
    points = []
    for point in traj.points:
        points.append(
            {
                "positions": list(point.positions),
                "velocities": list(point.velocities),
                "accelerations": list(point.accelerations),
                "time_from_start": point.time_from_start.sec + point.time_from_start.nanosec * 1e-9,
            }
        )
    return {
        "joint_names": list(traj.joint_names),
        "points": points,
    }


def _start_move_group() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("ROS_LOG_DIR", "/tmp/smartgrasp-ros-log")
    cmd = [
        "ros2",
        "launch",
        "jaka_zu3_moveit_config",
        "move_group.launch.py",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        start_new_session=True,
    )


def _stop_process(proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)


def _read_available_output(proc: subprocess.Popen | None) -> list[str]:
    return []


def _wait_for_service(node, service_name: str, timeout: float):
    client = node.create_client(GetMotionPlan, service_name)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.wait_for_service(timeout_sec=0.5):
            return client
    raise TimeoutError(f"MoveIt service not available: {service_name}")


def _plan_once(data: dict):
    rclpy.init(args=None)
    node = rclpy.create_node("smartgrasp_jaka_zu3_json_planner")
    try:
        client = _wait_for_service(
            node,
            data.get("service_name", "/plan_kinematic_path"),
            float(data.get("service_timeout", 20.0)),
        )
        req = GetMotionPlan.Request()
        mpr = req.motion_plan_request
        mpr.group_name = data.get("group_name", "jaka_zu3")
        mpr.pipeline_id = data.get("pipeline_id", "ompl")
        mpr.planner_id = data.get("planner_id", "")
        mpr.num_planning_attempts = int(data.get("num_planning_attempts", 8))
        mpr.allowed_planning_time = float(data.get("allowed_planning_time", 5.0))
        mpr.max_velocity_scaling_factor = float(data.get("max_velocity_scaling_factor", 0.25))
        mpr.max_acceleration_scaling_factor = float(data.get("max_acceleration_scaling_factor", 0.25))
        mpr.start_state = _make_start_state(data)
        mpr.goal_constraints.append(_make_goal_constraints(data))

        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=float(data.get("plan_timeout", 30.0)))
        if not future.done() or future.result() is None:
            raise TimeoutError("MoveIt planning call timed out")
        response = future.result()
        error_code = int(response.motion_plan_response.error_code.val)
        success = error_code == MoveItErrorCodes.SUCCESS
        return {
            "success": success,
            "error_code": error_code,
            "planning_time": float(response.motion_plan_response.planning_time),
            "trajectory": _trajectory_to_dict(response),
        }
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan a JAKA Zu3 trajectory with MoveIt 2")
    parser.add_argument("--request", required=True, help="Input JSON request")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--start-move-group", action="store_true", help="Launch move_group for this planning call")
    parser.add_argument("--startup-timeout", type=float, default=25.0)
    args = parser.parse_args()

    request_path = Path(args.request)
    output_path = Path(args.output)
    proc = None
    try:
        data = _load_request(request_path)
        if args.start_move_group:
            proc = _start_move_group()
            time.sleep(float(data.get("move_group_warmup", 4.0)))
        result = _plan_once(data)
        result["request"] = data
        _write_output(output_path, result)
        return 0 if result["success"] else 2
    except Exception as exc:
        _write_output(
            output_path,
            {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "move_group_output": _read_available_output(proc),
            },
        )
        return 1
    finally:
        _stop_process(proc)


if __name__ == "__main__":
    sys.exit(main())
