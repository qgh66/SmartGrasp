"""JAKA Zu3 arm with an attached Robotiq-85 gripper."""

from __future__ import annotations

import time
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from .planning.moveit_bridge import MoveItJakaPlanner


REPO_ROOT = Path(__file__).resolve().parents[2]
GRASPNET_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ASSET_DIR = GRASPNET_ROOT / "assets" / "robots" / "jaka_zu3"
JAKA_URDF = ROBOT_ASSET_DIR / "gazebo_jaka_zu3_robotiq.urdf"
ROBOTIQ_URDF = JAKA_URDF

JAKA_FIXED_JOINT = 0
JAKA_ARM_JOINTS = [1, 2, 3, 4, 5, 6]
JAKA_EE_LINK = 6
JAKA_JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
JAKA_FLANGE_LINK_NAME = "Link_6"
ROBOTIQ_BASE_LINK_NAME = "robotiq_85_base_link"
DEFAULT_JAKA_JOINTS = [0.0, -0.3, -0.1, -1.1, 1.57, 1.9]
ROBOTIQ_TCP_OFFSET = np.array([0.0, 0.0, 0.1625], dtype=float)
IK_CONTINUITY_WEIGHT = 0.003
MOVEIT_TCP_REFINE_THRESHOLD = 0.006
JOINT_MOVE_STEP = 0.035
JOINT_MOVE_MIN_STEPS = 8
JOINT_MOVE_MAX_STEPS = 80
JAKA_HOLD_FORCE = 500.0
GRIPPER_ANGLE_OPEN = 0.03
GRIPPER_ANGLE_CLOSE = 0.8
GRIPPER_ANGLE_CLOSE_THRESHOLD = 0.73
GRIPPER_MOTOR_FORCE = 8.0

EE_TIP_TRANSFORM = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, -0.1625],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)


def _joint_delta_norm(values, reference):
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    delta = (values - reference + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(delta))


def _joint_delta(values, reference):
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return (values - reference + np.pi) % (2.0 * np.pi) - np.pi


def _link_frame_pose(body_id: int, link_id: int):
    state = p.getLinkState(body_id, link_id)
    return state[4], state[5]


def _tip_pose_from_ee(ee_pos, ee_orn):
    ee_rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    tip_pos = np.asarray(ee_pos, dtype=float) + ee_rot @ ROBOTIQ_TCP_OFFSET
    return tip_pos, ee_orn


class JakaZu3Robotiq85Gripper:
    """Adapter exposing the evaluator's gripper interface.

    Unlike the previous visual-only wrapper, this class moves the JAKA arm with
    PyBullet IK and keeps the Robotiq-85 base fixed to the JAKA Link_6 flange.
    """

    _max_opening = 0.085

    def __init__(self, robot_base_position=(-0.05, 0.0, 0.0), planner: MoveItJakaPlanner | None = None):
        self.robot_base_position = np.asarray(robot_base_position, dtype=float)
        self.planner = planner
        self.robot_id = None
        self.robotiq_id = None
        self.base_id = None
        self.left_id = None
        self.right_id = None
        self.grasp_constraint = None
        self.robot_gripper_constraint = None
        self._mimic_constraints: list[int] = []
        self._joint_map: dict[str, int] = {}
        self._link_map: dict[str, int] = {}
        self._robotiq_joint_map: dict[str, int] = {}
        self._resolved_urdf_path: str | None = None
        self.ee_tip_id = None
        self.ee_finger_pad_ids: list[int] = []
        self._current_opening = 0.06
        self.robot_joint_values = list(DEFAULT_JAKA_JOINTS)
        self.last_motion_plan: dict | None = None
        self.last_motion_error: str | None = None

    def load(self, position=(0.3, 0.0, 0.3), orientation=(0, 0, 0, 1)):
        self._load_robot()
        self._load_robotiq()
        return self.base_id

    def remove(self):
        self.release_grasp()
        if self.robot_gripper_constraint is not None:
            p.removeConstraint(self.robot_gripper_constraint)
            self.robot_gripper_constraint = None
        for body_id in dict.fromkeys((self.robotiq_id, self.robot_id)):
            if body_id is not None:
                p.removeBody(body_id)
        self.robotiq_id = None
        self.robot_id = None
        if self._resolved_urdf_path:
            try:
                Path(self._resolved_urdf_path).unlink(missing_ok=True)
            except OSError:
                pass
            self._resolved_urdf_path = None

    def set_pose(self, position, rotation_matrix):
        grasp_pos = np.asarray(position, dtype=float)
        target_pos, target_rot = self._grasp_pose_to_robotiq_base(grasp_pos, rotation_matrix)
        target_orn = Rotation.from_matrix(target_rot).as_quat()
        self.last_motion_plan = None
        self.last_motion_error = None
        self._move_arm_to(target_pos, target_orn, grasp_pos)
        self._sync_attachment()

    def plan_and_execute_pose(self, position, rotation_matrix, steps_per_point: int = 3):
        if self.planner is None or not self.planner.enabled:
            self.set_pose(position, rotation_matrix)
            return False
        grasp_pos = np.asarray(position, dtype=float)
        target_pos, target_rot = self._grasp_pose_to_robotiq_base(grasp_pos, rotation_matrix)
        target_orn = Rotation.from_matrix(target_rot).as_quat()
        self.last_motion_plan = None
        self.last_motion_error = None
        try:
            plan = self.planner.plan(self.robot_joint_values, target_pos, target_orn)
        except Exception as exc:
            self.last_motion_error = str(exc)
            self.set_pose(position, rotation_matrix)
            return False
        self._execute_joint_trajectory(plan.get("trajectory", {}), steps_per_point=steps_per_point)
        self.last_motion_plan = plan
        self._sync_attachment()
        tcp_pos, _ = self.get_tcp_pose()
        tcp_error = float(np.linalg.norm(np.asarray(tcp_pos, dtype=float) - grasp_pos))
        self.last_motion_plan["pybullet_tcp_error"] = tcp_error
        if tcp_error > MOVEIT_TCP_REFINE_THRESHOLD:
            self._move_arm_to(target_pos, target_orn, grasp_pos)
            self._sync_attachment()
            tcp_pos, _ = self.get_tcp_pose()
            self.last_motion_plan["pybullet_tcp_error_after_refine"] = float(
                np.linalg.norm(np.asarray(tcp_pos, dtype=float) - grasp_pos)
            )
        return True

    def set_opening(self, width: float):
        self._current_opening = float(np.clip(width, 0.002, self._max_opening))
        self._move_gripper_angle(GRIPPER_ANGLE_OPEN)

    def close_fingers(self, target_width: float, steps: int = 100):
        target_width = float(np.clip(target_width, 0.002, self._max_opening))
        self._current_opening = target_width
        self._move_gripper_angle(GRIPPER_ANGLE_CLOSE, timeout=3.0, is_slow=True)

    def is_gripper_closed(self):
        main_joint = self._robotiq_joint_map.get("finger_joint")
        if main_joint is None:
            return False
        gripper_angle = p.getJointState(self.robotiq_id, main_joint)[0]
        return gripper_angle < GRIPPER_ANGLE_CLOSE_THRESHOLD

    def create_grasp_constraint(self, object_id):
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(self.robotiq_id)
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
        inv_gripper_pos, inv_gripper_orn = p.invertTransform(gripper_pos, gripper_orn)
        parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
            inv_gripper_pos, inv_gripper_orn, obj_pos, obj_orn
        )
        self.grasp_constraint = p.createConstraint(
            self.robotiq_id,
            -1,
            object_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            parent_frame_pos,
            [0, 0, 0],
            parentFrameOrientation=parent_frame_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )
        return self.grasp_constraint

    def get_tcp_pose(self):
        if self.robot_id == self.robotiq_id:
            ee_pos, ee_orn = _link_frame_pose(self.robot_id, JAKA_EE_LINK)
            return _tip_pose_from_ee(ee_pos, ee_orn)
        robotiq_pos, robotiq_orn = p.getBasePositionAndOrientation(self.robotiq_id)
        return _tip_pose_from_ee(robotiq_pos, robotiq_orn)

    def contact_body_ids(self):
        return list(dict.fromkeys([self.robotiq_id]))

    def collision_body_ids(self):
        return list(dict.fromkeys([self.robot_id, self.robotiq_id]))

    def release_grasp(self):
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None

    def snapshot_extra(self):
        robotiq_pos, robotiq_orn = p.getBasePositionAndOrientation(self.robotiq_id)
        ee_pos, ee_orn = _link_frame_pose(self.robot_id, JAKA_EE_LINK)
        joint_values = [p.getJointState(self.robot_id, joint_idx)[0] for joint_idx in JAKA_ARM_JOINTS]
        return {
            "robot_model": "jaka_zu3_robotiq85",
            "jaka_ee_pos": list(ee_pos),
            "jaka_ee_orn": list(ee_orn),
            "jaka_joint_values": joint_values,
            "robotiq_pos": list(robotiq_pos),
            "robotiq_orn": list(robotiq_orn),
            "robotiq_opening": float(self._current_opening),
            "moveit_planned": bool(self.last_motion_plan),
            "moveit_error": self.last_motion_error,
        }

    def metadata(self):
        return {
            "model": "jaka_zu3_robotiq85",
            "jaka_urdf": str(JAKA_URDF),
            "robotiq_urdf": str(ROBOTIQ_URDF),
            "jaka_base_position": list(self.robot_base_position),
            "jaka_joint_values": list(self.robot_joint_values),
            "execution": "jaka_moveit_attached_robotiq" if self.planner and self.planner.enabled else "jaka_ik_attached_robotiq",
            "moveit_enabled": bool(self.planner and self.planner.enabled),
        }

    def _execute_joint_trajectory(self, trajectory: dict, steps_per_point: int = 3):
        points = trajectory.get("points", [])
        joint_names = trajectory.get("joint_names", [])
        if not points or not joint_names:
            raise RuntimeError("MoveIt returned an empty joint trajectory")
        name_to_joint = {f"joint_{i + 1}": joint_idx for i, joint_idx in enumerate(JAKA_ARM_JOINTS)}
        ordered_joints = [name_to_joint[name] for name in joint_names if name in name_to_joint]
        if len(ordered_joints) != len(JAKA_ARM_JOINTS):
            raise RuntimeError(f"Unexpected MoveIt joint names: {joint_names}")

        current = np.array([p.getJointState(self.robot_id, joint_idx)[0] for joint_idx in ordered_joints], dtype=float)
        for point in points:
            target = np.asarray(point.get("positions", []), dtype=float)
            if len(target) != len(ordered_joints):
                continue
            for step in range(max(1, steps_per_point)):
                frac = (step + 1) / max(1, steps_per_point)
                values = current + (target - current) * frac
                for joint_idx, value in zip(ordered_joints, values):
                    p.resetJointState(self.robot_id, joint_idx, float(value))
                self._sync_attachment()
                p.stepSimulation()
            current = target
        self.robot_joint_values = [p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS]

    def _load_robot(self):
        global JAKA_ARM_JOINTS, JAKA_EE_LINK

        if not JAKA_URDF.exists():
            raise FileNotFoundError(f"JAKA Zu3 URDF not found: {JAKA_URDF}")
        resolved_urdf = self._resolve_package_uris(JAKA_URDF)
        self.robot_id = p.loadURDF(
            str(resolved_urdf),
            basePosition=self.robot_base_position.tolist(),
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        self._index_loaded_model()
        JAKA_ARM_JOINTS = [self._joint_map[name] for name in JAKA_JOINT_NAMES]
        JAKA_EE_LINK = self._link_map.get(ROBOTIQ_BASE_LINK_NAME, self._link_map[JAKA_FLANGE_LINK_NAME])
        for joint_idx, joint_value in zip(JAKA_ARM_JOINTS, self.robot_joint_values):
            p.resetJointState(self.robot_id, joint_idx, joint_value)

    def _load_robotiq(self):
        self.robotiq_id = self.robot_id
        self.base_id = self.robot_id
        self.left_id = self.robot_id
        self.right_id = self.robot_id
        self._robotiq_joint_map = dict(self._joint_map)
        self._add_robotiq_joint_aliases()
        for joint_idx in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, joint_idx)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            joint_type = info[2]
            if link_name == ROBOTIQ_BASE_LINK_NAME:
                self.ee_tip_id = joint_idx
            if "finger_tip" in joint_name or "finger_tip" in link_name:
                self.ee_finger_pad_ids.append(joint_idx)
                p.changeDynamics(self.robot_id, joint_idx, lateralFriction=3.0, spinningFriction=0.2)
            elif "robotiq_85" in joint_name and joint_type == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(
                    self.robot_id, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0
                )
        self._setup_robotiq_mimic_constraints()
        self.set_opening(0.06)
        self._sync_attachment()

    def _resolve_package_uris(self, urdf_path: Path) -> Path:
        content = urdf_path.read_text(encoding="utf-8")
        replacements = [
            ("package://robotiq_description/meshes/", f"{ROBOT_ASSET_DIR.as_posix()}/meshes/robotiq_description/"),
            ("package://robotiq_description/", f"{ROBOT_ASSET_DIR.as_posix()}/meshes/robotiq_description/"),
            ("package://jaka_description/", f"{ROBOT_ASSET_DIR.as_posix()}/"),
            ("package://jaka_rviz/", f"{ROBOT_ASSET_DIR.as_posix()}/"),
        ]
        for old, new in replacements:
            content = content.replace(old, new)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
        tmp.write(content)
        tmp.close()
        self._resolved_urdf_path = tmp.name
        return Path(tmp.name)

    def _index_loaded_model(self):
        self._joint_map = {}
        self._link_map = {}
        for joint_idx in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, joint_idx)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            self._joint_map[joint_name] = joint_idx
            self._link_map[link_name] = joint_idx

    def _add_robotiq_joint_aliases(self):
        aliases = {
            "finger_joint": "robotiq_85_left_knuckle_joint",
            "right_outer_knuckle_joint": "robotiq_85_right_knuckle_joint",
            "left_inner_knuckle_joint": "robotiq_85_left_inner_knuckle_joint",
            "right_inner_knuckle_joint": "robotiq_85_right_inner_knuckle_joint",
            "left_inner_finger_joint": "robotiq_85_left_finger_tip_joint",
            "right_inner_finger_joint": "robotiq_85_right_finger_tip_joint",
        }
        for alias, actual_name in aliases.items():
            if actual_name in self._robotiq_joint_map:
                self._robotiq_joint_map[alias] = self._robotiq_joint_map[actual_name]

    def _move_arm_to(self, target_pos, target_orn, tcp_target=None):
        lower_limits = [-6.28, -1.48, -3.05, -1.48, -6.28, -6.28]
        upper_limits = [6.28, 4.62, 3.05, 4.62, 6.28, 6.28]
        joint_ranges = [u - l for l, u in zip(lower_limits, upper_limits)]
        original = [p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS]
        # Keep IK solving continuous in GUI mode. The previous implementation
        # temporarily reset the live robot into several rest poses to score IK
        # candidates; PyBullet's GUI renders those internal probes, which looks
        # like violent random arm twitching. Use the current joint state as the
        # only rest pose and move smoothly to that single solution.
        solution = p.calculateInverseKinematics(
            self.robot_id,
            JAKA_EE_LINK,
            target_pos.tolist(),
            target_orn.tolist(),
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=original,
            maxNumIterations=2000,
            residualThreshold=1e-6,
        )
        if solution:
            self._move_joints_smooth(solution[: len(JAKA_ARM_JOINTS)])
        self.robot_joint_values = [p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS]

    def _move_joints_smooth(self, target_joints):
        """Move through waypoints while keeping the IK target exact in PyBullet.

        The imported JAKA URDF does not reliably converge with POSITION_CONTROL
        in this scene, so we interpolate joint states explicitly and step the
        simulator. This keeps the ThinkGrasp-style straight primitive visible
        without letting the TCP drift away from the commanded grasp pose.
        """
        target = np.asarray(target_joints, dtype=float)
        current = np.asarray([p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS], dtype=float)
        delta = _joint_delta(target, current)
        max_delta = float(np.max(np.abs(delta))) if len(delta) else 0.0
        steps = int(np.clip(np.ceil(max_delta / JOINT_MOVE_STEP), JOINT_MOVE_MIN_STEPS, JOINT_MOVE_MAX_STEPS))
        for step in range(steps):
            frac = (step + 1) / steps
            waypoint = current + delta * frac
            for joint_idx, joint_value in zip(JAKA_ARM_JOINTS, waypoint):
                p.resetJointState(self.robot_id, joint_idx, float(joint_value))
            self.robot_joint_values = waypoint.tolist()
            self._sync_attachment()
            p.stepSimulation()
        self.robot_joint_values = [p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS]
        self._hold_current_joints()

    def _hold_current_joints(self):
        if self.robot_id is None:
            return
        if not self.robot_joint_values:
            self.robot_joint_values = [p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS]
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=JAKA_ARM_JOINTS,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(self.robot_joint_values),
            positionGains=[1.0] * len(JAKA_ARM_JOINTS),
            forces=[JAKA_HOLD_FORCE] * len(JAKA_ARM_JOINTS),
        )

    @staticmethod
    def _grasp_pose_to_robotiq_base(position, rotation_matrix):
        grasp_pos = np.asarray(position, dtype=float)
        grasp_rot = np.asarray(rotation_matrix, dtype=float)
        approach_axis = grasp_rot[:, 0]
        opening_axis = grasp_rot[:, 1]
        approach_axis = approach_axis / max(np.linalg.norm(approach_axis), 1e-8)
        opening_axis = opening_axis - np.dot(opening_axis, approach_axis) * approach_axis
        opening_axis = opening_axis / max(np.linalg.norm(opening_axis), 1e-8)

        robotiq_z = approach_axis
        robotiq_y = opening_axis
        robotiq_x = np.cross(robotiq_y, robotiq_z)
        robotiq_x = robotiq_x / max(np.linalg.norm(robotiq_x), 1e-8)
        robotiq_y = np.cross(robotiq_z, robotiq_x)
        robotiq_rot = np.column_stack([robotiq_x, robotiq_y, robotiq_z])
        robotiq_base_pos = grasp_pos - robotiq_rot @ ROBOTIQ_TCP_OFFSET
        return robotiq_base_pos, robotiq_rot

    def _sync_attachment(self):
        if self.robot_id is None or self.robotiq_id is None:
            return
        if self.robot_id == self.robotiq_id:
            self._hold_current_joints()
            for _ in range(2):
                p.stepSimulation()
            return
        self._hold_current_joints()
        ee_pos, ee_orn = _link_frame_pose(self.robot_id, JAKA_EE_LINK)
        p.resetBasePositionAndOrientation(self.robotiq_id, ee_pos, ee_orn)
        for _ in range(2):
            self._hold_current_joints()
            p.stepSimulation()
        ee_pos, ee_orn = _link_frame_pose(self.robot_id, JAKA_EE_LINK)
        p.resetBasePositionAndOrientation(self.robotiq_id, ee_pos, ee_orn)

    def _setup_robotiq_mimic_constraints(self):
        for constraint_id in self._mimic_constraints:
            p.removeConstraint(constraint_id)
        self._mimic_constraints = []

        def add_gear(parent_name, child_name, axis, ratio, max_force=10000):
            parent_joint = self._robotiq_joint_map.get(parent_name)
            child_joint = self._robotiq_joint_map.get(child_name)
            if parent_joint is None or child_joint is None:
                return
            constraint_id = p.createConstraint(
                self.robotiq_id,
                parent_joint,
                self.robotiq_id,
                child_joint,
                jointType=p.JOINT_GEAR,
                jointAxis=axis,
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(constraint_id, gearRatio=ratio, erp=0.8, maxForce=max_force)
            self._mimic_constraints.append(constraint_id)

        add_gear("finger_joint", "left_inner_finger_joint", [1, 0, 0], 1)
        add_gear("finger_joint", "left_inner_knuckle_joint", [1, 0, 0], -1)
        add_gear("right_outer_knuckle_joint", "right_inner_finger_joint", [1, 0, 0], 1)
        add_gear("right_outer_knuckle_joint", "right_inner_knuckle_joint", [1, 0, 0], -1)
        add_gear("finger_joint", "right_outer_knuckle_joint", [0, 1, 0], -1, max_force=1000)

    def _command_gripper_angle(self, joint_value: float, step_count: int = 10):
        if self.robotiq_id is None:
            return
        main_joint = self._robotiq_joint_map.get("finger_joint")
        right_outer_joint = self._robotiq_joint_map.get("right_outer_knuckle_joint")
        for joint_idx in (main_joint, right_outer_joint):
            if joint_idx is None:
                continue
            p.setJointMotorControl2(
                self.robotiq_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_value,
                force=GRIPPER_MOTOR_FORCE,
            )
        for _ in range(step_count):
            self._hold_current_joints()
            p.stepSimulation()

    def _move_gripper_angle(self, target_angle: float, timeout: float = 3.0, is_slow: bool = False):
        main_joint = self._robotiq_joint_map.get("finger_joint")
        right_outer_joint = self._robotiq_joint_map.get("right_outer_knuckle_joint")
        if main_joint is None:
            return

        if is_slow:
            previous_angle = p.getJointState(self.robotiq_id, main_joint)[0]
            direction = 1.0 if target_angle > previous_angle else -1.0
            for joint_idx in (main_joint, right_outer_joint):
                if joint_idx is None:
                    continue
                p.setJointMotorControl2(
                    self.robotiq_id,
                    joint_idx,
                    p.VELOCITY_CONTROL,
                    targetVelocity=direction,
                    force=GRIPPER_MOTOR_FORCE,
                )
            for _ in range(10):
                self._hold_current_joints()
                p.stepSimulation()

            start_time = time.time()
            stable_steps = 0
            while (time.time() - start_time) < timeout:
                current_angle = p.getJointState(self.robotiq_id, main_joint)[0]
                if abs(current_angle - previous_angle) < 1e-4:
                    stable_steps += 1
                else:
                    stable_steps = 0
                if stable_steps >= 2:
                    break
                previous_angle = current_angle
                for _ in range(10):
                    self._hold_current_joints()
                    p.stepSimulation()

        self._command_gripper_angle(target_angle)
