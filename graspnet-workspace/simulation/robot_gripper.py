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
# Measured from robotiq_85_base_link to the midpoint between the open fingertip
# link frames in this combined URDF. The old 162.5 mm ThinkGrasp constant refers
# to a different gripper attachment/frame convention.
ROBOTIQ_TCP_OFFSET = np.array([0.1054, 0.0, 0.0], dtype=float)
IK_CONTINUITY_WEIGHT = 0.003
MOVEIT_TCP_REFINE_THRESHOLD = 0.006
MAX_EXECUTION_TCP_ERROR = 0.01
JOINT_MOVE_STEP = 0.035
JOINT_MOVE_MIN_STEPS = 8
JOINT_MOVE_MAX_STEPS = 80
TRANSPORT_JOINT_MOVE_STEP = 0.006
TRANSPORT_JOINT_MOVE_MIN_STEPS = 60
TRANSPORT_JOINT_MOVE_MAX_STEPS = 400
TRANSPORT_GUI_DELAY_SCALE = 2.0
JAKA_HOLD_FORCE = 500.0
GRIPPER_ANGLE_OPEN = 0.03
GRIPPER_ANGLE_CLOSE = 0.8
GRIPPER_ANGLE_CLOSE_THRESHOLD = 0.73
GRIPPER_MOTOR_FORCE = 50.0
GRIPPER_MIMIC_CONSTRAINT_FORCE = 10000.0
GRIPPER_JOINT_DAMPING = 0.20
GRIPPER_LINK_ANGULAR_DAMPING = 0.20
FINGER_LATERAL_FRICTION = 5.0
FINGER_SPINNING_FRICTION = 0.5
FINGER_CONTACT_STIFFNESS = 100000.0
FINGER_CONTACT_DAMPING = 1000.0
GRASP_ATTACHMENT_MAX_FORCE = 10000.0
GRASP_ATTACHMENT_ERP = 1.0

# PyBullet does not execute the URDF/Gazebo <mimic> plugins. Drive every
# movable finger joint from the same master angle so the linkage cannot sag.
GRIPPER_JOINT_MULTIPLIERS = {
    "finger_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": -1.0,
}

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

    def __init__(
        self,
        robot_base_position=(-0.05, 0.0, 0.0),
        planner: MoveItJakaPlanner | None = None,
        initial_joint_pose_deg=None,
        robot_base_yaw_deg: float = 0.0,
        gui_motion_step_delay: float = 0.0,
    ):
        self.robot_base_position = np.asarray(robot_base_position, dtype=float)
        self.robot_base_yaw_deg = float(robot_base_yaw_deg)
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
        self.left_finger_link_ids: set[int] = set()
        self.right_finger_link_ids: set[int] = set()
        self._current_opening = 0.06
        self.initial_joint_pose_deg = (
            [float(value) for value in initial_joint_pose_deg]
            if initial_joint_pose_deg is not None
            else list(np.rad2deg(DEFAULT_JAKA_JOINTS))
        )
        if len(self.initial_joint_pose_deg) != len(JAKA_JOINT_NAMES):
            raise ValueError("initial_joint_pose_deg must contain exactly 6 values")
        self.robot_joint_values = list(np.deg2rad(self.initial_joint_pose_deg))
        self.gui_motion_step_delay = max(0.0, float(gui_motion_step_delay))
        self.last_motion_plan: dict | None = None
        self.last_motion_error: str | None = None

    def load(self, position=(0.3, 0.0, 0.3), orientation=(0, 0, 0, 1)):
        self._load_robot()
        self._load_robotiq()
        return self.base_id

    def remove(self):
        self.release_grasp()
        for constraint_id in self._mimic_constraints:
            try:
                p.removeConstraint(constraint_id)
            except p.error:
                pass
        self._mimic_constraints.clear()
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
        tcp_pos, _ = self.get_tcp_pose()
        return float(np.linalg.norm(np.asarray(tcp_pos, dtype=float) - grasp_pos))

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

    def get_push_contact_link_ids(self) -> set[int]:
        """Return gripper links that are allowed to establish push contact."""
        link_ids = set(self.left_finger_link_ids)
        link_ids.update(self.right_finger_link_ids)
        if self.ee_tip_id is not None:
            link_ids.add(int(self.ee_tip_id))
        return link_ids

    def move_to_joint_pose_deg(self, joint_pose_deg):
        """Move the arm to an explicit six-joint target expressed in degrees."""
        joint_pose_deg = [float(value) for value in joint_pose_deg]
        if len(joint_pose_deg) != len(JAKA_ARM_JOINTS):
            raise ValueError("joint_pose_deg must contain exactly 6 values")
        self._move_joints_smooth(np.deg2rad(joint_pose_deg))
        return list(self.robot_joint_values)

    def move_to_place_joint_pose_deg(self, joint_pose_deg):
        """Transport a grasped object with a slower, acceleration-limited path."""
        joint_pose_deg = [float(value) for value in joint_pose_deg]
        if len(joint_pose_deg) != len(JAKA_ARM_JOINTS):
            raise ValueError("joint_pose_deg must contain exactly 6 values")
        self._move_joints_smooth(
            np.deg2rad(joint_pose_deg),
            max_joint_step=TRANSPORT_JOINT_MOVE_STEP,
            min_steps=TRANSPORT_JOINT_MOVE_MIN_STEPS,
            max_steps=TRANSPORT_JOINT_MOVE_MAX_STEPS,
            smooth_acceleration=True,
            gui_delay_scale=TRANSPORT_GUI_DELAY_SCALE,
            rigid_transport_attachment=True,
        )
        return list(self.robot_joint_values)

    def close_fingers(self, target_width: float, steps: int = 100, object_id=None):
        del object_id
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
        parent_link = self.ee_tip_id if self.robot_id == self.robotiq_id else -1
        if parent_link >= 0:
            gripper_pos, gripper_orn = _link_frame_pose(self.robotiq_id, parent_link)
        else:
            gripper_pos, gripper_orn = p.getBasePositionAndOrientation(self.robotiq_id)
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
        inv_gripper_pos, inv_gripper_orn = p.invertTransform(gripper_pos, gripper_orn)
        parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
            inv_gripper_pos, inv_gripper_orn, obj_pos, obj_orn
        )
        self.grasp_constraint = p.createConstraint(
            self.robotiq_id,
            parent_link,
            object_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            parent_frame_pos,
            [0, 0, 0],
            parentFrameOrientation=parent_frame_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )
        p.changeConstraint(
            self.grasp_constraint,
            maxForce=GRASP_ATTACHMENT_MAX_FORCE,
            erp=GRASP_ATTACHMENT_ERP,
        )
        return self.grasp_constraint

    def has_bilateral_finger_contact(self, object_id):
        contacts = p.getContactPoints(bodyA=self.robotiq_id, bodyB=object_id)
        contacted_links = {int(contact[3]) for contact in contacts}
        left_contact = bool(contacted_links & self.left_finger_link_ids)
        right_contact = bool(contacted_links & self.right_finger_link_ids)
        return left_contact and right_contact

    def finger_contact_links(self, object_id):
        contacts = p.getContactPoints(bodyA=self.robotiq_id, bodyB=object_id)
        return sorted({int(contact[3]) for contact in contacts})

    def finger_link_positions(self):
        def positions(link_ids):
            return [list(_link_frame_pose(self.robotiq_id, link_id)[0]) for link_id in sorted(link_ids)]

        return {
            "left": positions(self.left_finger_link_ids),
            "right": positions(self.right_finger_link_ids),
        }

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

    def set_collision_with_objects(self, object_ids, enabled):
        """Toggle collisions between all robot links and scene objects."""
        for object_id in dict.fromkeys(int(value) for value in object_ids):
            self.set_collision_with_object(object_id, enabled=enabled)

    def set_collision_with_object(self, object_id, enabled):
        """Enable or disable collisions between every robot link and an object."""
        enabled_flag = 1 if enabled else 0
        for body_id in self.collision_body_ids():
            if body_id is None:
                continue
            for link_index in range(-1, p.getNumJoints(body_id)):
                p.setCollisionFilterPair(
                    bodyUniqueIdA=body_id,
                    bodyUniqueIdB=object_id,
                    linkIndexA=link_index,
                    linkIndexB=-1,
                    enableCollision=enabled_flag,
                )

    def has_contact_with_object(self, object_id):
        return any(
            p.getContactPoints(bodyA=body_id, bodyB=object_id)
            for body_id in self.collision_body_ids()
            if body_id is not None
        )

    def max_penetration_depth(self, object_id):
        penetration_depth = 0.0
        for body_id in self.collision_body_ids():
            if body_id is None:
                continue
            for contact in p.getContactPoints(bodyA=body_id, bodyB=object_id):
                contact_distance = float(contact[8])
                if contact_distance < 0.0:
                    penetration_depth = max(penetration_depth, -contact_distance)
        return penetration_depth

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
            "jaka_base_yaw_deg": self.robot_base_yaw_deg,
            "jaka_joint_values": list(self.robot_joint_values),
            "initial_joint_pose_deg": list(self.initial_joint_pose_deg),
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
                self._gui_motion_pause()
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
            baseOrientation=p.getQuaternionFromEuler(
                [0.0, 0.0, np.deg2rad(self.robot_base_yaw_deg)]
            ),
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
            if "robotiq_85_left_" in link_name and any(
                part in link_name for part in ("finger", "knuckle")
            ):
                self.left_finger_link_ids.add(joint_idx)
            if "robotiq_85_right_" in link_name and any(
                part in link_name for part in ("finger", "knuckle")
            ):
                self.right_finger_link_ids.add(joint_idx)
            if "finger_tip" in joint_name or "finger_tip" in link_name:
                self.ee_finger_pad_ids.append(joint_idx)
                p.changeDynamics(
                    self.robot_id,
                    joint_idx,
                    lateralFriction=FINGER_LATERAL_FRICTION,
                    spinningFriction=FINGER_SPINNING_FRICTION,
                    rollingFriction=0.001,
                    restitution=0.0,
                    contactStiffness=FINGER_CONTACT_STIFFNESS,
                    contactDamping=FINGER_CONTACT_DAMPING,
                    frictionAnchor=1,
                )
            if "robotiq_85" in joint_name and joint_type == p.JOINT_REVOLUTE:
                p.changeDynamics(
                    self.robot_id,
                    joint_idx,
                    jointDamping=GRIPPER_JOINT_DAMPING,
                    angularDamping=GRIPPER_LINK_ANGULAR_DAMPING,
                )
                p.setJointMotorControl2(
                    self.robot_id, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0
                )
        self._create_mimic_constraints()
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
        rest_candidates = [
            original,
            list(DEFAULT_JAKA_JOINTS),
            [0.0, 0.4, -0.2, -1.7, -1.57, 0.0],
            [0.0, -0.3, 2.0, -0.1, 1.57, 1.2],
            [-0.4, -0.3, 2.0, -0.1, 1.57, 1.2],
        ]
        tcp_target = np.asarray(tcp_target if tcp_target is not None else target_pos, dtype=float)
        best_solution = None
        best_error = float("inf")
        saved_state = p.saveState()
        try:
            for rest_poses in rest_candidates:
                solution = p.calculateInverseKinematics(
                    self.robot_id,
                    JAKA_EE_LINK,
                    target_pos.tolist(),
                    target_orn.tolist(),
                    lowerLimits=lower_limits,
                    upperLimits=upper_limits,
                    jointRanges=joint_ranges,
                    restPoses=rest_poses,
                    maxNumIterations=2000,
                    residualThreshold=1e-6,
                )
                candidate = solution[: len(JAKA_ARM_JOINTS)]
                for joint_idx, joint_value in zip(JAKA_ARM_JOINTS, candidate):
                    p.resetJointState(self.robot_id, joint_idx, joint_value)
                ee_pos, ee_orn = _link_frame_pose(self.robot_id, JAKA_EE_LINK)
                tcp_pos, _ = _tip_pose_from_ee(ee_pos, ee_orn)
                position_error = float(np.linalg.norm(tcp_pos - tcp_target))
                orientation_error = 1.0 - abs(float(np.dot(np.asarray(ee_orn), target_orn)))
                continuity_error = _joint_delta_norm(candidate, original)
                error = position_error + 0.02 * orientation_error + IK_CONTINUITY_WEIGHT * continuity_error
                if error < best_error:
                    best_error = error
                    best_solution = candidate
                p.restoreState(saved_state)
        finally:
            p.restoreState(saved_state)
            p.removeState(saved_state)

        if best_solution is not None:
            self._move_joints_smooth(best_solution)
        self.last_motion_error = None if best_solution is not None else "No IK solution"
        self.robot_joint_values = [p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS]

    def _move_joints_smooth(
        self,
        target_joints,
        max_joint_step: float = JOINT_MOVE_STEP,
        min_steps: int = JOINT_MOVE_MIN_STEPS,
        max_steps: int = JOINT_MOVE_MAX_STEPS,
        smooth_acceleration: bool = False,
        gui_delay_scale: float = 1.0,
        rigid_transport_attachment: bool = False,
    ):
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
        steps = int(np.clip(np.ceil(max_delta / max_joint_step), min_steps, max_steps))
        for step in range(steps):
            frac = (step + 1) / steps
            if smooth_acceleration:
                # Quintic smootherstep has zero velocity and acceleration at
                # both ends, reducing inertial shocks on a grasped object.
                frac = frac ** 3 * (frac * (frac * 6.0 - 15.0) + 10.0)
            waypoint = current + delta * frac
            for joint_idx, joint_value in zip(JAKA_ARM_JOINTS, waypoint):
                p.resetJointState(self.robot_id, joint_idx, float(joint_value))
            self.robot_joint_values = waypoint.tolist()
            if rigid_transport_attachment:
                self._sync_transport_attachment()
            else:
                self._sync_attachment()
            p.stepSimulation()
            if rigid_transport_attachment:
                self._sync_transport_attachment()
            self._gui_motion_pause(gui_delay_scale)
        self.robot_joint_values = [p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS]
        self._hold_current_joints()
        if rigid_transport_attachment:
            self._sync_transport_attachment()

    def _hold_current_joints(self):
        if self.robot_id is None:
            return
        if not self.robot_joint_values:
            self.robot_joint_values = [p.getJointState(self.robot_id, j)[0] for j in JAKA_ARM_JOINTS]
        # The combined JAKA/Robotiq URDF has incomplete inertial data. Finger
        # constraints otherwise kick the arm away from the IK pose during each
        # simulation step. Keep the arm kinematic while leaving finger/object
        # contacts dynamic.
        for joint_idx, joint_value in zip(JAKA_ARM_JOINTS, self.robot_joint_values):
            p.resetJointState(
                self.robot_id,
                joint_idx,
                targetValue=float(joint_value),
                targetVelocity=0.0,
            )
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=JAKA_ARM_JOINTS,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(self.robot_joint_values),
            positionGains=[1.0] * len(JAKA_ARM_JOINTS),
            forces=[JAKA_HOLD_FORCE] * len(JAKA_ARM_JOINTS),
        )

    def _gui_motion_pause(self, delay_scale: float = 1.0):
        if self.gui_motion_step_delay > 0.0:
            time.sleep(self.gui_motion_step_delay * max(0.0, float(delay_scale)))

    @staticmethod
    def _grasp_pose_to_robotiq_base(position, rotation_matrix):
        grasp_pos = np.asarray(position, dtype=float)
        grasp_rot = np.asarray(rotation_matrix, dtype=float)
        approach_axis = grasp_rot[:, 0]
        opening_axis = grasp_rot[:, 1]
        approach_axis = approach_axis / max(np.linalg.norm(approach_axis), 1e-8)
        opening_axis = opening_axis - np.dot(opening_axis, approach_axis) * approach_axis
        opening_axis = opening_axis / max(np.linalg.norm(opening_axis), 1e-8)

        robotiq_x = approach_axis
        robotiq_y = opening_axis
        robotiq_z = np.cross(robotiq_x, robotiq_y)
        robotiq_z = robotiq_z / max(np.linalg.norm(robotiq_z), 1e-8)
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

    def _sync_transport_attachment(self):
        """Keep the end effector aligned during grasped-object transport."""
        self._sync_attachment()

    def _controlled_gripper_joints(self):
        """Return unique (joint id, multiplier) pairs for the six-bar linkage."""
        controlled = []
        seen = set()
        for name, multiplier in GRIPPER_JOINT_MULTIPLIERS.items():
            joint_id = self._robotiq_joint_map.get(name)
            if joint_id is None or joint_id in seen:
                continue
            seen.add(joint_id)
            controlled.append((joint_id, multiplier))
        return controlled

    def _create_mimic_constraints(self):
        """Enforce the URDF mimic relationships that PyBullet does not load."""
        master_joint = self._robotiq_joint_map.get("finger_joint")
        if master_joint is None:
            return
        for joint_id, multiplier in self._controlled_gripper_joints():
            if joint_id == master_joint:
                continue
            constraint_id = p.createConstraint(
                parentBodyUniqueId=self.robotiq_id,
                parentLinkIndex=master_joint,
                childBodyUniqueId=self.robotiq_id,
                childLinkIndex=joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0.0, 0.0, 1.0],
                parentFramePosition=[0.0, 0.0, 0.0],
                childFramePosition=[0.0, 0.0, 0.0],
            )
            p.changeConstraint(
                constraint_id,
                gearRatio=-float(multiplier),
                maxForce=GRIPPER_MIMIC_CONSTRAINT_FORCE,
                erp=0.8,
            )
            self._mimic_constraints.append(constraint_id)

    def _command_gripper_angle(self, joint_value: float, step_count: int = 10):
        if self.robotiq_id is None:
            return
        main_joint = self._robotiq_joint_map.get("finger_joint")
        if main_joint is None:
            return
        p.setJointMotorControl2(
            self.robotiq_id,
            main_joint,
            p.POSITION_CONTROL,
            targetPosition=joint_value,
            positionGain=1.0,
            velocityGain=1.0,
            force=GRIPPER_MOTOR_FORCE,
        )
        for _ in range(step_count):
            self._hold_current_joints()
            p.stepSimulation()

    def _move_gripper_angle(self, target_angle: float, timeout: float = 3.0, is_slow: bool = False):
        main_joint = self._robotiq_joint_map.get("finger_joint")
        if main_joint is None:
            return

        if is_slow:
            previous_angle = p.getJointState(self.robotiq_id, main_joint)[0]
            direction = 1.0 if target_angle > previous_angle else -1.0
            p.setJointMotorControl2(
                self.robotiq_id,
                main_joint,
                p.VELOCITY_CONTROL,
                targetVelocity=direction,
                maxVelocity=1.0,
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
