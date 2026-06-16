"""
平行二指夹爪模块 — Phase 2 约束力版。

方案：夹爪基座位姿控制 + 手指位置显示用，
物理夹持效果通过 constraint 施加——当判定抓取成功时，
在物体和夹爪基座之间创建一个 fixed constraint，
提升基座时物体会跟随上升（模拟抓取效果）。

判定逻辑：夹爪基座提升后，检查物体是否跟随（Z是否升高）。
"""

import tempfile
from pathlib import Path

import pybullet as p
import numpy as np


class ParallelJawGripper:
    FINGER_LENGTH = 0.10
    FINGER_WIDTH  = 0.012
    FINGER_HEIGHT = 0.03
    BASE_SIZE = 0.03
    BASE_WIDTH = 0.144

    def __init__(self, visual_alpha: float = 1.0):
        self.base_id = None
        self.left_id = None
        self.right_id = None
        self.visual_alpha = float(visual_alpha)
        self._max_opening = 0.12
        self._current_opening = 0.10
        self.grasp_constraint = None

    def load(self, position=(0.3, 0.0, 0.3), orientation=(0, 0, 0, 1)):
        base_half_extents = [
            self.BASE_SIZE / 2,
            self.BASE_WIDTH / 2,
            self.FINGER_HEIGHT / 2,
        ]
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half_extents,
                                        rgbaColor=[0.4, 0.4, 0.4, self.visual_alpha])
        base_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=base_half_extents)
        self.base_id = p.createMultiBody(0.0, base_col, base_vis, position, orientation)

        finger_vis = p.createVisualShape(p.GEOM_BOX,
            halfExtents=[self.FINGER_LENGTH/2, self.FINGER_WIDTH/2, self.FINGER_HEIGHT/2],
            rgbaColor=[0.2, 0.6, 0.8, self.visual_alpha])
        finger_col = p.createCollisionShape(p.GEOM_BOX,
            halfExtents=[self.FINGER_LENGTH/2, self.FINGER_WIDTH/2, self.FINGER_HEIGHT/2])

        # Fingers are kinematic collision bodies. Their poses are controlled
        # explicitly so gravity and contact cannot make the gripper fall apart.
        self.left_id = p.createMultiBody(
            0.0, finger_col, finger_vis, position, orientation)
        self.right_id = p.createMultiBody(
            0.0, finger_col, finger_vis, position, orientation)
        self._sync_fingers()
        return self.base_id

    def remove(self):
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
        for obj in [self.left_id, self.right_id, self.base_id]:
            if obj is not None: p.removeBody(obj)

    def set_pose(self, position, rotation_matrix):
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(rotation_matrix).as_quat()
        p.resetBasePositionAndOrientation(self.base_id, position.tolist(),
                                          [quat[0], quat[1], quat[2], quat[3]])
        self._sync_fingers()

    def _sync_fingers(self):
        pos, orn = p.getBasePositionAndOrientation(self.base_id)
        rot_m = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        forward = self.BASE_SIZE / 2.0 + self.FINGER_LENGTH / 2.0
        half = self._current_opening / 2.0 + self.FINGER_WIDTH / 2.0
        lp = np.array(pos) + rot_m @ np.array([forward, -half, 0.0])
        rp = np.array(pos) + rot_m @ np.array([forward, half, 0.0])
        p.resetBasePositionAndOrientation(self.left_id, lp.tolist(), orn)
        p.resetBasePositionAndOrientation(self.right_id, rp.tolist(), orn)

    def set_opening(self, width: float):
        self._current_opening = min(max(width, 0.002), self._max_opening)

    def close_fingers(self, target_width: float, steps: int = 100):
        """手指显示性闭合（不产生物理力）。"""
        target_width = max(target_width, 0.002)
        start = self._current_opening
        for i in range(steps):
            frac = (i + 1) / steps
            self._current_opening = start + (target_width - start) * frac
            self._sync_fingers()
            p.stepSimulation()
        self._current_opening = target_width

    def create_grasp_constraint(self, object_id):
        """创建物体到夹爪基座的固定约束。"""
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
        # parentFrame/childFrame 必须是局部坐标，用 [0,0,0]
        self.grasp_constraint = p.createConstraint(
            self.base_id, -1, object_id, -1,
            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        return self.grasp_constraint

    def release_grasp(self):
        """释放抓取约束。"""
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None


class JakaZu3VisualGripper(ParallelJawGripper):
    """Current contact gripper plus a JAKA ZU3 + Robotiq visual/IK model.

    The lightweight box gripper is still used for stable push contact. The
    JAKA model is kinematic and collision-disabled so it does not change the
    existing evaluator's success/failure semantics.
    """

    DEFAULT_URDF = (
        "/home/admin128/Desktop/liboyan/Trans_MP/lby_moveit/src/"
        "robotiq_test/config/gazebo_jaka_zu3_robotiq.urdf"
    )
    PACKAGE_MAP = {
        "package://jaka_description/meshes/jaka_zu3_meshes": (
            "/home/admin128/JunyuFan/phantom/submodules/phantom-robosuite/"
            "robosuite/models/assets/robots/jaka_zu3/meshes"
        ),
        "package://jaka_rviz/meshes/jaka_zu3_meshes": (
            "/home/admin128/JunyuFan/phantom/submodules/phantom-robosuite/"
            "robosuite/models/assets/robots/jaka_zu3/meshes"
        ),
        "package://robotiq_description/meshes": (
            "/home/admin128/Desktop/liboyan/Trans_MP/lby_moveit/src/"
            "robotiq_new/robotiq_description/meshes"
        ),
    }
    DEFAULT_HOME = [0.0, 1.5708, 0.0, 0.0, 1.5708, 0.7854]

    def __init__(
        self,
        robot_urdf: str | None = None,
        robot_base_position=(-0.10, -0.35, 0.0),
        robot_base_orientation=(0.0, 0.0, 0.0, 1.0),
    ):
        super().__init__(visual_alpha=0.08)
        self.robot_urdf = robot_urdf or self.DEFAULT_URDF
        self.robot_base_position = robot_base_position
        self.robot_base_orientation = robot_base_orientation
        self.robot_id = None
        self.robot_urdf_resolved = None
        self.arm_joint_indices = []
        self.ee_link_index = None
        self.link_indices_for_log = []

    def load(self, position=(0.3, 0.0, 0.3), orientation=(0, 0, 0, 1)):
        base_id = super().load(position, orientation)
        self._load_robot()
        return base_id

    def remove(self):
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
            self.robot_id = None
        if self.robot_urdf_resolved is not None:
            try:
                Path(self.robot_urdf_resolved).unlink(missing_ok=True)
            except OSError:
                pass
            self.robot_urdf_resolved = None
        super().remove()

    def set_pose(self, position, rotation_matrix):
        super().set_pose(position, rotation_matrix)
        self._sync_robot_to_tcp(position, rotation_matrix)

    def set_opening(self, width: float):
        super().set_opening(width)
        self._sync_gripper_joints()

    def close_fingers(self, target_width: float, steps: int = 100):
        super().close_fingers(target_width, steps)
        self._sync_gripper_joints()

    def _load_robot(self):
        urdf = self._resolve_urdf()
        if urdf is None:
            print("[JAKA] URDF or meshes not found; using simple gripper only.")
            return
        self.robot_id = p.loadURDF(
            urdf,
            self.robot_base_position,
            self.robot_base_orientation,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        )
        self._index_robot()
        for link_idx in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterGroupMask(self.robot_id, link_idx, 0, 0)
        for joint_idx, value in zip(self.arm_joint_indices, self.DEFAULT_HOME):
            p.resetJointState(self.robot_id, joint_idx, value)
        self._sync_gripper_joints()

    def _resolve_urdf(self):
        source = Path(self.robot_urdf)
        if not source.is_file():
            return None
        text = source.read_text(encoding="utf-8")
        for package_uri, real_path in self.PACKAGE_MAP.items():
            if package_uri in text and not Path(real_path).is_dir():
                return None
            text = text.replace(package_uri, real_path)
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="jaka_zu3_robotiq_",
            suffix=".urdf",
            delete=False,
        )
        with tmp:
            tmp.write(text)
        self.robot_urdf_resolved = tmp.name
        return tmp.name

    def _index_robot(self):
        self.arm_joint_indices = []
        self.ee_link_index = None
        self.link_indices_for_log = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            if joint_name in {f"joint_{j}" for j in range(1, 7)}:
                self.arm_joint_indices.append(i)
            if link_name in {"Link_0", "Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"}:
                self.link_indices_for_log.append(i)
            if link_name == "robotiq_85_base_link":
                self.ee_link_index = i
        if self.ee_link_index is None:
            self.ee_link_index = self.arm_joint_indices[-1] if self.arm_joint_indices else None

    def _sync_robot_to_tcp(self, position, rotation_matrix):
        if self.robot_id is None or self.ee_link_index is None:
            return
        solution = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            np.asarray(position, dtype=float).tolist(),
            maxNumIterations=80,
            residualThreshold=1e-4,
        )
        for joint_idx, joint_value in zip(self.arm_joint_indices, solution):
            p.resetJointState(self.robot_id, joint_idx, joint_value)
        self._sync_gripper_joints()

    def _sync_gripper_joints(self):
        if self.robot_id is None:
            return
        opening_ratio = 1.0 - min(max(self._current_opening / self._max_opening, 0.0), 1.0)
        knuckle = opening_ratio * 0.70
        for i in range(p.getNumJoints(self.robot_id)):
            name = p.getJointInfo(self.robot_id, i)[1].decode("utf-8")
            if name == "robotiq_85_left_knuckle_joint":
                p.resetJointState(self.robot_id, i, knuckle)
            elif "robotiq_85" in name and p.getJointInfo(self.robot_id, i)[2] == p.JOINT_REVOLUTE:
                p.resetJointState(self.robot_id, i, knuckle)

    def snapshot_extra(self):
        if self.robot_id is None:
            return {}
        links = []
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        links.append({"name": "base", "pos": list(base_pos), "orn": list(base_orn)})
        for idx in self.link_indices_for_log:
            state = p.getLinkState(self.robot_id, idx, computeForwardKinematics=True)
            name = p.getJointInfo(self.robot_id, idx)[12].decode("utf-8")
            links.append({"name": name, "pos": list(state[4]), "orn": list(state[5])})
        if self.ee_link_index is not None:
            state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
            links.append({"name": "tcp", "pos": list(state[4]), "orn": list(state[5])})
        return {
            "robot_model": "jaka_zu3_robotiq",
            "robot_links": links,
        }
