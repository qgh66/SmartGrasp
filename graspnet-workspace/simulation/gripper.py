"""
平行二指夹爪模块 — Phase 2 约束力版。

方案：夹爪基座位姿控制 + 手指位置显示用，
物理夹持效果通过 constraint 施加——当判定抓取成功时，
在物体和夹爪基座之间创建一个 fixed constraint，
提升基座时物体会跟随上升（模拟抓取效果）。

判定逻辑：夹爪基座提升后，检查物体是否跟随（Z是否升高）。
"""

import pybullet as p
import numpy as np


class ParallelJawGripper:
    FINGER_LENGTH = 0.10
    FINGER_WIDTH  = 0.012
    FINGER_HEIGHT = 0.03
    BASE_SIZE = 0.03

    def __init__(self):
        self.base_id = None
        self.left_id = None
        self.right_id = None
        self._max_opening = 0.12
        self._current_opening = 0.10
        self.grasp_constraint = None

    def load(self, position=(0.3, 0.0, 0.3), orientation=(0, 0, 0, 1)):
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.BASE_SIZE/2]*3,
                                        rgbaColor=[0.4, 0.4, 0.4, 1])
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.BASE_SIZE/2]*3)
        self.base_id = p.createMultiBody(0.0, base_col, base_vis, position, orientation)

        finger_vis = p.createVisualShape(p.GEOM_BOX,
            halfExtents=[self.FINGER_LENGTH/2, self.FINGER_WIDTH/2, self.FINGER_HEIGHT/2],
            rgbaColor=[0.2, 0.6, 0.8, 1])
        finger_col = p.createCollisionShape(p.GEOM_BOX,
            halfExtents=[self.FINGER_LENGTH/2, self.FINGER_WIDTH/2, self.FINGER_HEIGHT/2])

        half = self._current_opening / 2.0
        self.left_id = p.createMultiBody(0.05, finger_col, finger_vis,
                                          [position[0], position[1]-half, position[2]])
        self.right_id = p.createMultiBody(0.05, finger_col, finger_vis,
                                           [position[0], position[1]+half, position[2]])
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
        half = self._current_opening / 2.0
        lp = np.array(pos) + rot_m @ np.array([0.0, -half, 0.0])
        rp = np.array(pos) + rot_m @ np.array([0.0, half, 0.0])
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
