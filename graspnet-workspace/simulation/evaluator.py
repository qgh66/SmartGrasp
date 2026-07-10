"""
抓取执行与物理评估模块（逐帧轨迹版，JAKA Zu3 + Robotiq-85）。

职责：
- 逐一对 GraspGroup 中的候选抓取执行物理仿真
- **每个仿真步记录物体+夹爪完整位姿**（用于 Dash GUI 真实场景回放）
- 判定抓取是否成功

抓取流程参照 ThinkGrasp/UR5e 风格的 grasp() 原语，但执行体换成 JAKA Zu3
机械臂 + Robotiq-85 夹爪（见 robot_gripper.py）：
  张开 → 移到目标上方 over 点 → 沿 approach 方向直线下插 → 欠驱动闭合（真实摩擦
  夹持，不再用固定约束吸附）→ 直线抬回 → 用夹爪关节角 is_gripper_closed()
  判定是否真夹到实体。
"""

import sys
import os
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "graspnetAPI"))
from graspnetAPI.grasp import GraspGroup
from .robot_gripper import JakaZu3Robotiq85Gripper

TABLE_Z = 0.0
TABLE_CLEARANCE = 0.005
MAX_GRASP_CENTER_DIST = 0.04
OBJECT_POINT_Z_THRESHOLD = 0.005
MIN_EXECUTION_CENTER_Z = TABLE_Z + TABLE_CLEARANCE


def _snapshot(obj_id, gripper):
    """拍摄一帧场景状态。"""
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
    base_pos, base_orn = p.getBasePositionAndOrientation(gripper.base_id)
    left_pos, _ = p.getBasePositionAndOrientation(gripper.left_id)
    right_pos, _ = p.getBasePositionAndOrientation(gripper.right_id)
    return {
        'obj_pos': list(obj_pos), 'obj_orn': list(obj_orn),
        'gripper_pos': list(base_pos), 'gripper_orn': list(base_orn),
        'left_pos': list(left_pos), 'right_pos': list(right_pos),
        'opening': gripper._current_opening,
        }


def _unit(vec, fallback):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=float)
    return np.asarray(vec, dtype=float) / norm


def _object_waist_z(object_points: np.ndarray | None) -> float:
    """Estimate a stable mid-body grasp height from visible object points."""
    if object_points is None or len(object_points) == 0:
        return MIN_EXECUTION_CENTER_Z
    z_values = np.asarray(object_points[:, 2], dtype=float)
    z_low, z_high = np.percentile(z_values, [10, 90])
    waist_z = 0.5 * (float(z_low) + float(z_high))
    return max(waist_z, MIN_EXECUTION_CENTER_Z)


class GraspEvaluator:
    """基于 PyBullet 的抓取物理评估器（逐帧轨迹版）。"""

    def __init__(self, object_id: int, gripper: JakaZu3Robotiq85Gripper,
                 point_cloud: np.ndarray = None, gui: bool = False):
        self.object_id = object_id
        self.gripper = gripper
        self.gui = gui
        if point_cloud is not None:
            self.obj_pts = point_cloud[point_cloud[:, 2] > 0.005]
        else:
            self.obj_pts = None
        self.object_waist_z = _object_waist_z(self.obj_pts)

    def _gui_step(self, n: int = 1, sleep: float = 0.0):
        for _ in range(n):
            p.stepSimulation()
        if self.gui and sleep > 0:
            time.sleep(sleep)

    def evaluate(self, grasp_group: GraspGroup, top_k: int = 10,
                 approach_depth_offset: float = 0.05,
                 lift_height: float = 0.20, lift_steps: int = 200):
        grasp_group.sort_by_score()
        gg_top = grasp_group[:min(top_k, len(grasp_group))]
        results = []
        for idx, grasp in enumerate(gg_top):
            frame_log = []
            res = self._execute(grasp, approach_depth_offset, lift_height, lift_steps, frame_log)
            res['grasp_index'] = idx
            res['frame_log'] = frame_log
            results.append(res)
            self.gripper.release_grasp()
            self._reset_object()
        return results

    def _execute(self, grasp, approach_depth_offset, lift_height, lift_steps, frame_log):
        center = grasp.translation.copy()
        raw_center = center.copy()
        rot_m = grasp.rotation_matrix.copy()
        width = grasp.width
        depth = grasp.depth

        # GraspNet frequently proposes centers on the support plane for cropped
        # tabletop point clouds. Execute at the object's visible waist instead of
        # driving the TCP into z=0, while keeping the original center for debug.
        center[2] = self.object_waist_z

        # GraspNet local X 轴作为 approach/depth 方向
        approach_dir = _unit(rot_m[:, 0], [0, 0, -1])
        # 预抓取点：沿 approach 反方向退出 depth + offset
        pre_grasp_pos = center - approach_dir * (depth + approach_depth_offset)
        # over 点：在预抓取点正上方 0.2m，对齐参考 grasp() 的“先到目标上方”
        over_pos = pre_grasp_pos + np.array([0.0, 0.0, 0.20])

        if self.obj_pts is not None and len(self.obj_pts) > 0:
            center_dist = float(np.linalg.norm(self.obj_pts - center, axis=1).min())
            if center_dist > MAX_GRASP_CENTER_DIST:
                obj_pos_before, _ = p.getBasePositionAndOrientation(self.object_id)
                return {'success': False, 'lift_z': obj_pos_before[2],
                        'score': grasp.score, 'translation': center,
                        'rotation': rot_m, 'width': width, 'depth': depth,
                        'failure_reason': 'grasp_center_not_on_object',
                        'center_object_dist': center_dist,
                        'raw_translation': raw_center,
                        'execution_center_z': float(center[2]),
                        'object_waist_z': float(self.object_waist_z)}

        # 注：已按需求取消抓取中心/预抓取点的桌面高度限制（原 approach_below_table 护栏），
        # 允许 GraspNet 给出贴近桌面（z≈0）的抓取直接进入执行阶段。

        # 记录抓取前物体高度，用于 lift 后判定物体是否被真正抬起
        obj_z_before = float(p.getBasePositionAndOrientation(self.object_id)[0][2])

        # Step 1: 张开夹爪并移动到目标上方 over 点（参考 grasp(): open + move over）
        self.gripper.set_opening(0.085)
        self.gripper.set_pose(over_pos, rot_m)
        self._gui_step(10, sleep=0.02)
        frame_log.append({'phase': 'approach', 'step': 'over', **_snapshot(self.object_id, self.gripper)})

        # 从 over 点直线下插到预抓取点，再沿 approach 方向插值推进到抓取中心。
        # JAKA 夹爪无逐步力检测原语，这里用插值 waypoint 反复 set_pose 模拟直线运动。
        descend_start = over_pos
        for i in range(10):
            frac = (i + 1) / 10
            pos = descend_start + (pre_grasp_pos - descend_start) * frac
            self.gripper.set_pose(pos, rot_m)
            self._gui_step(2, sleep=0.005)
            frame_log.append({'phase': 'approach', 'step': f'descend_{i}',
                              **_snapshot(self.object_id, self.gripper)})

        # Step 2: 沿 approach 方向直线推进到抓取中心
        for i in range(20):
            frac = (i + 1) / 20
            pos = pre_grasp_pos + approach_dir * (depth + approach_depth_offset) * frac
            self.gripper.set_pose(pos, rot_m)
            self._gui_step(2, sleep=0.005)
            frame_log.append({'phase': 'approach', 'step': i, **_snapshot(self.object_id, self.gripper)})

        # Step 3: 欠驱动闭合，靠 Robotiq-85 真实摩擦夹持（不再创建固定约束吸附）
        self.gripper.close_fingers(target_width=width, steps=30)
        self._gui_step(20, sleep=0.01)
        frame_log.append({'phase': 'close', 'step': 'done', **_snapshot(self.object_id, self.gripper)})

        # Step 4: 直线抬升
        final_pos = center + np.array([0, 0, lift_height])
        start_pos = self._get_gripper_pos()
        for i in range(30):
            frac = (i + 1) / 30
            pos = start_pos + (final_pos - start_pos) * frac
            self.gripper.set_pose(pos, rot_m)
            self._gui_step(6, sleep=0.003)
            frame_log.append({'phase': 'lift', 'step': i, **_snapshot(self.object_id, self.gripper)})

        # Step 5: 判定（完全照参考 environment_sim.grasp）——夹爪关节角未完全合死即视为
        # 夹到实体；物体能否跟随上升是物理自然结果，不作为额外判据。
        grasped = bool(self.gripper.is_gripper_closed())
        obj_pos_after = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
        obj_z_after = float(obj_pos_after[2])
        obj_lift_delta = obj_z_after - float(obj_z_before)
        success = grasped
        frame_log.append({'phase': 'done', 'step': 'final', 'success': success,
                          **_snapshot(self.object_id, self.gripper)})

        if self.gui:
            p.removeAllUserDebugItems()
            p.addUserDebugText('SUCCESS' if success else 'FAILED', final_pos,
                               [0, 1, 0] if success else [1, 0, 0], 2.5, lifeTime=0)
            time.sleep(1.0)

        return {'success': success, 'lift_z': obj_z_after,
                'score': grasp.score, 'translation': center,
                'rotation': rot_m, 'width': width, 'depth': depth,
                # 判定依据：grasped_by_gripper（夹爪关节角）。obj_lift_delta 仅供诊断对照。
                'grasped_by_gripper': grasped,
                'obj_z_before': float(obj_z_before),
                'obj_z_after': obj_z_after,
                'obj_lift_delta': float(obj_lift_delta),
                'raw_translation': raw_center,
                'execution_center_z': float(center[2]),
                'object_waist_z': float(self.object_waist_z)}

    def _reset_object(self):
        pos, orn = p.getBasePositionAndOrientation(self.object_id)
        p.resetBasePositionAndOrientation(self.object_id, [pos[0], pos[1], 0.05], orn)
        for _ in range(150):
            p.stepSimulation()

    def _get_gripper_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.gripper.base_id)[0])
