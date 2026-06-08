"""
抓取执行与物理评估模块（逐帧轨迹版）。

职责：
- 逐一对 GraspGroup 中的候选抓取执行物理仿真
- **每个仿真步记录物体+夹爪完整位姿**（用于 Dash GUI 真实场景回放）
- 判定抓取是否成功
"""

import sys
import os
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "graspnetAPI"))
from graspnetAPI.grasp import GraspGroup
from .gripper import ParallelJawGripper

TABLE_Z = 0.0
TABLE_CLEARANCE = 0.005
MAX_GRASP_CENTER_DIST = 0.04


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


class GraspEvaluator:
    """基于 PyBullet 的抓取物理评估器（逐帧轨迹版）。"""

    def __init__(self, object_id: int, gripper: ParallelJawGripper,
                 point_cloud: np.ndarray = None, gui: bool = False):
        self.object_id = object_id
        self.gripper = gripper
        self.gui = gui
        if point_cloud is not None:
            self.obj_pts = point_cloud[point_cloud[:, 2] > 0.005]
        else:
            self.obj_pts = None

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
        rot_m = grasp.rotation_matrix.copy()
        width = grasp.width
        depth = grasp.depth
        approach_dir = _unit(rot_m[:, 0], [0, 0, -1])
        pre_grasp_pos = center - approach_dir * (depth + approach_depth_offset)

        if self.obj_pts is not None and len(self.obj_pts) > 0:
            center_dist = float(np.linalg.norm(self.obj_pts - center, axis=1).min())
            if center_dist > MAX_GRASP_CENTER_DIST:
                obj_pos_before, _ = p.getBasePositionAndOrientation(self.object_id)
                return {'success': False, 'lift_z': obj_pos_before[2],
                        'score': grasp.score, 'translation': center,
                        'rotation': rot_m, 'width': width, 'depth': depth,
                        'failure_reason': 'grasp_center_not_on_object',
                        'center_object_dist': center_dist}

        if min(float(center[2]), float(pre_grasp_pos[2])) < TABLE_Z + TABLE_CLEARANCE:
            obj_pos_before, _ = p.getBasePositionAndOrientation(self.object_id)
            return {'success': False, 'lift_z': obj_pos_before[2],
                    'score': grasp.score, 'translation': center,
                    'rotation': rot_m, 'width': width, 'depth': depth,
                    'failure_reason': 'approach_below_table',
                    'approach_min_z': min(float(center[2]), float(pre_grasp_pos[2]))}

        obj_pos_before = np.array(p.getBasePositionAndOrientation(self.object_id)[0])

        # Step 1: approach
        self.gripper.set_opening(0.06)
        self.gripper.set_pose(pre_grasp_pos, rot_m)
        self._gui_step(30, sleep=0.02)
        frame_log.append({'phase': 'approach', 'step': 'ready', **_snapshot(self.object_id, self.gripper)})

        # Step 2: forward
        for i in range(20):
            frac = (i + 1) / 20
            pos = pre_grasp_pos + approach_dir * (depth + approach_depth_offset) * frac
            self.gripper.set_pose(pos, rot_m)
            self._gui_step(2, sleep=0.005)
            frame_log.append({'phase': 'approach', 'step': i, **_snapshot(self.object_id, self.gripper)})

        # Step 3: close + constraint
        self.gripper.close_fingers(target_width=width, steps=30)
        self.gripper.create_grasp_constraint(self.object_id)
        self._gui_step(20, sleep=0.01)
        frame_log.append({'phase': 'close', 'step': 'done', **_snapshot(self.object_id, self.gripper)})

        # Step 4: lift
        final_pos = center + np.array([0, 0, lift_height])
        start_pos = self._get_gripper_pos()
        for i in range(30):
            frac = (i + 1) / 30
            pos = start_pos + (final_pos - start_pos) * frac
            self.gripper.set_pose(pos, rot_m)
            self._gui_step(6, sleep=0.003)
            frame_log.append({'phase': 'lift', 'step': i, **_snapshot(self.object_id, self.gripper)})

        # Judgment
        obj_pos_after = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
        success = obj_pos_after[2] > 0.10
        frame_log.append({'phase': 'done', 'step': 'final', 'success': success,
                          **_snapshot(self.object_id, self.gripper)})

        if self.gui:
            p.removeAllUserDebugItems()
            p.addUserDebugText('SUCCESS' if success else 'FAILED', final_pos,
                               [0, 1, 0] if success else [1, 0, 0], 2.5, lifeTime=0)
            time.sleep(1.0)

        return {'success': success, 'lift_z': obj_pos_after[2],
                'score': grasp.score, 'translation': center,
                'rotation': rot_m, 'width': width, 'depth': depth}

    def _reset_object(self):
        pos, orn = p.getBasePositionAndOrientation(self.object_id)
        p.resetBasePositionAndOrientation(self.object_id, [pos[0], pos[1], 0.05], orn)
        for _ in range(150):
            p.stepSimulation()

    def _get_gripper_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.gripper.base_id)[0])
