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
    left_pos, left_orn = p.getBasePositionAndOrientation(gripper.left_id)
    right_pos, right_orn = p.getBasePositionAndOrientation(gripper.right_id)
    return {
        'obj_pos': list(obj_pos), 'obj_orn': list(obj_orn),
        'gripper_pos': list(base_pos), 'gripper_orn': list(base_orn),
        'left_pos': list(left_pos), 'left_orn': list(left_orn),
        'right_pos': list(right_pos), 'right_orn': list(right_orn),
        'opening': gripper._current_opening,
        'gripper_geometry': {
            'base_size': gripper.BASE_SIZE,
            'base_width': gripper.BASE_WIDTH,
            'finger_length': gripper.FINGER_LENGTH,
            'finger_width': gripper.FINGER_WIDTH,
            'finger_height': gripper.FINGER_HEIGHT,
        },
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

    def evaluate_push(self, center_point, push_distance_x=0.05,
                      rotation_matrix=None,
                      approach_distance=0.04, approach_steps=20,
                      push_steps=50, closed_width=0.01):
        """Execute a Reveal +X/-X push and record frames for Dash replay."""
        center = np.asarray(center_point, dtype=float)
        if center.shape != (3,):
            raise ValueError("center_point must contain exactly three XYZ values")
        if abs(push_distance_x) < 1e-8:
            raise ValueError("push_distance_x must be non-zero")

        direction = np.array([np.sign(push_distance_x), 0.0, 0.0])
        move_distance = abs(float(push_distance_x))
        rotation = np.asarray(
            rotation_matrix if rotation_matrix is not None else np.eye(3),
            dtype=float,
        ).reshape(3, 3)
        frame_log = []

        aabb_min, aabb_max = p.getAABB(self.object_id)
        aabb_min = np.asarray(aabb_min, dtype=float)
        aabb_max = np.asarray(aabb_max, dtype=float)
        contact_face_x = aabb_min[0] if direction[0] > 0 else aabb_max[0]
        finger_tip_offset = (
            self.gripper.BASE_SIZE / 2.0 + self.gripper.FINGER_LENGTH)
        contact_pos = np.array([
            contact_face_x - direction[0] * (finger_tip_offset + 0.002),
            center[1],
            max(center[2], TABLE_Z + self.gripper.FINGER_HEIGHT / 2.0),
        ])
        pre_push_pos = contact_pos - direction * approach_distance
        push_end_pos = contact_pos + direction * move_distance

        self.gripper.release_grasp()
        self.gripper.set_opening(closed_width)
        self.gripper.set_pose(pre_push_pos, rotation)
        self._gui_step(30, sleep=0.01)
        frame_log.append({
            'phase': 'push_ready', 'step': 'ready',
            **_snapshot(self.object_id, self.gripper),
        })

        for i in range(max(1, int(approach_steps))):
            frac = (i + 1) / max(1, int(approach_steps))
            pos = pre_push_pos + (contact_pos - pre_push_pos) * frac
            self.gripper.set_pose(pos, rotation)
            self._gui_step(3, sleep=0.004)
            frame_log.append({
                'phase': 'push_approach', 'step': i,
                **_snapshot(self.object_id, self.gripper),
            })

        obj_pos_before = np.asarray(
            p.getBasePositionAndOrientation(self.object_id)[0], dtype=float)
        for i in range(max(1, int(push_steps))):
            frac = (i + 1) / max(1, int(push_steps))
            pos = contact_pos + (push_end_pos - contact_pos) * frac
            self.gripper.set_pose(pos, rotation)
            self._gui_step(4, sleep=0.004)
            frame_log.append({
                'phase': 'push', 'step': i,
                **_snapshot(self.object_id, self.gripper),
            })

        self._gui_step(60, sleep=0.01)
        obj_pos_after = np.asarray(
            p.getBasePositionAndOrientation(self.object_id)[0], dtype=float)
        displacement = obj_pos_after - obj_pos_before
        signed_displacement = float(np.dot(displacement, direction))
        success_threshold = min(0.01, move_distance * 0.2)
        success = signed_displacement >= success_threshold
        frame_log.append({
            'phase': 'done', 'step': 'final', 'success': success,
            **_snapshot(self.object_id, self.gripper),
        })

        if self.gui:
            p.removeAllUserDebugItems()
            p.addUserDebugText(
                'PUSH SUCCESS' if success else 'PUSH FAILED',
                obj_pos_after.tolist(),
                [0, 1, 0] if success else [1, 0, 0],
                2.0,
                lifeTime=0,
            )
            time.sleep(1.0)

        return {
            'action_type': 'push',
            'success': success,
            'score': 1.0,
            'lift_z': float(obj_pos_after[2]),
            'translation': center,
            'rotation': rotation,
            'width': float(closed_width),
            'depth': 0.0,
            'push_direction': direction,
            'requested_distance': move_distance,
            'actual_displacement': displacement,
            'signed_displacement': signed_displacement,
            'start_position': obj_pos_before,
            'target_position': obj_pos_before + direction * move_distance,
            'final_position': obj_pos_after,
            'frame_log': frame_log,
        }
