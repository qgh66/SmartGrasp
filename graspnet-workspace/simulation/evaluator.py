"""
抓取执行与物理评估模块（逐帧轨迹版，JAKA Zu3 + Robotiq-85）。

职责：
- 逐一对 GraspGroup 中的候选抓取执行物理仿真
- **每个仿真步记录物体+夹爪完整位姿**（用于 Dash GUI 真实场景回放）
- 判定抓取是否成功

抓取流程参照 ThinkGrasp/UR5e 风格的 grasp() 原语，但执行体换成 JAKA Zu3
机械臂 + Robotiq-85 夹爪（见 robot_gripper.py）：
  张开 → 移到目标上方 over 点 → 沿 approach 方向直线下插 → 欠驱动闭合（真实摩擦
  夹持，可选约束仅用于抑制数值抖动）→ 直线抬回 → 搬运到配置的放置关节位姿
  → 可选松爪投放。配置目标关节位姿时，以物体是否随夹爪到达该位置判定成功；
  松爪后的下落只记录为诊断信息，不改变已经完成的搬运判定。
"""

import sys
import os
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "graspnetAPI"))
from graspnetAPI.grasp import GraspGroup
from .robot_gripper import JakaZu3Robotiq85Gripper, MAX_EXECUTION_TCP_ERROR

TABLE_Z = 0.0
TABLE_CLEARANCE = 0.005
MAX_GRASP_CENTER_DIST = 0.04
OBJECT_POINT_Z_THRESHOLD = 0.005
MIN_EXECUTION_CENTER_Z = TABLE_Z + TABLE_CLEARANCE
MIN_SUCCESSFUL_LIFT_DELTA = 0.03
TRANSPORT_MIN_FOLLOW_RATIO = 0.5
TRANSPORT_RELATIVE_DISTANCE_TOLERANCE = 0.05
PLACE_JOINT_TOLERANCE_DEG = 1.0
ASSISTED_GRASP_MAX_TCP_DISTANCE = 0.12


def _snapshot(obj_id, gripper):
    """拍摄一帧场景状态。"""
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
    base_pos, base_orn = p.getBasePositionAndOrientation(gripper.base_id)
    left_pos, _ = p.getBasePositionAndOrientation(gripper.left_id)
    right_pos, _ = p.getBasePositionAndOrientation(gripper.right_id)
    tcp_pos, tcp_orn = gripper.get_tcp_pose()
    return {
        'obj_pos': list(obj_pos), 'obj_orn': list(obj_orn),
        'gripper_pos': list(base_pos), 'gripper_orn': list(base_orn),
        'tcp_pos': list(tcp_pos), 'tcp_orn': list(tcp_orn),
        'left_pos': list(left_pos), 'right_pos': list(right_pos),
        'opening': gripper._current_opening,
        }


def _unit(vec, fallback):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=float)
    return np.asarray(vec, dtype=float) / norm


def _object_z_bounds(object_points: np.ndarray | None) -> tuple[float, float, float]:
    """Return robust target-object z bounds and a fallback waist height."""
    if object_points is None or len(object_points) == 0:
        return MIN_EXECUTION_CENTER_Z, MIN_EXECUTION_CENTER_Z, MIN_EXECUTION_CENTER_Z
    z_values = np.asarray(object_points[:, 2], dtype=float)
    z_low, z_high = np.percentile(z_values, [10, 90])
    waist_z = 0.5 * (float(z_low) + float(z_high))
    return (
        max(float(z_low), MIN_EXECUTION_CENTER_Z),
        max(float(z_high), MIN_EXECUTION_CENTER_Z),
        max(waist_z, MIN_EXECUTION_CENTER_Z),
    )


class GraspEvaluator:
    """基于 PyBullet 的抓取物理评估器（逐帧轨迹版）。"""

    def __init__(self, object_id: int, gripper: JakaZu3Robotiq85Gripper,
                 point_cloud: np.ndarray = None, gui: bool = False,
                 assisted_grasp: bool = False,
                 validate_target_center: bool = True,
                 place_target_joint_pose_deg=None,
                 release_after_place: bool = False,
                 release_settle_steps: int = 120,
                 gui_speed: float = 1.0):
        self.object_id = object_id
        self.gripper = gripper
        self.gui = gui
        self.assisted_grasp = assisted_grasp
        self.validate_target_center = validate_target_center
        self.place_target_joint_pose_deg = (
            [float(value) for value in place_target_joint_pose_deg]
            if place_target_joint_pose_deg is not None
            else None
        )
        if self.place_target_joint_pose_deg is not None and len(self.place_target_joint_pose_deg) != 6:
            raise ValueError("place_target_joint_pose_deg must contain exactly 6 values")
        self.release_after_place = bool(release_after_place)
        if self.release_after_place and self.place_target_joint_pose_deg is None:
            raise ValueError(
                "release_after_place requires place_target_joint_pose_deg"
            )
        if int(release_settle_steps) < 0:
            raise ValueError("release_settle_steps must be non-negative")
        self.release_settle_steps = int(release_settle_steps)
        if gui_speed <= 0:
            raise ValueError("gui_speed must be greater than zero")
        self.gui_speed = float(gui_speed)
        if point_cloud is not None:
            self.obj_pts = point_cloud[point_cloud[:, 2] > 0.005]
        else:
            self.obj_pts = None
        self.object_z_low, self.object_z_high, self.object_waist_z = _object_z_bounds(self.obj_pts)
        self._initial_state_id = p.saveState()

    def _gui_step(self, n: int = 1, sleep: float = 0.0):
        for _ in range(n):
            p.stepSimulation()
        if self.gui and sleep > 0:
            time.sleep(sleep / self.gui_speed)

    def evaluate(self, grasp_group: GraspGroup, top_k: int = 10,
                 approach_depth_offset: float = 0.05,
                 lift_height: float = 0.20, lift_steps: int = 200,
                 stop_on_success: bool = False,
                 preserve_success_state: bool = False):
        gg_top = grasp_group[:min(top_k, len(grasp_group))]
        results = []
        successful_state_id = None
        successful_grasp_constraint = None
        try:
            for idx, grasp in enumerate(gg_top):
                p.restoreState(self._initial_state_id)
                self.gripper.grasp_constraint = None
                frame_log = []
                res = self._execute(grasp, approach_depth_offset, lift_height, lift_steps, frame_log)
                res['grasp_index'] = idx
                res['frame_log'] = frame_log
                results.append(res)
                if preserve_success_state and res['success']:
                    successful_state_id = p.saveState()
                    successful_grasp_constraint = self.gripper.grasp_constraint
                if stop_on_success and res['success']:
                    break
        finally:
            p.restoreState(successful_state_id or self._initial_state_id)
            self.gripper.grasp_constraint = successful_grasp_constraint
            if successful_state_id is not None:
                p.removeState(successful_state_id)
            p.removeState(self._initial_state_id)
            self._initial_state_id = None
        return results

    def _execute(self, grasp, approach_depth_offset, lift_height, lift_steps, frame_log):
        center = grasp.translation.copy()
        raw_center = center.copy()
        rot_m = grasp.rotation_matrix.copy()
        raw_width = float(grasp.width)
        width = min(raw_width, self.gripper._max_opening)
        depth = grasp.depth

        # Preserve GraspNet's predicted height. Only clamp clear support-plane or
        # above-object outliers to robust bounds derived from the target cloud.
        center[2] = float(np.clip(center[2], self.object_z_low, self.object_z_high))

        # GraspNet local X 轴作为 approach/depth 方向
        approach_dir = _unit(rot_m[:, 0], [0, 0, -1])
        # 预抓取点：沿 approach 反方向退出 depth + offset
        pre_grasp_pos = center - approach_dir * (depth + approach_depth_offset)
        # over 点：在预抓取点正上方 0.2m，对齐参考 grasp() 的“先到目标上方”
        over_pos = pre_grasp_pos + np.array([0.0, 0.0, 0.20])

        if self.validate_target_center and self.obj_pts is not None and len(self.obj_pts) > 0:
            center_dist = float(np.linalg.norm(self.obj_pts - center, axis=1).min())
            if center_dist > MAX_GRASP_CENTER_DIST:
                obj_pos_before, _ = p.getBasePositionAndOrientation(self.object_id)
                return {'success': False, 'lift_z': obj_pos_before[2],
                        'score': grasp.score, 'translation': center,
                        'rotation': rot_m, 'width': width, 'depth': depth,
                        'raw_width': raw_width, 'width_clipped': raw_width > width,
                        'failure_reason': 'grasp_center_not_on_object',
                        'center_object_dist': center_dist,
                        'raw_translation': raw_center,
                        'execution_center_z': float(center[2]),
                        'object_waist_z': float(self.object_waist_z)}

        # 注：已按需求取消抓取中心/预抓取点的桌面高度限制（原 approach_below_table 护栏），
        # 允许 GraspNet 给出贴近桌面（z≈0）的抓取直接进入执行阶段。

        initial_obj_pos, initial_obj_orn = p.getBasePositionAndOrientation(self.object_id)
        initial_linear_velocity, initial_angular_velocity = p.getBaseVelocity(self.object_id)

        # Step 1: 张开夹爪并移动到目标上方 over 点（参考 grasp(): open + move over）
        # Joint-space interpolation has no collision-aware planner and may sweep
        # through the target. Ignore target contact only during free-space transit,
        # then restore the exact initial object state before physical approach.
        self.gripper.set_collision_with_object(self.object_id, enabled=False)
        self.gripper.set_opening(0.085)
        self.gripper.set_pose(over_pos, rot_m)
        self._gui_step(10, sleep=0.02)
        p.resetBasePositionAndOrientation(self.object_id, initial_obj_pos, initial_obj_orn)
        p.resetBaseVelocity(
            self.object_id,
            linearVelocity=initial_linear_velocity,
            angularVelocity=initial_angular_velocity,
        )
        # Keep target collision disabled during the unplanned IK descent. The
        # model-free filter has already checked the gripper approach corridor;
        # this phase only positions the open gripper for physical closure.
        obj_z_before = float(initial_obj_pos[2])
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
            tcp_error = self.gripper.set_pose(pos, rot_m)
            self._gui_step(2, sleep=0.005)
            frame_log.append({'phase': 'approach', 'step': i, **_snapshot(self.object_id, self.gripper)})

        if tcp_error > MAX_EXECUTION_TCP_ERROR:
            return {
                'success': False,
                'lift_z': obj_z_before,
                'score': grasp.score,
                'translation': center,
                'rotation': rot_m,
                'width': width,
                'depth': depth,
                'raw_width': raw_width,
                'width_clipped': raw_width > width,
                'failure_reason': 'tcp_target_unreachable',
                'tcp_target_error': tcp_error,
                'raw_translation': raw_center,
                'execution_center_z': float(center[2]),
                'object_waist_z': float(self.object_waist_z),
            }

        # Restore the untouched target at the final open-gripper pose, then
        # reject only severe final-pose interpenetration before enabling contact.
        p.resetBasePositionAndOrientation(self.object_id, initial_obj_pos, initial_obj_orn)
        p.resetBaseVelocity(
            self.object_id,
            linearVelocity=initial_linear_velocity,
            angularVelocity=initial_angular_velocity,
        )
        self.gripper.set_collision_with_object(self.object_id, enabled=True)
        p.performCollisionDetection()
        penetration_depth = self.gripper.max_penetration_depth(self.object_id)
        if penetration_depth > 0.005:
            return {
                'success': False,
                'lift_z': obj_z_before,
                'score': grasp.score,
                'translation': center,
                'rotation': rot_m,
                'width': width,
                'depth': depth,
                'raw_width': raw_width,
                'width_clipped': raw_width > width,
                'failure_reason': 'final_gripper_pose_penetrates_target',
                'penetration_depth': penetration_depth,
                'raw_translation': raw_center,
                'execution_center_z': float(center[2]),
                'object_waist_z': float(self.object_waist_z),
            }

        # Step 3: 欠驱动闭合，靠 Robotiq-85 真实摩擦夹持（不再创建固定约束吸附）
        self.gripper.close_fingers(
            target_width=width,
            steps=30,
            object_id=self.object_id,
        )
        self._gui_step(20, sleep=0.01)
        finger_link_positions = self.gripper.finger_link_positions()
        finger_contact_links = self.gripper.finger_contact_links(self.object_id)
        bilateral_contact = self.gripper.has_bilateral_finger_contact(self.object_id)
        close_obj_pos = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
        close_tcp_pos = self._get_tcp_pos()
        close_obj_tcp_distance = float(np.linalg.norm(close_obj_pos - close_tcp_pos))
        gripper_blocked_by_object = bool(self.gripper.is_gripper_closed())
        attachment_eligible = bool(
            bilateral_contact
            or (
                gripper_blocked_by_object
                and close_obj_tcp_distance <= ASSISTED_GRASP_MAX_TCP_DISTANCE
            )
        )
        assisted_constraint = False
        if self.assisted_grasp and attachment_eligible:
            self.gripper.create_grasp_constraint(self.object_id)
            assisted_constraint = True
        frame_log.append({'phase': 'close', 'step': 'done', **_snapshot(self.object_id, self.gripper)})

        # Step 4: 直线抬升
        final_pos = center + np.array([0, 0, lift_height])
        start_pos = self._get_tcp_pos()
        max_lift_tcp_error = 0.0
        for i in range(30):
            frac = (i + 1) / 30
            pos = start_pos + (final_pos - start_pos) * frac
            lift_tcp_error = self.gripper.set_pose(pos, rot_m)
            max_lift_tcp_error = max(max_lift_tcp_error, lift_tcp_error)
            self._gui_step(6, sleep=0.003)
            frame_log.append({'phase': 'lift', 'step': i, **_snapshot(self.object_id, self.gripper)})

        # Keep lift measurements for diagnostics and for legacy scenes that do
        # not define a placement pose. Placement-enabled tasks are judged only
        # after the object has been transported to the configured joint pose.
        grasped = bool(self.gripper.is_gripper_closed())
        obj_pos_after = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
        obj_z_after = float(obj_pos_after[2])
        obj_lift_delta = obj_z_after - float(obj_z_before)
        lifted = obj_lift_delta >= MIN_SUCCESSFUL_LIFT_DELTA
        placement = None
        failure_reason = None
        if self.place_target_joint_pose_deg is not None:
            actual_place_joints = self.gripper.move_to_place_joint_pose_deg(
                self.place_target_joint_pose_deg
            )
            self._gui_step(20, sleep=0.4)

            target_place_joints = np.deg2rad(self.place_target_joint_pose_deg)
            actual_place_joints = np.asarray(actual_place_joints, dtype=float)
            joint_error = (
                actual_place_joints - target_place_joints + np.pi
            ) % (2.0 * np.pi) - np.pi
            place_joint_max_error_deg = float(
                np.max(np.abs(np.rad2deg(joint_error)))
            )
            place_joint_reached = place_joint_max_error_deg <= PLACE_JOINT_TOLERANCE_DEG

            transport_target_obj_pos = np.array(
                p.getBasePositionAndOrientation(self.object_id)[0]
            )
            place_tcp_pos = self._get_tcp_pos()
            tcp_transport_distance = float(
                np.linalg.norm(place_tcp_pos - close_tcp_pos)
            )
            object_transport_distance = float(
                np.linalg.norm(transport_target_obj_pos - close_obj_pos)
            )
            place_obj_tcp_distance = float(
                np.linalg.norm(transport_target_obj_pos - place_tcp_pos)
            )
            transport_follow_ratio = object_transport_distance / max(
                tcp_transport_distance, 1e-6
            )
            relative_distance_limit = (
                close_obj_tcp_distance + TRANSPORT_RELATIVE_DISTANCE_TOLERANCE
            )
            object_followed_to_place = bool(
                place_obj_tcp_distance <= relative_distance_limit
                and (
                    tcp_transport_distance < 0.01
                    or transport_follow_ratio >= TRANSPORT_MIN_FOLLOW_RATIO
                )
            )
            success = bool(place_joint_reached and object_followed_to_place)
            if not place_joint_reached:
                failure_reason = 'place_joint_pose_not_reached'
            elif not object_followed_to_place:
                failure_reason = 'object_did_not_follow_to_place'
            frame_log.append({'phase': 'place', 'step': 'target_pose',
                              **_snapshot(self.object_id, self.gripper)})
            placement = {
                'target_joint_pose_deg': list(self.place_target_joint_pose_deg),
                'joint_max_error_deg': place_joint_max_error_deg,
                'joint_pose_reached': place_joint_reached,
                'object_position': transport_target_obj_pos.tolist(),
                'close_object_tcp_distance': close_obj_tcp_distance,
                'place_object_tcp_distance': place_obj_tcp_distance,
                'relative_distance_limit': relative_distance_limit,
                'tcp_transport_distance': tcp_transport_distance,
                'object_transport_distance': object_transport_distance,
                'transport_follow_ratio': transport_follow_ratio,
                'object_followed_to_place': object_followed_to_place,
                'grasp_held_at_target': not self.release_after_place,
                'released_after_place': self.release_after_place,
            }
            if self.release_after_place:
                release_start_position = transport_target_obj_pos.copy()
                self.gripper.release_grasp()
                self.gripper.set_opening(self.gripper._max_opening)
                self._gui_step(self.release_settle_steps, sleep=0.25)
                release_end_position = np.array(
                    p.getBasePositionAndOrientation(self.object_id)[0]
                )
                placement.update({
                    'release_settle_steps': self.release_settle_steps,
                    'post_release_object_position': (
                        release_end_position.tolist()
                    ),
                    'release_fall_distance': max(
                        0.0,
                        float(
                            release_start_position[2]
                            - release_end_position[2]
                        ),
                    ),
                })
                frame_log.append({
                    'phase': 'release',
                    'step': 'settled',
                    **_snapshot(self.object_id, self.gripper),
                })
        else:
            success = bool(bilateral_contact and lifted)
            if not bilateral_contact:
                failure_reason = 'no_bilateral_finger_contact'
            elif not lifted:
                failure_reason = 'object_not_lifted_3cm'

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
                'raw_width': raw_width, 'width_clipped': raw_width > width,
                # Joint/contact/lift fields remain diagnostic; placement-enabled
                # tasks use the transport and post-release checks above.
                'grasped_by_gripper': grasped,
                'bilateral_finger_contact': bilateral_contact,
                'attachment_eligible': attachment_eligible,
                'attachment_max_tcp_distance': ASSISTED_GRASP_MAX_TCP_DISTANCE,
                'finger_contact_links': finger_contact_links,
                'finger_link_positions': finger_link_positions,
                'assisted_constraint': assisted_constraint,
                'placement': placement,
                'failure_reason': failure_reason,
                'tcp_target_error': tcp_error,
                'max_lift_tcp_error': max_lift_tcp_error,
                'final_pose_penetration_depth': penetration_depth,
                'lifted': lifted,
                'min_successful_lift_delta': MIN_SUCCESSFUL_LIFT_DELTA,
                'obj_z_before': float(obj_z_before),
                'obj_z_after': obj_z_after,
                'obj_lift_delta': float(obj_lift_delta),
                'raw_translation': raw_center,
                'execution_center_z': float(center[2]),
                'object_waist_z': float(self.object_waist_z)}

    def _get_tcp_pos(self):
        tcp_pos, _ = self.gripper.get_tcp_pose()
        return np.asarray(tcp_pos, dtype=float)
