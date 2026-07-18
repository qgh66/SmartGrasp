"""Rigid two-box parallel gripper for stable grasp-pose simulation."""

from __future__ import annotations

import numpy as np
import pybullet as p

from .robot_gripper import JakaZu3Robotiq85Gripper, ROBOTIQ_TCP_OFFSET


BOX_FINGER_LENGTH = 0.060
BOX_FINGER_THICKNESS = 0.008
BOX_FINGER_HEIGHT = 0.020
BOX_MIN_OPENING = 0.002
BOX_FINGER_FRICTION = 5.0
BOX_PALM_X_OFFSET = -0.045
BOX_PALM_X_HALF_EXTENT = 0.012
BOX_WRIST_RADIUS = 0.022
BOX_WRIST_OVERLAP = 0.003
ROBOT_GRAY_RGBA = [0.84706, 0.84706, 0.84706, 1.0]


class JakaZu3BoxParallelGripper(JakaZu3Robotiq85Gripper):
    """JAKA arm with two rigid kinematic box fingers.

    The existing combined JAKA/Robotiq URDF remains the robot source, but its
    Robotiq visuals and collisions are disabled only for this class. Two
    joint-free box bodies become the actual grasping collision geometry.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.palm_id = None
        self.wrist_adapter_id = None
        self._box_bodies_ready = False
        self._last_bilateral_contact = False
        self._arm_collision_link_ids = []

    def load(self, position=(0.3, 0.0, 0.3), orientation=(0, 0, 0, 1)):
        self._box_bodies_ready = False
        super().load(position, orientation)
        self._disable_robotiq_geometry()
        self._create_box_gripper()
        self._box_bodies_ready = True
        self._sync_box_bodies()
        return self.base_id

    def remove(self):
        self.release_grasp()
        self._box_bodies_ready = False
        for body_id in (
            self.wrist_adapter_id,
            self.palm_id,
            self.left_id,
            self.right_id,
        ):
            if body_id is not None and p.isConnected():
                try:
                    p.removeBody(body_id)
                except p.error:
                    pass
        self.palm_id = None
        self.wrist_adapter_id = None
        self.left_id = None
        self.right_id = None
        self.base_id = self.robot_id
        super().remove()

    def set_opening(self, width: float):
        self._current_opening = float(np.clip(width, BOX_MIN_OPENING, self._max_opening))
        self._last_bilateral_contact = False
        if self._box_bodies_ready:
            self._sync_box_bodies()

    def close_fingers(self, target_width: float, steps: int = 100, object_id=None):
        del target_width
        start = float(self._current_opening)
        step_count = max(1, int(steps))
        self._last_bilateral_contact = False
        for step in range(step_count):
            fraction = (step + 1) / step_count
            self._current_opening = start + (BOX_MIN_OPENING - start) * fraction
            self._sync_box_bodies()
            p.performCollisionDetection()
            p.stepSimulation()
            if object_id is not None and self.has_bilateral_finger_contact(object_id):
                self._last_bilateral_contact = True
                break
        self._sync_box_bodies()

    def is_gripper_closed(self):
        return self._last_bilateral_contact

    def has_bilateral_finger_contact(self, object_id):
        return bool(
            p.getContactPoints(bodyA=self.left_id, bodyB=object_id)
            and p.getContactPoints(bodyA=self.right_id, bodyB=object_id)
        )

    def finger_contact_links(self, object_id):
        contacts = []
        if p.getContactPoints(bodyA=self.left_id, bodyB=object_id):
            contacts.append(int(self.left_id))
        if p.getContactPoints(bodyA=self.right_id, bodyB=object_id):
            contacts.append(int(self.right_id))
        return contacts

    def finger_link_positions(self):
        return {
            "left": [list(p.getBasePositionAndOrientation(self.left_id)[0])],
            "right": [list(p.getBasePositionAndOrientation(self.right_id)[0])],
        }

    def contact_body_ids(self):
        return [self.left_id, self.right_id]

    def collision_body_ids(self):
        return [self.left_id, self.right_id]

    def set_collision_with_object(self, object_id, enabled):
        enabled_flag = 1 if enabled else 0
        for link_index in self._arm_collision_link_ids:
            p.setCollisionFilterPair(
                self.robot_id,
                object_id,
                link_index,
                -1,
                enabled_flag,
            )
        for finger_id in (self.left_id, self.right_id):
            p.setCollisionFilterPair(
                finger_id,
                object_id,
                -1,
                -1,
                enabled_flag,
            )

    def max_penetration_depth(self, object_id):
        penetration_depth = 0.0
        for link_index in self._arm_collision_link_ids:
            for contact in p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=object_id,
                linkIndexA=link_index,
            ):
                penetration_depth = max(penetration_depth, max(0.0, -float(contact[8])))
        for finger_id in (self.left_id, self.right_id):
            for contact in p.getContactPoints(bodyA=finger_id, bodyB=object_id):
                penetration_depth = max(penetration_depth, max(0.0, -float(contact[8])))
        return penetration_depth

    def snapshot_extra(self):
        extra = super().snapshot_extra()
        extra.update({
            "robot_model": "jaka_zu3_box_parallel",
            "box_finger_size": [
                BOX_FINGER_LENGTH,
                BOX_FINGER_THICKNESS,
                BOX_FINGER_HEIGHT,
            ],
        })
        return extra

    def metadata(self):
        metadata = super().metadata()
        metadata.update({
            "model": "jaka_zu3_box_parallel",
            "gripper_model": "box_parallel",
            "box_finger_size": [
                BOX_FINGER_LENGTH,
                BOX_FINGER_THICKNESS,
                BOX_FINGER_HEIGHT,
            ],
            "max_opening": self._max_opening,
            "internal_gripper_joints": 0,
        })
        return metadata

    def _sync_attachment(self):
        super()._sync_attachment()
        if self._box_bodies_ready:
            self._sync_box_bodies()

    def _sync_transport_attachment(self):
        """Rigidly align all simplified gripper bodies without hidden steps."""
        if self.robot_id is None:
            return
        self._hold_current_joints()
        if self._box_bodies_ready:
            self._sync_box_bodies()

    def _disable_robotiq_geometry(self):
        self._arm_collision_link_ids = [-1]
        for joint_index in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, joint_index)[12].decode("utf-8")
            if "robotiq" not in link_name:
                self._arm_collision_link_ids.append(joint_index)
                continue
            p.changeVisualShape(
                self.robot_id,
                joint_index,
                rgbaColor=[0.0, 0.0, 0.0, 0.0],
            )
            p.setCollisionFilterGroupMask(self.robot_id, joint_index, 0, 0)

    def _create_box_gripper(self):
        finger_half_extents = [
            BOX_FINGER_LENGTH / 2.0,
            BOX_FINGER_THICKNESS / 2.0,
            BOX_FINGER_HEIGHT / 2.0,
        ]
        finger_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=finger_half_extents,
        )
        finger_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=finger_half_extents,
            rgbaColor=ROBOT_GRAY_RGBA,
        )
        self.left_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=finger_collision,
            baseVisualShapeIndex=finger_visual,
        )
        self.right_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=finger_collision,
            baseVisualShapeIndex=finger_visual,
        )

        palm_half_extents = [0.012, (self._max_opening + 0.025) / 2.0, 0.014]
        palm_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=palm_half_extents,
            rgbaColor=ROBOT_GRAY_RGBA,
        )
        self.palm_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=palm_visual,
        )

        # The hidden Robotiq body originally occupied the distance from the
        # robot flange to this palm. Fill it with a visual-only wrist adapter
        # so the simplified gripper remains attached without changing contact.
        flange_x = -float(ROBOTIQ_TCP_OFFSET[0])
        palm_rear_x = BOX_PALM_X_OFFSET - BOX_PALM_X_HALF_EXTENT
        wrist_length = palm_rear_x - flange_x + 2.0 * BOX_WRIST_OVERLAP
        wrist_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=BOX_WRIST_RADIUS,
            length=wrist_length,
            rgbaColor=ROBOT_GRAY_RGBA,
            visualFrameOrientation=p.getQuaternionFromEuler([0.0, np.pi / 2.0, 0.0]),
        )
        self.wrist_adapter_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=wrist_visual,
        )
        self.base_id = self.palm_id

        for finger_id in (self.left_id, self.right_id):
            p.changeDynamics(
                finger_id,
                -1,
                lateralFriction=BOX_FINGER_FRICTION,
                spinningFriction=0.5,
                rollingFriction=0.001,
                restitution=0.0,
                contactStiffness=100000.0,
                contactDamping=1000.0,
                frictionAnchor=1,
            )
            for robot_link in range(-1, p.getNumJoints(self.robot_id)):
                p.setCollisionFilterPair(
                    self.robot_id,
                    finger_id,
                    robot_link,
                    -1,
                    0,
                )

    def _sync_box_bodies(self):
        if not self._box_bodies_ready:
            return
        tcp_position, tcp_orientation = self.get_tcp_pose()
        rotation = np.asarray(p.getMatrixFromQuaternion(tcp_orientation)).reshape(3, 3)
        tcp_position = np.asarray(tcp_position, dtype=float)
        separation = self._current_opening / 2.0 + BOX_FINGER_THICKNESS / 2.0
        x_offset = 0.0
        left_position = tcp_position + rotation @ np.array([x_offset, separation, 0.0])
        right_position = tcp_position + rotation @ np.array([x_offset, -separation, 0.0])
        palm_position = tcp_position + rotation @ np.array([BOX_PALM_X_OFFSET, 0.0, 0.0])
        flange_x = -float(ROBOTIQ_TCP_OFFSET[0])
        palm_rear_x = BOX_PALM_X_OFFSET - BOX_PALM_X_HALF_EXTENT
        wrist_center_x = 0.5 * (flange_x + palm_rear_x)
        wrist_position = tcp_position + rotation @ np.array([wrist_center_x, 0.0, 0.0])

        for body_id, body_position in (
            (self.left_id, left_position),
            (self.right_id, right_position),
            (self.palm_id, palm_position),
            (self.wrist_adapter_id, wrist_position),
        ):
            if body_id is None:
                continue
            p.resetBasePositionAndOrientation(body_id, body_position, tcp_orientation)
            p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])
