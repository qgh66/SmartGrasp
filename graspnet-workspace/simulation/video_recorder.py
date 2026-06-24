"""Offscreen PyBullet video recorder for grasp simulation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pybullet as p


class PyBulletVideoRecorder:
    """Capture frames from the active PyBullet scene and write an mp4 file."""

    def __init__(
        self,
        output_path: str | Path,
        width: int = 1280,
        height: int = 720,
        fps: int = 20,
        camera_position=(0.55, -0.55, 0.42),
        camera_target=(0.25, 0.0, 0.08),
        camera_up=(0.0, 0.0, 1.0),
    ):
        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.camera_position = camera_position
        self.camera_target = camera_target
        self.camera_up = camera_up
        self._writer = None
        self._view_matrix = p.computeViewMatrix(camera_position, camera_target, camera_up)
        self._projection_matrix = p.computeProjectionMatrixFOV(
            fov=55.0,
            aspect=self.width / self.height,
            nearVal=0.01,
            farVal=5.0,
        )

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def start(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"Could not open video writer: {self.output_path}")

    def capture(self):
        if self._writer is None:
            return
        _, _, rgba, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        frame = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
        bgr = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
