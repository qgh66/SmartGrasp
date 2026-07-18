"""PyBullet GUI video recorder for grasp simulation."""

from __future__ import annotations

from pathlib import Path

import pybullet as p


class PyBulletVideoRecorder:
    """Record the native PyBullet GUI framebuffer to an MP4 file."""

    def __init__(
        self,
        output_path: str | Path,
    ):
        self.output_path = Path(output_path)
        self._logging_id = None

    @property
    def enabled(self) -> bool:
        return self._logging_id is not None

    def start(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        connection = p.getConnectionInfo()
        if connection.get("connectionMethod") != p.GUI:
            raise RuntimeError(
                "Native PyBullet MP4 recording requires a GUI connection; "
                "run with PYBULLET_GUI=1"
            )
        self._logging_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4,
            str(self.output_path),
        )
        if self._logging_id < 0:
            self._logging_id = None
            raise RuntimeError(f"Could not start PyBullet GUI recording: {self.output_path}")

    def close(self):
        if self._logging_id is not None:
            p.stopStateLogging(self._logging_id)
            self._logging_id = None
