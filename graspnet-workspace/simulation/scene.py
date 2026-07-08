"""
PyBullet 场景管理模块。

职责：
- 初始化 PyBullet 物理引擎（GUI / DIRECT 模式）
- 加载桌面（平面）
- 加载物体 mesh（.obj 格式）
- 设置重力、光照等全局参数
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pybullet as p
import pybullet_data


@dataclass
class SceneObject:
    """Metadata for one object loaded into the PyBullet scene."""

    body_id: int
    name: str
    path: str
    scale: float
    mass: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulationScene:
    """管理 PyBullet 仿真场景的创建与销毁。"""

    def __init__(self, gui: bool = True, gravity: float = -9.8):
        """
        Args:
            gui: True 打开图形窗口，False 使用无头模式（DIRECT）。
            gravity: 重力加速度，默认 -9.8 m/s²（Z 轴向下）。
        """
        self.gui = gui
        self.gravity = gravity
        self.client_id = None
        self.plane_id = None
        self.object_ids = []
        self.objects_by_id: dict[int, SceneObject] = {}
        self.objects_by_name: dict[str, int] = {}
        self._temp_urdfs: list[str] = []

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def connect(self):
        """连接到 PyBullet 引擎。"""
        mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity)
        # 实时仿真模式，stepSimulation 立即刷新画面
        p.setRealTimeSimulation(0)

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=30,
                cameraPitch=-25,
                cameraTargetPosition=[0.3, 0.0, 0.15],
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    def disconnect(self):
        """断开 PyBullet 连接，释放资源。"""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None
        for urdf_path in self._temp_urdfs:
            try:
                Path(urdf_path).unlink(missing_ok=True)
            except OSError:
                pass
        self._temp_urdfs = []
        self.object_ids = []
        self.objects_by_id = {}
        self.objects_by_name = {}

    # ------------------------------------------------------------------
    # 场景搭建
    # ------------------------------------------------------------------

    def load_plane(self):
        """加载地面平面。"""
        self.plane_id = p.loadURDF("plane.urdf")

    def load_object(self, obj_path: str, position=(0.3, 0.0, 0.1),
                    orientation=(0, 0, 0, 1), scale: float = 1.0,
                    mass: float = 0.1, lateral_friction: float = 1.0,
                    spinning_friction: float = 0.1, name: str | None = None,
                    metadata: dict[str, Any] | None = None):
        """
        加载物体到场景中。支持 .obj（自动包装为 URDF）和 .urdf 两种格式。

        Args:
            obj_path: .obj 或 .urdf 文件路径。
            position: 初始位置 (x, y, z)。
            orientation: 初始朝向（四元数）。
            scale: 缩放因子。
            mass: 质量（kg）。
            lateral_friction: 横向摩擦系数。
            spinning_friction: 旋转摩擦系数。
            name: 场景内物体名称。不传时用文件名自动生成。
            metadata: 任务层附加信息，例如类别、目标标记、语言描述等。

        Returns:
            object_id: PyBullet 物体 ID。
        """
        obj_path = str(obj_path)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Object model not found: {obj_path}")

        suffix = Path(obj_path).suffix.lower()
        if suffix == ".obj":
            # .obj 的缩放已写进生成的 URDF 内 <mesh scale=...>，这里不能再用
            # globalScaling，否则会二次缩放（scale²）导致物体尺寸严重偏小。
            urdf_path = self._obj_to_urdf(obj_path, scale)
            self._temp_urdfs.append(urdf_path)
            global_scaling = 1.0
        elif suffix == ".urdf":
            urdf_path = obj_path  # 直接加载 URDF
            global_scaling = scale
        else:
            raise ValueError(f"Unsupported object model type: {obj_path}")

        obj_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=global_scaling,
        )
        p.changeDynamics(obj_id, -1, mass=mass,
                         lateralFriction=lateral_friction,
                         spinningFriction=spinning_friction)
        self.object_ids.append(obj_id)
        object_name = self._unique_object_name(name or Path(obj_path).stem)
        scene_object = SceneObject(
            body_id=obj_id,
            name=object_name,
            path=obj_path,
            scale=float(scale),
            mass=float(mass),
            metadata=dict(metadata or {}),
        )
        self.objects_by_id[obj_id] = scene_object
        self.objects_by_name[object_name] = obj_id
        return obj_id

    def load_objects(self, object_specs: list[dict[str, Any]]) -> list[int]:
        """批量加载对象配置。

        每个 spec 至少需要 `path`，可选字段包括 `name`、`position`、
        `orientation`（四元数）、`euler`（RPY）、`scale`、`mass`、
        `lateral_friction`、`spinning_friction` 和 `metadata`。
        """
        body_ids = []
        for spec in object_specs:
            if "path" not in spec:
                raise KeyError(f"Object spec missing required key 'path': {spec}")
            orientation = spec.get("orientation")
            if orientation is None and "euler" in spec:
                orientation = p.getQuaternionFromEuler(spec["euler"])
            body_ids.append(
                self.load_object(
                    spec["path"],
                    position=spec.get("position", (0.3, 0.0, 0.1)),
                    orientation=orientation or (0, 0, 0, 1),
                    scale=spec.get("scale", 1.0),
                    mass=spec.get("mass", 0.1),
                    lateral_friction=spec.get("lateral_friction", 1.0),
                    spinning_friction=spec.get("spinning_friction", 0.1),
                    name=spec.get("name"),
                    metadata=spec.get("metadata"),
                )
            )
        return body_ids

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _obj_to_urdf(self, obj_path: str, scale: float = 1.0) -> str:
        """将 .obj 文件包装为 URDF 格式（临时文件）。"""
        import tempfile

        abs_path = os.path.abspath(obj_path)

        urdf_content = f"""<?xml version="1.0"?>
<robot name="object">
  <link name="base_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{abs_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{abs_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
  </link>
</robot>"""

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False
        )
        tmp.write(urdf_content)
        tmp.close()
        return tmp.name

    def _unique_object_name(self, base_name: str) -> str:
        name = base_name
        idx = 1
        while name in self.objects_by_name:
            idx += 1
            name = f"{base_name}_{idx}"
        return name

    def step(self, steps: int = 1):
        """推进仿真步数。"""
        for _ in range(steps):
            p.stepSimulation()

    def get_object_pose(self, obj_id: int):
        """获取物体当前位姿。"""
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        return pos, orn

    def get_object_info(self, obj_id: int) -> SceneObject:
        """获取已加载物体的登记信息。"""
        return self.objects_by_id[obj_id]

    def get_object_registry(self) -> dict[int, SceneObject]:
        """返回 body_id 到物体信息的映射副本。"""
        return dict(self.objects_by_id)

    def get_body_id_by_name(self, name: str) -> int:
        """按物体名称查 PyBullet body id。"""
        return self.objects_by_name[name]

    def get_object_poses(self) -> dict[int, dict[str, Any]]:
        """返回所有已登记物体的当前位姿和元信息。"""
        poses = {}
        for body_id, obj in self.objects_by_id.items():
            pos, orn = self.get_object_pose(body_id)
            poses[body_id] = {
                "name": obj.name,
                "path": obj.path,
                "position": list(pos),
                "orientation": list(orn),
                "scale": obj.scale,
                "mass": obj.mass,
                "metadata": dict(obj.metadata),
            }
        return poses
