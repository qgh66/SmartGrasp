"""
PyBullet 场景管理模块。

职责：
- 初始化 PyBullet 物理引擎（GUI / DIRECT 模式）
- 加载桌面（平面）
- 加载物体 mesh（.obj 格式）
- 设置重力、光照等全局参数
"""

import os
import pybullet as p
import pybullet_data


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

    # ------------------------------------------------------------------
    # 场景搭建
    # ------------------------------------------------------------------

    def load_plane(self):
        """加载地面平面。"""
        self.plane_id = p.loadURDF("plane.urdf")

    def load_object(self, obj_path: str, position=(0.3, 0.0, 0.1),
                    orientation=(0, 0, 0, 1), scale: float = 1.0,
                    mass: float = 0.1, lateral_friction: float = 1.0,
                    spinning_friction: float = 0.1):
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

        Returns:
            object_id: PyBullet 物体 ID。
        """
        if obj_path.endswith(".obj"):
            urdf_path = self._obj_to_urdf(obj_path, scale)
        else:
            urdf_path = obj_path  # 直接加载 URDF

        obj_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=scale,
        )
        p.changeDynamics(obj_id, -1, mass=mass,
                         lateralFriction=lateral_friction,
                         spinningFriction=spinning_friction)
        self.object_ids.append(obj_id)
        return obj_id

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _obj_to_urdf(self, obj_path: str, scale: float = 1.0) -> str:
        """将 .obj 文件包装为 URDF 格式（临时文件）。"""
        import tempfile

        abs_path = os.path.abspath(obj_path)
        mesh_dir = os.path.dirname(abs_path)
        mesh_file = os.path.basename(abs_path)

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
        <mesh filename="{mesh_file}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_file}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
  </link>
</robot>"""

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", dir=mesh_dir, delete=False
        )
        tmp.write(urdf_content)
        tmp.close()
        return tmp.name

    def step(self, steps: int = 1):
        """推进仿真步数。"""
        for _ in range(steps):
            p.stepSimulation()

    def get_object_pose(self, obj_id: int):
        """获取物体当前位姿。"""
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        return pos, orn
