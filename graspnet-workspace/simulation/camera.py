"""
虚拟 RGB-D 相机模块。

职责：
- 在 PyBullet 场景中放置虚拟相机并拍摄 RGB-D 图像
- 将深度图反投影为三维点云（复用现有 CameraInfo / create_point_cloud_from_depth_image）
- 输出与 GraspNet 推理兼容的点云格式
"""

import sys
import os
import numpy as np

# 复用现有工具
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from data_utils import CameraInfo, create_point_cloud_from_depth_image

import torch


class VirtualCamera:
    """PyBullet 虚拟 RGB-D 相机。"""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fov: float = 60.0,
        near: float = 0.01,
        far: float = 5.0,
        position=(0.3, 0.0, 0.5),
        target=(0.3, 0.0, 0.05),
        up=(0.0, 1.0, 0.0),
    ):
        """
        Args:
            width, height: 图像分辨率。
            fov: 视场角（度）。
            near, far: 近/远裁剪面。
            position: 相机位置 (x, y, z)。
            target: 相机看向的目标点。
            up: 上方向。
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.position = position
        self.target = target
        self.up = up

        self.view_matrix = self._compute_view_matrix(position, target, up)
        self.proj_matrix = self._compute_proj_matrix(fov, near, far, width / height)

        # 用虚拟相机内参构造 CameraInfo（近似针孔模型）。fov 为垂直 FOV，焦距以 fy 为基准。
        fy = (height / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
        fx = fy
        cx = width / 2.0
        cy = height / 2.0
        self.camera_info = CameraInfo(width, height, fx, fy, cx, cy, scale=1.0)

    # ------------------------------------------------------------------
    # 拍摄
    # ------------------------------------------------------------------

    def capture(self):
        """
        拍摄一帧 RGB-D 图像。

        Returns:
            rgb:  (H, W, 4) uint8 RGBA 图像。
            depth: (H, W)   float32 深度缓冲区（米）。
            seg:  (H, W)   int32 分割掩码。
        """
        import pybullet as p

        # DIRECT 模式下必须用 ER_TINY_RENDERER（CPU），
        # ER_BULLET_HARDWARE_OPENGL 在无头模式下无法正确渲染物体。
        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        rgb = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)

        # PyBullet 深度缓冲区: 0=远裁剪面, 1=近裁剪面
        # 逆映射: linear_depth = far * near / (far - (far - near) * depth_buf)
        depth_buf = np.array(depth, dtype=np.float32).reshape(self.height, self.width)
        depth_metric = self.far * self.near / (self.far - (self.far - self.near) * depth_buf)

        seg = np.array(seg, dtype=np.int32).reshape(self.height, self.width)
        return rgb, depth_metric, seg

    # ------------------------------------------------------------------
    # 点云生成
    # ------------------------------------------------------------------

    def generate_point_cloud(self, depth: np.ndarray, num_points: int = 20000):
        """
        将深度图转换为世界坐标系点云 (Z=0为桌面, +Z向上)。

        Args:
            depth: (H, W) 深度图（米）。
            num_points: 采样点数。
        Returns:
            (1, num_points, 3) torch.Tensor，世界坐标系。
        """
        import pybullet as p

        H, W = depth.shape
        pos = np.array(self.position, dtype=np.float64)
        tgt = np.array(self.target, dtype=np.float64)
        up = np.array(self.up, dtype=np.float64)

        # 相机局部坐标系基向量（世界系下）
        forward = tgt - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        # 针孔模型反投影。PyBullet 的 computeProjectionMatrixFOV 用的是垂直 FOV
        # （沿 height），水平视场由 aspect=W/H 决定，因此焦距以 fy 为基准、fx=fy。
        fy = (H / 2.0) / np.tan(np.deg2rad(self.fov) / 2.0)
        fx = fy
        cx, cy = W / 2.0, H / 2.0
        xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))

        px = (xmap - cx) * depth / fx
        py = (ymap - cy) * depth / fy
        pz = depth

        # 过滤无效点
        valid = (np.isfinite(px) & np.isfinite(py) &
                 (depth > 0.01) & (depth < self.far - 0.1))
        px, py, pz = px[valid], py[valid], pz[valid]

        if len(px) == 0:
            raise RuntimeError("深度图中无有效点云，请检查相机位置。")

        # 相机系 → 世界系
        world_x = pos[0] + px*right[0] + py*cam_up[0] + pz*forward[0]
        world_y = pos[1] + px*right[1] + py*cam_up[1] + pz*forward[1]
        world_z = pos[2] + px*right[2] + py*cam_up[2] + pz*forward[2]
        cloud = np.column_stack([world_x, world_y, world_z])

        # 分开物体和桌面，确保物体点被保留
        obj_mask = cloud[:, 2] > 0.005  # Z>5mm 视为物体
        obj_pts = cloud[obj_mask]
        table_pts = cloud[~obj_mask]

        # 采样
        N_obj = len(obj_pts)
        # 物体点至少保留 min(N_obj, 1000) 个
        n_keep_obj = min(N_obj, max(1000, int(num_points * 0.1)))
        n_keep_table = num_points - n_keep_obj

        if N_obj > 0:
            idxs_obj = np.random.choice(N_obj, min(n_keep_obj, N_obj), replace=N_obj < n_keep_obj)
            sampled_obj = obj_pts[idxs_obj]
        else:
            sampled_obj = np.zeros((0, 3))

        if len(table_pts) > 0:
            idxs_table = np.random.choice(len(table_pts), min(n_keep_table, len(table_pts)), replace=len(table_pts) < n_keep_table)
            sampled_table = table_pts[idxs_table]
        else:
            sampled_table = np.zeros((0, 3))

        cloud_final = np.concatenate([sampled_obj, sampled_table], axis=0)
        if len(cloud_final) < num_points:
            # 补齐
            extra = np.random.choice(len(cloud), num_points - len(cloud_final), replace=True)
            cloud_final = np.concatenate([cloud_final, cloud[extra]], axis=0)
        elif len(cloud_final) > num_points:
            idxs = np.random.choice(len(cloud_final), num_points, replace=False)
            cloud_final = cloud_final[idxs]

        return torch.from_numpy(cloud_final[np.newaxis].astype(np.float32))

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_view_matrix(position, target, up):
        import pybullet as p
        return p.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=up,
        )

    @staticmethod
    def _compute_proj_matrix(fov, near, far, aspect):
        # aspect 必须与实际分辨率 (width/height) 一致，否则点云反投影会被系统性拉偏
        import pybullet as p
        return p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far,
        )
