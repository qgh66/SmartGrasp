#!/usr/bin/env python
"""
GraspNet 模型 Demo — 纯推理演示（无需 PyBullet）。

用法:
  conda activate smartgrasp
  cd /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace
  python scripts/demo_inference.py
"""

import os, sys

# 设置工作区根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'models'))
sys.path.insert(0, os.path.join(ROOT, 'pointnet2'))
sys.path.insert(0, os.path.join(ROOT, 'utils'))
sys.path.insert(0, os.path.join(ROOT, 'knn'))
sys.path.insert(0, os.path.join(ROOT, 'graspnet_api'))

import numpy as np
import torch
import open3d as o3d

from models.graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from utils.collision_detector import ModelFreeCollisionDetector


def create_dummy_pointcloud(num_points=20000):
    """创建虚拟点云用于测试。"""
    # 在桌面上方生成一个球形物体点云
    r = 0.03
    center = np.array([0.02, 0.0, 0.05])
    points = center + r * np.random.randn(num_points, 3)
    # 裁剪到有效区域
    points = points[points[:, 2] > 0.005]
    # 随机采样至 num_points
    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]
    colors = np.ones_like(points) * 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1. 加载模型
    print("[1/4] 加载 GraspNet...")
    ckpt_path = os.path.join(ROOT, 'checkpoints', 'checkpoint-rs.tar')
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️ checkpoint 不存在: {ckpt_path}")
        print(f"  请将 checkpoint 放到该路径下后重试。")
        return

    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02,
                   hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.to(device)
    net.eval()
    print(f"  ✅ epoch {ckpt.get('epoch', '?')}")

    # 2. 创建虚拟点云
    print("[2/4] 创建虚拟点云...")
    pcd = create_dummy_pointcloud()
    pcd = pcd.voxel_down_sample(0.001)
    points = np.asarray(pcd.points)
    print(f"  points: {points.shape}")

    # 3. 推理
    print("[3/4] 推理中...")
    cloud_tensor = torch.from_numpy(points[np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        end_points = net({'point_clouds': cloud_tensor})
        grasp_preds = pred_decode(end_points)

    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    gg.sort_by_score()
    print(f"  ✅ 生成 {len(gg)} 个候选抓取")
    print(f"  Top-5 得分: {[f'{gg[i].score:.4f}' for i in range(min(5, len(gg)))]}")

    # 4. 碰撞检测
    print("[4/4] 碰撞检测...")
    mfcd = ModelFreeCollisionDetector(points, voxel_size=0.01)
    collision_mask = mfcd.detect(gg, approach_dist=0.05, collision_thresh=0.01, empty_thresh=0.15)
    gg = gg[~collision_mask]
    print(f"  ✅ 碰撞过滤后剩余 {len(gg)} 个")
    if len(gg) > 0:
        print(f"  最佳抓取: score={gg[0].score:.4f}, width={gg[0].width:.4f}, "
              f"center=({gg[0].translation[0]:.3f},{gg[0].translation[1]:.3f},{gg[0].translation[2]:.3f})")

    print("\n✅ Demo 完成！")


if __name__ == '__main__':
    main()
