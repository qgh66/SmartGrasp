#!/usr/bin/env python
"""
Phase 1 仿真主入口。

串联整个闭环流程：
  PyBullet 场景 → 虚拟相机 → 点云 → GraspNet 推理 → 逐抓取执行 → 评估

用法：
  conda activate smartgrasp
  python -m simulation.run_sim --obj_path /path/to/mesh.obj --checkpoint_path /path/to/checkpoint.tar
"""

import sys
import os
import argparse
import json
import numpy as np
import torch

# 将项目根目录加入 path，以便复用现有模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "models"))
sys.path.insert(0, os.path.join(ROOT_DIR, "graspnetAPI"))

from graspnetAPI.grasp import GraspGroup
from models.graspnet import GraspNet, pred_decode

from simulation.scene import SimulationScene
from simulation.camera import VirtualCamera
from simulation.gripper import ParallelJawGripper
from simulation.evaluator import GraspEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="GraspNet + PyBullet 仿真验证")
    parser.add_argument("--obj_path", type=str, required=True,
                        help="物体 .obj 路径")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="GraspNet 模型 checkpoint 路径")
    parser.add_argument("--num_point", type=int, default=20000,
                        help="点云采样点数")
    parser.add_argument("--num_view", type=int, default=300,
                        help="视角数量")
    parser.add_argument("--top_k", type=int, default=10,
                        help="评估前 K 个抓取")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="是否打开 GUI")
    parser.add_argument("--output", type=str, default="results.json",
                        help="结果输出 JSON 文件路径")
    parser.add_argument("--device", type=str, default="cpu",
                        help="推理设备: cpu / cuda:0")
    parser.add_argument("--random_orientation", action="store_true", default=False,
                        help="随机旋转物体朝向")
    return parser.parse_args()


def load_graspnet(checkpoint_path: str, num_view: int, device: torch.device):
    """加载预训练 GraspNet 模型。"""
    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()
    print(f"[GraspNet] 模型已加载 (epoch {checkpoint.get('epoch', '?')})")
    return net


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # ================================================================
    # 1. 初始化 PyBullet 场景
    # ================================================================
    scene = SimulationScene(gui=args.gui)
    scene.connect()
    scene.load_plane()

    # 物体朝向: 固定或随机
    from scipy.spatial.transform import Rotation as R
    if args.random_orientation:
        r = R.random()
        orientation = tuple(r.as_quat()[[0, 1, 2, 3]])  # xyzw
    else:
        orientation = (0, 0, 0, 1)

    # 物体初始 Z 坐标: 根据实际网格大小估算
    obj_id = scene.load_object(args.obj_path, position=(0.3, 0.0, 0.05),
                               orientation=orientation)
    print(f"[Scene] 物体 ID={obj_id} 已加载: {args.obj_path} orientation={orientation}")

    # 等物体稳定
    for _ in range(300):
        scene.step()

    # ================================================================
    # 2. 虚拟相机拍摄 + 点云生成
    # ================================================================
    camera = VirtualCamera(
        position=(0.3, 0.0, 1.2),
        target=(0.3, 0.0, 0.2),
        up=(0, 1, 0),
    )
    rgb, depth, seg = camera.capture()
    cloud_tensor = camera.generate_point_cloud(depth, num_points=args.num_point)
    cloud_tensor = cloud_tensor.to(device)
    print(f"[Camera] 点云形状: {cloud_tensor.shape}")

    # ---- 保存可视化中间数据 ----
    import pickle
    viz_data = {
        "rgb": rgb,                        # (H, W, 4) RGBA
        "depth": depth,                    # (H, W) float32 (米)
        "point_cloud": cloud_tensor.cpu().numpy(),  # (1, N, 3)
    }
    viz_path = os.path.join(ROOT_DIR, "results", "viz_data.pkl")
    with open(viz_path, "wb") as f:
        pickle.dump(viz_data, f)
    print(f"[Viz] 场景数据已保存: {viz_path}")

    # ================================================================
    # 3. GraspNet 推理
    # ================================================================
    net = load_graspnet(args.checkpoint_path, args.num_view, device)
    end_points = {"point_clouds": cloud_tensor}
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    gg.sort_by_score()
    print(f"[GraspNet] 生成 {len(gg)} 个候选抓取, 最高分: {gg[0].score:.4f}")

    # ================================================================
    # 4. 加载夹爪 + 评估
    # ================================================================
    gripper = ParallelJawGripper()
    gripper.load()
    evaluator = GraspEvaluator(object_id=obj_id, gripper=gripper)

    results = evaluator.evaluate(
        grasp_group=gg,
        top_k=args.top_k,
    )

    # ================================================================
    # 5. 输出结果
    # ================================================================
    success_count = sum(1 for r in results if r["success"])
    print(f"\n[结果] {success_count}/{len(results)} 抓取成功")

    # 保存为 JSON（将 numpy array 转为 list）
    output = []
    for r in results:
        output.append({
            "grasp_index": r["grasp_index"],
            "success": r["success"],
            "score": float(r["score"]),
            "lift_z": float(r["lift_z"]),
            "width": float(r["width"]),
            "depth": float(r["depth"]),
            "translation": r["translation"].tolist(),
            "rotation": r["rotation"].tolist(),
        })

    output_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[保存] 结果已写入 {output_path}")

    # ================================================================
    # 6. 清理
    # ================================================================
    gripper.remove()
    scene.disconnect()


if __name__ == "__main__":
    main()
