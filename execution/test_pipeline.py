import numpy as np
import cv2
import open3d as o3d
from pointcloud_utils import generate_local_pointcloud
from grasp_generator import generate_optimal_grasp

print("=== SmartGrasp 执行模块集成测试 ===")

H, W = 720, 1280
dummy_color = np.ones((H, W, 3), dtype=np.uint8) * 100 
cv2.rectangle(dummy_color, (500, 300), (800, 500), (255, 0, 0), -1) 
dummy_depth = np.ones((H, W), dtype=np.uint16) * 1000
cv2.rectangle(dummy_depth, (500, 300), (800, 500), 800, -1)
dummy_mask = np.zeros((H, W), dtype=np.uint8)
cv2.rectangle(dummy_mask, (500, 300), (800, 500), 1, -1)

real_intrinsics = {'fx': 913.54499375, 'fy': 915.195828, 'cx': 630.28532943, 'cy': 381.85590217}

target_pcd = generate_local_pointcloud(dummy_color, dummy_depth, dummy_mask, real_intrinsics)
best_grasp = generate_optimal_grasp(dummy_color, dummy_mask, real_intrinsics, target_pcd)

if best_grasp is not None:
    print("\n🎉 测试成功！找到了最优抓取位姿：")
    print(best_grasp)