import numpy as np
import cv2
import os
from pointcloud_utils import generate_local_pointcloud
from grasp_generator import generate_optimal_grasp

print("=== Execution Module Integration Test ===")

# 1. Load real data from ThinkGrasp directory
DATA_DIR = "/home/admin128/qiuguanhe/ThinkGrasp/"
color_path = os.path.join(DATA_DIR, "color_map.png")
depth_path = os.path.join(DATA_DIR, "height_map.png")
mask_path = os.path.join(DATA_DIR, "image_mask_1.png")

print(f"[Load] RGB: {color_path}")
real_color = cv2.imread(color_path)
print(f"[Load] Depth: {depth_path}")
real_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
print(f"[Load] Mask: {mask_path}")
real_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Safely binarize the mask
real_mask = (real_mask > 0).astype(np.uint8) * 255

# RealSense D435i Intrinsics
real_intrinsics = {
    'fx': 913.54499375, 'fy': 915.195828, 
    'cx': 630.28532943, 'cy': 381.85590217
}

# 2. Convert to 3D point cloud (Returns an Open3D PointCloud object)
target_pcd = generate_local_pointcloud(real_color, real_depth, real_mask, real_intrinsics)

# 3. Generate Grasp Pose
# [Fixed] Pass the Open3D target_pcd directly. ThinkGrasp expects Open3D, not Numpy!
if target_pcd is not None and len(target_pcd.points) > 0:
    best_grasp = generate_optimal_grasp(target_pcd)
    if best_grasp is not None:
        print("\n[Success] Optimal grasp pose found:")
        print(best_grasp)
else:
    print("[Warning] Point cloud is empty. Cannot generate grasps.")