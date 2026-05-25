import numpy as np
import cv2
import os
from pointcloud_utils import generate_local_pointcloud
from grasp_generator import generate_optimal_grasp

def format_pointcloud_for_graspnet(o3d_pcd, target_points=20000):
    """
    Sample Open3D point cloud to fixed size (20000, 3) Numpy array.
    """
    points = np.asarray(o3d_pcd.points)
    # [Fixed] Extract the exact integer length (e.g., 1767) using 
    num_pts = points.shape  
    
    if num_pts == 0:
        print("[Warning] Empty point cloud.")
        return None
        
    print(f"[Data Format] Adjusting points from {num_pts} to {target_points}...")

    # Downsample or oversample
    replace_flag = num_pts < target_points
    sampled_indices = np.random.choice(num_pts, target_points, replace=replace_flag)
        
    return points[sampled_indices, :]

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

# 2. Convert to 3D point cloud
target_pcd = generate_local_pointcloud(real_color, real_depth, real_mask, real_intrinsics)

# 3. Format for GraspNet
graspnet_input = format_pointcloud_for_graspnet(target_pcd)

# 4. Generate Grasp Pose
if graspnet_input is not None:
    best_grasp = generate_optimal_grasp(graspnet_input)
    if best_grasp is not None:
        print("\n[Success] Optimal grasp pose found:")
        print(best_grasp)