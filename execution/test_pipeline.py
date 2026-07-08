import numpy as np
import cv2
import os
from pointcloud_utils import generate_local_pointcloud
from grasp_generator import generate_optimal_grasp

def select_best_grasp(grasp_group, target_centroid=None):
    """
    Select the optimal grasp pose from the candidates.
    If target_centroid is provided, it uses the AffordGrasp distance penalty formula.
    Otherwise, it returns the top-1 pose based purely on physical stability score.
    """
    if grasp_group is None or len(grasp_group) == 0:
        return None
        
    if target_centroid is None:
        # GraspGroup is sorted by score descendingly.
        top_index = 0
        return grasp_group[top_index]
    else:
        # Formula: argmax (score(g) - ||t(g) - c||_2)
        best_idx = 0
        max_score = -float('inf')
        for i, g in enumerate(grasp_group):
            penalty = np.linalg.norm(g.translation - target_centroid)
            final_score = g.score - penalty
            if final_score > max_score:
                max_score = final_score
                best_idx = i
        return grasp_group[best_idx]

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

# 3. Generate and Filter Grasp Pose
if target_pcd is not None and len(target_pcd.points) > 0:
    all_grasps = generate_optimal_grasp(target_pcd)
    
    # Select the absolute best pose from the candidates
    best_grasp = select_best_grasp(all_grasps)
    
    if best_grasp is not None:
        print("\n[Success] Final Top-1 Executable Grasp Pose:")
        print(f"Score:       {best_grasp.score:.4f}")
        print(f"Width:       {best_grasp.width:.4f}")
        print(f"Translation: {best_grasp.translation}")
        # [Fixed] Correct attribute name is 'rotation_matrix'
        print(f"Rotation:\n{best_grasp.rotation_matrix}")
else:
    print("[Warning] Point cloud is empty. Cannot generate grasps.")