import numpy as np
import cv2
import os
from pointcloud_utils import generate_local_pointcloud
from grasp_generator import generate_optimal_grasp

def select_best_grasp(grasp_group, target_centroid=None):
    """
    Select the optimal grasp pose from the candidates.
    """
    if grasp_group is None or len(grasp_group) == 0:
        return None
        
    if target_centroid is None:
        top_index = 0
        return grasp_group[top_index]
    else:
        best_idx = 0
        max_score = -float('inf')
        for i, g in enumerate(grasp_group):
            penalty = np.linalg.norm(g.translation - target_centroid)
            final_score = g.score - penalty
            if final_score > max_score:
                max_score = final_score
                best_idx = i
        return grasp_group[best_idx]

def get_grasp_pose(target_id, target_centroid=None, data_dir="/home/admin128/qiuguanhe/ThinkGrasp/"):
    """
    Standard API for the Reasoning Module.
    Returns a dictionary containing the optimal 6-DoF grasp pose parameters.
    """
    color_path = os.path.join(data_dir, "color_map.png")
    depth_path = os.path.join(data_dir, "height_map.png")
    mask_path = os.path.join(data_dir, f"image_mask_{target_id}.png")

    if not os.path.exists(mask_path):
        print(f"[Execution API] Error: Mask file for ID {target_id} not found at {mask_path}")
        return None

    # 1. Load Data
    real_color = cv2.imread(color_path)
    real_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    real_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    real_mask = (real_mask > 0).astype(np.uint8) * 255

    # RealSense D435i Intrinsics
    real_intrinsics = {
        'fx': 913.54499375, 'fy': 915.195828, 
        'cx': 630.28532943, 'cy': 381.85590217
    }

    # 2. Generate Point Cloud
    target_pcd = generate_local_pointcloud(real_color, real_depth, real_mask, real_intrinsics)

    # 3. Generate and Select Best Grasp
    if target_pcd is not None and len(target_pcd.points) > 0:
        all_grasps = generate_optimal_grasp(target_pcd)
        best_grasp = select_best_grasp(all_grasps, target_centroid)
        
        if best_grasp is not None:
            return {
                "score": best_grasp.score,
                "width": best_grasp.width,
                "translation": best_grasp.translation,
                "rotation_matrix": best_grasp.rotation_matrix
            }
            
    print(f"[Execution API] Warning: Failed to generate grasp for ID {target_id}.")
    return None

# use the codes below to connect this API function
# from execution_api import get_grasp_pose
# pose = get_grasp_pose(target_id=2)