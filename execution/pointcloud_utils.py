import numpy as np
import open3d as o3d
import cv2

def generate_local_pointcloud(color_img, depth_img, mask, intrinsics):
    """
    Convert 2D masked RGB-D to 3D local point cloud.
    """
    print("[Module] Converting 2D Mask to 3D local point cloud...")
    
    # Apply mask to depth image
    binary_mask = (mask > 0).astype(np.uint8)
    masked_depth = cv2.bitwise_and(depth_img, depth_img, mask=binary_mask)
    
    # Create Open3D RGBD image
    o3d_color = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    o3d_depth = o3d.geometry.Image(masked_depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
    )
    
    # Define camera intrinsics
    height, width = depth_img.shape
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    )
    
    # Generate point cloud and remove noise
    local_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
    local_pcd, _ = local_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"[Module] Point cloud generated. Total points: {len(local_pcd.points)}")
    return local_pcd

if __name__ == "__main__":
    print("This module provides 3D point cloud utilities.")