import numpy as np
import open3d as o3d
import cv2  # Used for data simulation image processing
import os   # Used for file path handling

def generate_local_pointcloud(color_img, depth_img, mask, intrinsics):
    """
    Generates a local 3D point cloud of the target object using RGB-D images, 
    a 2D mask, and camera intrinsics.
    
    Principles:
        Pinhole Camera Model Projection:
        Z = d / depth_scale
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    print("[Module] Converting 2D Mask to 3D local point cloud...")

    # Step 1: Apply Masking
    binary_mask = (mask > 0).astype(np.uint8) 
    masked_depth = cv2.bitwise_and(depth_img, depth_img, mask=binary_mask)
    
    # Step 2: Convert Numpy arrays to Open3D image format
    o3d_color = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    o3d_depth = o3d.geometry.Image(masked_depth)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, 
        o3d_depth, 
        depth_scale=1000.0, # RealSense default: 1000mm = 1m
        depth_trunc=3.0,    # Truncate points beyond 3 meters
        convert_rgb_to_intensity=False
    )
    
    # Step 3: Construct Camera Intrinsics
    height, width = depth_img.shape
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, 
        height, 
        intrinsics['fx'], 
        intrinsics['fy'], 
        intrinsics['cx'], 
        intrinsics['cy']
    )
    
    # Step 4: Deproject to generate 3D point cloud
    local_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        pinhole_camera_intrinsic
    )
    
    # Statistical outlier removal to filter out noise
    local_pcd, ind = local_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"[Module] Point cloud generated successfully. Total points: {len(local_pcd.points)}")
    return local_pcd


# =========================================================================
# Simulation Testing (Optimized for SSH remote server environments)
# =========================================================================
if __name__ == "__main__":
    print("--- Starting local simulation test (Real Intrinsics) ---")
    
    # 1. Standard lab resolution: 1280x720
    H, W = 720, 1280
    
    # Dummy color image: grey background with a blue square target
    dummy_color = np.ones((H, W, 3), dtype=np.uint8) * 100 
    cv2.rectangle(dummy_color, (500, 300), (800, 500), (255, 0, 0), -1) 
    
    # Dummy depth image: tabletop at 1000mm, object raised at 800mm
    dummy_depth = np.ones((H, W), dtype=np.uint16) * 1000
    cv2.rectangle(dummy_depth, (500, 300), (800, 500), 800, -1)
    
    # Dummy mask: simulated detection mask for the blue square
    dummy_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (500, 300), (800, 500), 1, -1)
    
    # 2. Real intrinsics for lab D435i camera
    real_intrinsics = {
        'fx': 913.54499375, 
        'fy': 915.195828, 
        'cx': 630.28532943, 
        'cy': 381.85590217
    }
    
    # 3. Call core function
    target_pointcloud = generate_local_pointcloud(
        color_img=dummy_color,
        depth_img=dummy_depth,
        mask=dummy_mask,
        intrinsics=real_intrinsics
    )
    
    # Step 4: Save results relative to the script location (execution/results)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, "results")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[Info] Target directory created: {output_dir}")
        
    output_filename = os.path.join(output_dir, "test_output_real_intrinsics.pcd")
    
    # Write point cloud to file
    o3d.io.write_point_cloud(output_filename, target_pointcloud)
    
    print(f"--- Test Complete ---")
    print(f"Point cloud saved to: {output_filename}")
    print("Download the file via VS Code, Xftp, or scp to view it in a local 3D viewer.")