import numpy as np
import open3d as o3d
import cv2 # 仅用于模拟数据时的图像处理
import os  # 用于处理文件路径和创建文件夹

def generate_local_pointcloud(color_img, depth_img, mask, intrinsics):
    """
    根据 RGB-D 图像和 2D 掩码，结合相机内参，生成只包含目标物体的局部 3D 点云。
    
    【核心原理说明】：
    针孔相机模型 (Pinhole Camera Model) 的投影几何原理：
    对于图像上的任意一个像素点 (u, v)，以及它对应的深度值 d (即 Z 轴距离)：
    三维空间中的真实坐标 (X, Y, Z) 计算公式为：
        Z = d / 深度比例尺 (depth_scale，通常为 1000)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    这也就是所谓的 "反向投影" (Deprojection)。
    
    参数:
        color_img (numpy.ndarray): 彩色图像，形状为 (H, W, 3)
        depth_img (numpy.ndarray): 深度图像，形状为 (H, W)，单位通常为毫米
        mask (numpy.ndarray): 目标物体的 2D 掩码，形状为 (H, W)，包含0和1(或255)
        intrinsics (dict): 相机内参字典，需包含 'fx', 'fy', 'cx', 'cy'
        
    返回:
        local_pcd (open3d.geometry.PointCloud): 裁剪后的局部 3D 点云对象
    """
    print("[执行模块] 正在执行 2D Mask 到 3D 局部点云的转换...")

    # ==========================================
    # 步骤 1: 应用掩码 (Masking)
    # ==========================================
    # 确保 mask 为布尔或 0/1 格式
    binary_mask = (mask > 0).astype(np.uint8) 
    
    # 过滤掉不需要的背景深度 (利用按位与操作)
    masked_depth = cv2.bitwise_and(depth_img, depth_img, mask=binary_mask)
    
    # ==========================================
    # 步骤 2: 将 Numpy 数组转为 Open3D 图像格式
    # ==========================================
    o3d_color = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    o3d_depth = o3d.geometry.Image(masked_depth)
    
    # 将彩色图和深度图打包成一个 RGBDImage 对象
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, 
        o3d_depth, 
        depth_scale=1000.0, # RealSense 默认 1000mm = 1m
        depth_trunc=3.0,    # 截断距离：丢弃距离相机 3 米以外的点
        convert_rgb_to_intensity=False
    )
    
    # ==========================================
    # 步骤 3: 构造相机内参对象 (Camera Intrinsics)
    # ==========================================
    height, width = depth_img.shape
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, 
        height, 
        intrinsics['fx'], 
        intrinsics['fy'], 
        intrinsics['cx'], 
        intrinsics['cy']
    )
    
    # ==========================================
    # 步骤 4: 执行反向投影，生成 3D 点云
    # ==========================================
    local_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        pinhole_camera_intrinsic
    )
    
    # 对点云进行下采样或统计滤波，去除离群飞点噪点
    local_pcd, ind = local_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"[执行模块] 成功生成局部点云，当前点云数量: {len(local_pcd.points)} 个点。")
    return local_pcd


# =========================================================================
# 模拟测试代码 (专为 SSH 远程服务器环境修改)
# =========================================================================
if __name__ == "__main__":
    print("--- 开始本地模拟测试 (基于真实内参) ---")
    
    # 1. 采用实验室真实的 1280x720 分分辨率
    H, W = 720, 1280
    
    # 假彩色图：全灰色背景，中间有个蓝色方块代表物体
    dummy_color = np.ones((H, W, 3), dtype=np.uint8) * 100 
    # 坐标适配 1280x720，画在画面偏中心的位置
    cv2.rectangle(dummy_color, (500, 300), (800, 500), (255, 0, 0), -1) 
    
    # 假深度图：整个桌面距离相机 1000mm，方块凸起距离相机 800mm
    dummy_depth = np.ones((H, W), dtype=np.uint16) * 1000
    cv2.rectangle(dummy_depth, (500, 300), (800, 500), 800, -1)
    
    # 假掩码：假装感知模块发来了这个蓝色方块的掩码
    dummy_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (500, 300), (800, 500), 1, -1)
    
    # 2. 填入实验室 D435i 相机的真实内参 (中国制造版)
    real_intrinsics = {
        'fx': 913.54499375, 
        'fy': 915.195828, 
        'cx': 630.28532943, 
        'cy': 381.85590217
    }
    
    # 3. 调用核心函数
    target_pointcloud = generate_local_pointcloud(
        color_img=dummy_color,
        depth_img=dummy_depth,
        mask=dummy_mask,
        intrinsics=real_intrinsics
    )
    
    # =========================================================================
    # 步骤 4: 结果保存（修改后：通过脚本绝对路径定位，锁定保存到 execution/results）
    # =========================================================================
    # 获取当前脚本 pointcloud_utils.py 所在的绝对路径目录 (即 ~/sangxiyuan/SmartGrasp/execution)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 锁定目标文件夹为当前脚本目录下的 results 文件夹
    output_dir = os.path.join(current_script_dir, "results")
    
    # 自动检查并创建 execution/results 文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[提示] 成功创建了目标文件夹: {output_dir}")
        
    # 安全地拼接最终的绝对路径文件名
    output_filename = os.path.join(output_dir, "test_output_real_intrinsics.pcd")
    
    # 写入文件
    o3d.io.write_point_cloud(output_filename, target_pointcloud)
    
    print(f"--- 测试完成！ ---")
    print(f"点云文件已成功保存至: {output_filename}")
    print("请使用 VS Code、Xftp 或 scp 命令将该文件下载到您的个人电脑，并用本地的 3D 软件打开查看！")