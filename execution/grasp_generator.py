import sys
import os
import numpy as np

# 将组长的 ThinkGrasp 目录加入系统环境变量
THINKGRASP_PATH = "/home/admin128/qiuguanhe/ThinkGrasp"
if THINKGRASP_PATH not in sys.path:
    sys.path.append(THINKGRASP_PATH)

try:
    from grasp_detetor import Graspnet
except ImportError as e:
    print(f"[Grasp Module-Error] 导入组长代码失败: {e}")

def generate_optimal_grasp(rgb_image, mask, camera_intrinsics, pointcloud):
    print("[Grasp Module] Preparing to call standard GraspNet-1Billion...")
    print("[Grasp Module] Instantiating Graspnet class...")
    
    # ---------------- 核心修复区域 ----------------
    original_cwd = os.getcwd() # 1. 记住我们当前的执行路径
    os.chdir(THINKGRASP_PATH)  # 2. 临时“穿越”到组长的工作目录
    
    # 3. 实例化！此时组长代码里的相对路径 'models/...' 就能完美匹配了
    model = Graspnet()         
    
    os.chdir(original_cwd)     # 4. 加载成功后，立刻“穿越”回我们自己的目录
    # ----------------------------------------------
    
    print("[Grasp Module] Starting grasp computation and collision detection...")
    
    try:
        best_grasp = model.grasp_detection(pointcloud)
    except Exception as e:
        print(f"[Grasp Module-Warning] grasp_detection failed: {e}. Trying compute_grasp_pose...")
        best_grasp = model.compute_grasp_pose(pointcloud)
    
    if best_grasp is None or len(best_grasp) == 0:
        print("[Grasp Module-Warning] No safe or valid grasp poses found!")
        return None
        
    print(f"[Grasp Module-Success] Successfully generated valid 6D grasp candidates!")
    
    return best_grasp

if __name__ == "__main__":
    print("This module serves as the wrapper interface for calling GraspNet.")