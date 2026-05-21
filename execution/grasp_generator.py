import sys
import numpy as np
from grasp_model import grasp_model

FREEGRASP_PATH = "/home/admin128/qiuguanhe/FreeGrasp"
if FREEGRASP_PATH not in sys.path:
    sys.path.append(FREEGRASP_PATH)


class GraspArgs:
    def __init__(self):
        self.num_view = 1                      
        # 指向全局共享的 ThinkGrasp 权重文件
        self.checkpoint_grasp_path = "/home/admin128/qiuguanhe/ThinkGrasp/models/graspnet/logs/log_rs/checkpoint.tar" 
        self.collision_thresh = 0.01           
        self.viz = False                       
        self.voxel_size = 0.01                 

def generate_optimal_grasp(rgb_image, mask, camera_intrinsics, pointcloud):
    print("[执行模块] 正在准备调用预训练的 GraspNet...")
    args = GraspArgs()
    device = 'cuda'  
    
    print("[执行模块] 实例化 grasp_model...")
    model = grasp_model(args=args, device=device, image=rgb_image, mask=mask, camera_info=camera_intrinsics)
    
    print("[执行模块] 加载神经网络权重...")
    model.load_grasp_net()
    
    print("[执行模块] 开始计算抓取并进行碰撞检测...")
    endpoint = {}  
    path = ""      
    
    # 核心前向推理
    gg, _ = model.forward(endpoint, pointcloud, path)
    
    if len(gg) == 0:
        print("[执行模块-警告] 未能找到任何安全/有效的抓取位姿！")
        return None
        
    print(f"[执行模块-成功] 成功生成并过滤出 {len(gg)} 个有效的 6D 抓取候选！")
    best_grasp = gg # 获取最高分抓取位姿
    return best_grasp

if __name__ == "__main__":
    print("此模块为 GraspNet 封装调用接口。")