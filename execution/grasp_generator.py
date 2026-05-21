import sys
import numpy as np
from grasp_model import grasp_model

FREEGRASP_PATH = "/home/admin128/qiuguanhe/FreeGrasp"
if FREEGRASP_PATH not in sys.path:
    sys.path.append(FREEGRASP_PATH)


class GraspArgs:
    def __init__(self):
        self.num_view = 1                      
        # Path to the shared ThinkGrasp weight file
        self.checkpoint_grasp_path = "/home/admin128/qiuguanhe/ThinkGrasp/models/graspnet/logs/log_rs/checkpoint.tar" 
        self.collision_thresh = 0.01           
        self.viz = False                       
        self.voxel_size = 0.01                 

def generate_optimal_grasp(rgb_image, mask, camera_intrinsics, pointcloud):
    print("[Grasp Module] Preparing to call pretrained GraspNet...")
    args = GraspArgs()
    device = 'cuda'  
    
    print("[Grasp Module] Instantiating grasp_model...")
    model = grasp_model(args=args, device=device, image=rgb_image, mask=mask, camera_info=camera_intrinsics)
    
    print("[Grasp Module] Loading neural network weights...")
    model.load_grasp_net()
    
    print("[Grasp Module] Starting grasp computation and collision detection...")
    endpoint = {}  
    path = ""      
    
    # Core forward inference
    gg, _ = model.forward(endpoint, pointcloud, path)
    
    if len(gg) == 0:
        print("[Grasp Module-Warning] No safe or valid grasp poses found!")
        return None
        
    print(f"[Grasp Module-Success] Successfully generated and filtered {len(gg)} valid 6D grasp candidates!")
    best_grasp = gg # Get the highest scoring grasp pose
    return best_grasp

if __name__ == "__main__":
    print("This module serves as the wrapper interface for calling GraspNet.")