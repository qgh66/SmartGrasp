import sys
import os

THINKGRASP_PATH = "/home/admin128/qiuguanhe/ThinkGrasp"
if THINKGRASP_PATH not in sys.path:
    sys.path.append(THINKGRASP_PATH)

try:
    from grasp_detetor import Graspnet
except ImportError as e:
    print(f"[Grasp Module-Error] Import failed: {e}")

def generate_optimal_grasp(pointcloud):
    """
    Load GraspNet-1Billion and generate optimal 6-DoF grasp pose.
    """
    print("[Grasp Module] Preparing standard GraspNet-1Billion...")
    
    # Path trick to load relative weights in ThinkGrasp
    original_cwd = os.getcwd()
    os.chdir(THINKGRASP_PATH)
    model = Graspnet()
    os.chdir(original_cwd)
    
    print("[Grasp Module] Computing grasps...")
    
    # Fallback mechanism for reasoning
    try:
        best_grasp = model.grasp_detection(pointcloud)
    except Exception as e:
        print(f"[Grasp Module-Warning] grasp_detection failed: {e}. Trying compute_grasp_pose...")
        best_grasp = model.compute_grasp_pose(pointcloud)
    
    if best_grasp is None or len(best_grasp) == 0:
        print("[Grasp Module-Warning] No valid grasp poses found.")
        return None
        
    print("[Grasp Module-Success] Valid 6D grasp candidates generated.")
    return best_grasp

if __name__ == "__main__":
    print("This module serves as the wrapper interface for GraspNet.")
