import time
from execution_api import get_grasp_pose

print("=== Execution Module: Batch Target Testing ===")
print("Loading standard API to process target ID 1 to 16...\n")

# Loop through all 16 target objects in the directory
for target_id in range(1, 17):
    print(f"--- Processing Target ID: {target_id} ---")
    start_time = time.time()
    
    # [Action] Call your newly encapsulated API
    pose = get_grasp_pose(target_id=target_id)
    
    if pose is not None:
        print(f"[Success] Optimal Pose Found for ID {target_id}:")
        print(f"  Score:       {pose['score']:.4f}")
        print(f"  Width:       {pose['width']:.4f}")
        print(f"  Translation: {pose['translation']}")
        # Not printing rotation_matrix to keep terminal clean, but it's safely in the dict
    else:
        print(f"[Warning] Failed or no valid point cloud for ID {target_id}.")
        
    print(f"  Time cost:   {time.time() - start_time:.2f}s\n")

print("=== Batch Testing Completed ===")