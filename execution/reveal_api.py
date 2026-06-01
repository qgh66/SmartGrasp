import numpy as np

def execute_reveal_action(occluder_id, center_point, action_type="push"):
    """
    Reveal Module: Execute 3-5cm micro-moves away from targets via pushing or pick-and-place.
    Bypassing heavy neural networks for occluders using geometric center heuristics.
    """
    print("\n==================================================")
    print(f"🤖 [Reveal Module] Initiating micro-movement strategy")
    print(f"-> Target Occluder ID: {occluder_id}")
    print(f"-> Action Type: {action_type.upper()}")
    print("==================================================")

    # 1. Parse geometric center from upstream module
    translation = np.array(center_point, dtype=np.float32)
    
    # FIX: Corrected indexing to extract individual X, Y, Z elements
    print(f"[Step 1] Using occluder center: X={translation[0]:.3f}, Y={translation[1]:.3f}, Z={translation[2]:.3f}")

    # 2. Default top-down gripper rotation matrix
    rotation = np.eye(3) 

    # 3. Apply 5cm nudge strategy along X-axis
    move_dist = 0.05
    target_translation = np.copy(translation)

    print("\n[Step 2] Applying micro-movement strategy...")
    if action_type == "push":
        print(f"-> Executing PUSH: Shifting along X-axis by {move_dist * 100} cm.")
        # FIX: Modify index 0 only, avoiding unintended Y/Z shifting
        target_translation[0] += move_dist  
    elif action_type == "pick_and_place":
        print(f"-> Executing PICK & PLACE: Shifting along X-axis by {move_dist * 100} cm.")
        target_translation[0] += move_dist  

    # FIX: Corrected indexing for target translation printout
    print(f"[Ready] Target Position: X={target_translation[0]:.3f}, Y={target_translation[1]:.3f}, Z={target_translation[2]:.3f}")

    # 4. Trigger closed-loop pipeline update
    print("\n[Step 3] Triggering Closed-loop System...")
    print("🔄 [RE-LOOPS] Requesting RGB-D image update and scene state refresh...")
    print("==================================================\n")

    return {
        "status": "success",
        "action_executed": action_type,
        "new_translation": target_translation.tolist(),
        "default_rotation": rotation.tolist(),
        "request_reloop": True
    }


# ================= Testing Block =================
if __name__ == "__main__":
    print("--- Running Test for Reveal API ---")
    
    # Mock upstream occluder center coordinates
    dummy_center = [0.20, 0.30, 0.00]
    
    execute_reveal_action(
        occluder_id=4,
        center_point=dummy_center,
        action_type="pick_and_place"
    )