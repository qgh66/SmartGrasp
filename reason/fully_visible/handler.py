from __future__ import annotations

from ..graspability import score_current_objects
from ..schemas import PerceptionOutput, GraspDecision, Branch


def handle(perception: PerceptionOutput) -> GraspDecision:
    target_mid = perception.target_molmo_id

    # Guard against branch/classifier mismatch.
    target_node = perception.molmo_to_node.get(target_mid)
    if target_node is None:
        return GraspDecision(
            branch=Branch.FULLY_VISIBLE,
            target_molmo_id=target_mid,
            is_terminal=False,
            success=False,
            message=f"target molmo_id={target_mid} unexpectedly not in graph; "
                    f"branch_judge inconsistency",
        )

    target_label = perception.node_info[target_node].get("label", "")
    prompt_mode = getattr(perception, "prior_prompt_mode", "original")
    if prompt_mode == "graspability":
        graspability_payload = score_current_objects([target_mid], perception)
    else:
        graspability_payload = {
            "graspability": {target_mid: 1.0},
            "graspability_part_id": {target_mid: None},
            "graspability_parts": {target_mid: {}},
            "reason": "original mode; graspability disabled",
        }
    graspability = graspability_payload.get("graspability", {}).get(target_mid, 1.0)
    graspability_part_id = graspability_payload.get("graspability_part_id", {}).get(target_mid)
    graspability_parts = graspability_payload.get("graspability_parts", {}).get(target_mid, {})
    vlm_reason = str(graspability_payload.get("reason") or "")

    return GraspDecision(
        branch=Branch.FULLY_VISIBLE,
        grasp_id=target_mid,
        grasp_label=target_label,
        target_molmo_id=target_mid,
        is_terminal=True,
        success=True,
        message=f"target molmo_id={target_mid} ({target_label}) is fully visible, "
                f"directly graspable; G={graspability:.3f} "
                f"best_part={graspability_part_id}; vlm_reason={vlm_reason}",
        details={
            target_mid: {
                "graspability": graspability,
                "graspability_part_id": graspability_part_id,
                "graspability_parts": graspability_parts,
                "score": graspability,
                "vlm_reason": vlm_reason,
            }
        },
    )
