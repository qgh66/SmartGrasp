from __future__ import annotations

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

    return GraspDecision(
        branch=Branch.FULLY_VISIBLE,
        grasp_id=target_mid,
        grasp_label=target_label,
        target_molmo_id=target_mid,
        is_terminal=True,
        success=True,
        message=f"target molmo_id={target_mid} ({target_label}) is fully visible, "
                f"directly graspable",
    )
