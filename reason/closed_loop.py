"""Simple closed-loop simulator for the reasoning policy.

Each step classifies the current scene, runs the matching handler, removes the
selected object from the occlusion graph, and repeats until the target is
directly grasped or the loop stops.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .schemas import PerceptionOutput, GraspDecision, Branch
from .branch_judge.classifier import classify_branch
from .fully_visible import handle as handle_fully_visible
from .partially_visible import handle as handle_partially_visible
from .invisible import handle as handle_fully_occluded


@dataclass
class ClosedLoopResult:
    """Result of a closed-loop rollout."""
    target_molmo_id: int
    actions: list[GraspDecision] = field(default_factory=list)
    success: bool = False
    final_status: str = ""
    num_steps: int = 0


def simulate_remove(
    perception: PerceptionOutput,
    removed_mid: int,
) -> PerceptionOutput:
    """Return a new perception state after removing one object node."""
    if removed_mid not in perception.molmo_to_node:
        raise ValueError(f"cannot remove mid={removed_mid}: not in graph")

    removed_node = perception.molmo_to_node[removed_mid]

    new_graph = perception.occlusion_graph.copy()
    new_graph.remove_node(removed_node)

    new_molmo_to_node = {
        mid: n for mid, n in perception.molmo_to_node.items()
        if mid != removed_mid
    }
    new_node_info = {
        n: info for n, info in perception.node_info.items()
        if n != removed_node
    }

    return replace(
        perception,
        occlusion_graph=new_graph,
        molmo_to_node=new_molmo_to_node,
        node_info=new_node_info,
    )


def run_closed_loop(
    perception: PerceptionOutput,
    max_steps: int = 20,
    prior_prompt_mode: str | None = None,
    ranking_score: str | None = None,
) -> ClosedLoopResult:
    """Run the full branch -> action -> remove loop for one target."""
    target_mid = perception.target_molmo_id
    result = ClosedLoopResult(target_molmo_id=target_mid)
    current = replace(
        perception,
        prior_prompt_mode=prior_prompt_mode or perception.prior_prompt_mode,
        ranking_score=ranking_score or perception.ranking_score,
    )

    for step in range(max_steps):
        # Step 1: classify the current target state.
        try:
            branch, reason = classify_branch(current)
        except Exception as e:
            result.final_status = f"classify_error_step{step}: {e}"
            return result

        # Step 2: dispatch to the matching handler.
        if branch == Branch.FULLY_VISIBLE:
            try:
                decision = handle_fully_visible(current)
            except Exception as e:
                result.final_status = f"handler_error_step{step}: {e}"
                return result
            result.actions.append(decision)
            result.num_steps = step + 1

            # Fully visible should end by grasping the target itself.
            if decision.grasp_id == target_mid:
                result.success = True
                result.final_status = "target_grasped"
                return result
            else:
                result.final_status = (
                    f"unexpected_step{step}: fully_visible but "
                    f"grasp_id={decision.grasp_id} != target={target_mid}"
                )
                return result

        elif branch == Branch.PARTIALLY_OCCLUDED:
            try:
                decision = handle_partially_visible(current)
            except Exception as e:
                result.final_status = f"handler_error_step{step}: {e}"
                return result
            result.actions.append(decision)
            result.num_steps = step + 1

            if not decision.success or decision.grasp_id is None:
                result.final_status = (
                    f"handler_failed_step{step}: {decision.message}"
                )
                return result

            # Remove the chosen object and continue the rollout.
            try:
                current = simulate_remove(current, decision.grasp_id)
            except Exception as e:
                result.final_status = f"simulate_remove_error_step{step}: {e}"
                return result

        elif branch == Branch.FULLY_OCCLUDED:
            try:
                decision = handle_fully_occluded(current)
            except Exception as e:
                result.final_status = f"handler_error_step{step}: {e}"
                return result
            result.actions.append(decision)
            result.num_steps = step + 1

            if not decision.success or decision.grasp_id is None:
                result.final_status = (
                    f"handler_failed_step{step}: {decision.message}"
                )
                return result

            try:
                current = simulate_remove(current, decision.grasp_id)
            except Exception as e:
                result.final_status = f"simulate_remove_error_step{step}: {e}"
                return result

        else:  # FAULT
            result.final_status = (
                f"fault_branch_step{step}: target={target_mid}, reason={reason}"
            )
            result.num_steps = step
            return result

    result.final_status = f"max_steps_reached ({max_steps})"
    return result
