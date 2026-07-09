"""Iterate sample_data/<scene>/perception/summary.json,
classify branches and dispatch handlers per (scene, target).

Outputs (default under runs/<model>/):
    results.csv              one row per (scene, target)
    branch_results.json      full per-scene structured results
    scene_details/scene_<id>.csv
                             candidate-level details for
                             partially_occluded targets
                             (P_s / P_g / P / IG / cost / score)

Usage:
    python test.py --root sample_data
    python test.py --root sample_data --model gpt-4o --closed-loop
"""
from dotenv import load_dotenv
load_dotenv()

# Parse --model early for backward-compatible CLI shape. The actual default
# model now lives in reason/vlm/config.py.
import argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--model", default=None)
_pre_args, _ = _pre.parse_known_args()

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from reason.data_loader import load_sample
from reason.branch_judge.classifier import classify_branch
from reason.schemas import Branch
from reason.fully_visible import handle as handle_fully_visible
from reason.partially_visible import handle as handle_partially_visible
from reason.invisible import handle as handle_fully_occluded
from reason.closed_loop import run_closed_loop
from reason.vlm import config as vlm_config
from reason.intent_handle import resolve_intent
from run_intent import (
    RUN_INTENT_API_KEY_ENV,
    RUN_INTENT_BASE_URL,
    RUN_INTENT_MODEL,
    RUN_INTENT_TIMEOUT,
)


def find_perception_summaries(root: Path):
    """Recursively find all perception/summary.json files."""
    return [
        (str(p.parent.relative_to(root)), p)
        for p in sorted(root.rglob("summary.json"))
        if p.parent.name == "perception"
    ]


def _sanitize_model_name(name: str) -> str:
    """Make a model name safe for file paths."""
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def _target_entries(args: argparse.Namespace, summary_path: Path, perception) -> list[dict]:
    """Resolve the target ids to evaluate for one scene."""
    source = args.target_source
    if source == "auto":
        source = "id" if args.target_id is not None else "all"

    if source == "all":
        return [
            {
                "target_id": mid,
                "target_source": "all",
                "intent_instruction": None,
                "intent_reason": None,
                "intent_candidate_ids": None,
                "intent_vlm_decision": None,
            }
            for mid in sorted(perception.molmo_to_node.keys())
        ]

    if source == "id":
        if args.target_id is None:
            raise ValueError("--target-source id requires --target-id")
        return [
            {
                "target_id": args.target_id,
                "target_source": "id",
                "intent_instruction": None,
                "intent_reason": None,
                "intent_candidate_ids": None,
                "intent_vlm_decision": None,
            }
        ]

    instruction = args.instruction or str(perception.annotation or "").strip()
    if not instruction:
        raise ValueError(f"--target-source intent requires --instruction or annotation in {summary_path}")

    result = resolve_intent(
        instruction,
        summary_path,
        api_key_env=args.intent_api_key_env,
        base_url=args.intent_base_url,
        model=args.intent_model,
        timeout=args.intent_timeout,
    )
    selected_id = result.target_object.object_id if result.target_object else None
    return [
        {
            "target_id": selected_id,
            "target_source": "intent",
            "intent_instruction": instruction,
            "intent_reason": result.reason,
            "intent_candidate_ids": [obj.object_id for obj in result.candidates],
            "intent_vlm_decision": result.vlm_decision,
        }
    ]


def _jsonable_part_scores(raw_parts) -> dict[str, float]:
    if not isinstance(raw_parts, dict):
        return {}
    out: dict[str, float] = {}
    for part_id, value in raw_parts.items():
        try:
            out[str(int(part_id))] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _selected_graspability_fields(decision) -> dict:
    fields = {
        "selected_object_id": None,
        "selected_object_graspability": None,
        "selected_object_graspability_part_id": None,
        "selected_object_graspability_parts": {},
    }
    if decision is None or getattr(decision, "grasp_id", None) is None:
        return fields

    selected_id = int(decision.grasp_id)
    fields["selected_object_id"] = selected_id
    details = getattr(decision, "details", None) or {}
    selected_info = details.get(selected_id, {})
    if not isinstance(selected_info, dict):
        return fields

    graspability = selected_info.get("graspability")
    try:
        fields["selected_object_graspability"] = None if graspability is None else float(graspability)
    except (TypeError, ValueError):
        fields["selected_object_graspability"] = None

    best_part_id = selected_info.get("graspability_part_id")
    try:
        fields["selected_object_graspability_part_id"] = (
            None if best_part_id is None else int(best_part_id)
        )
    except (TypeError, ValueError):
        fields["selected_object_graspability_part_id"] = None

    fields["selected_object_graspability_parts"] = _jsonable_part_scores(
        selected_info.get("graspability_parts")
    )
    return fields


def _selected_summary_row(row: dict) -> dict:
    return {
        "scene_key": row.get("scene_key"),
        "scene_id": row.get("scene_id"),
        "target_id": row.get("target_id"),
        "target_label": row.get("target_label"),
        "branch": row.get("branch"),
        "selected_object_id": row.get("selected_object_id"),
        "selected_object_graspability": row.get("selected_object_graspability"),
        "selected_object_graspability_part_id": row.get("selected_object_graspability_part_id"),
        "selected_object_graspability_parts": row.get("selected_object_graspability_parts", {}),
    }


def _reason_block(row: dict, decision, actions_seq=None) -> str:
    """Build one human-readable reason section for reason.txt."""
    lines = [
        f"scene_key: {row.get('scene_key')}",
        f"scene_id: {row.get('scene_id')}",
        f"target_source: {row.get('target_source')}",
        f"target_id: {row.get('target_id')}",
        f"target_label: {row.get('target_label')}",
        f"branch: {row.get('branch')}",
        f"grasp_id: {row.get('grasp_id')}",
        f"selected_object_graspability: {row.get('selected_object_graspability')}",
        f"selected_object_graspability_part_id: {row.get('selected_object_graspability_part_id')}",
        f"selected_object_graspability_parts: {row.get('selected_object_graspability_parts')}",
        f"status: {row.get('status')}",
    ]

    if row.get("target_source") == "intent":
        lines.append(f"intent_instruction: {row.get('intent_instruction')}")
        lines.append(f"intent_reason: {row.get('intent_reason')}")

    if actions_seq:
        lines.append("downstream_reason_seq:")
        for step_idx, action in enumerate(actions_seq, start=1):
            step_reason = str(getattr(action, "message", "") or "")
            lines.append(f"  step {step_idx}: {step_reason}")
    else:
        downstream_reason = ""
        if decision is not None and getattr(decision, "message", None):
            downstream_reason = str(decision.message)
        elif row.get("reason"):
            downstream_reason = str(row.get("reason"))
        lines.append(f"downstream_reason: {downstream_reason}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Batch branch classification + handler dispatch")
    parser.add_argument("--root", default="sample_data", help="Scene root directory")
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="Only run a specific scene_id, for example 564",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=None,
        help="Only run a specific target_id, including ids not present in the graph",
    )
    parser.add_argument(
        "--target-source",
        choices=["auto", "all", "id", "intent"],
        default="auto",
        help="Target source: all graph ids, direct --target-id, or run_intent-style VLM intent resolution.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction for --target-source intent. Defaults to summary annotation.",
    )
    parser.add_argument("--intent-api-key-env", default=RUN_INTENT_API_KEY_ENV)
    parser.add_argument("--intent-base-url", default=RUN_INTENT_BASE_URL)
    parser.add_argument("--intent-model", default=RUN_INTENT_MODEL)
    parser.add_argument("--intent-timeout", type=float, default=RUN_INTENT_TIMEOUT)
    parser.add_argument("--model", default=None,
                        help="VLM model name (overrides reason/vlm/config.py for this run)")
    parser.add_argument("--out-root", default="runs_detail",
                        help="Root for outputs; each model gets its own subdir")
    parser.add_argument("--csv", default=None,
                        help="Override summary csv path")
    parser.add_argument("--json", default=None,
                        help="Override summary json path")
    parser.add_argument("--details-dir", default=None,
                        help="Override scene_details dir")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Occlusion edge threshold")
    parser.add_argument(
        "--prior-prompt",
        choices=["original", "graspability"],
        default="original",
        help="VLM prior prompt variant. original is the old prompt; graspability also asks for part graspability.",
    )
    parser.add_argument(
        "--ranking-score",
        choices=["legacy", "ig", "ig_graspability", "theory"],
        default="legacy",
        help=(
            "Candidate ranking score. legacy keeps the original algorithm; "
            "theory uses the normalized utility from reason/theory.md."
        ),
    )
    parser.add_argument(
        "--reason-algorithm",
        choices=["legacy", "theory"],
        default=None,
        help=(
            "Convenience switch for old vs new reasoning. "
            "Equivalent to --ranking-score legacy/theory when set."
        ),
    )
    parser.add_argument("--closed-loop", action="store_true",
                        help="Closed-loop mode: full action sequence per target")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max steps in closed-loop (safety cap)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N scenes (debug)")
    args = parser.parse_args()

    if args.reason_algorithm is not None:
        args.ranking_score = args.reason_algorithm

    if args.model:
        vlm_config.VLM_MODEL = args.model

    model_name = vlm_config.VLM_MODEL
    model_safe = _sanitize_model_name(model_name)
    print(f"[CONFIG] using VLM_MODEL = {model_name}")
    print(f"[CONFIG] prior_prompt = {args.prior_prompt}, ranking_score = {args.ranking_score}")

    # Per-model output dirs.
    out_dir = Path(args.out_root) / model_safe / args.prior_prompt / args.ranking_score
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_dir / "results.csv"
    json_path = Path(args.json) if args.json else out_dir / "branch_results.json"
    details_dir = Path(args.details_dir) if args.details_dir else out_dir / "scene_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    reason_path = csv_path.parent / "reason.txt"
    summary_path = csv_path.parent / "summary.json"

    print(f"[CONFIG] outputs -> {out_dir}")

    root = Path(args.root)
    summaries = find_perception_summaries(root)
    if not summaries:
        print(f"[WARN] no perception/summary.json found under {root}")
        return

    if args.scene_id is not None:
        target_suffix = f"scene_{args.scene_id}/perception"
        summaries = [
            (scene_key, path)
            for scene_key, path in summaries
            if scene_key == target_suffix
        ]
        if not summaries:
            print(f"[WARN] scene_id={args.scene_id} not found under {root}")
            return

    csv_rows = []
    reason_blocks = []
    detail_json = {}
    selected_graspability_summary = []
    branch_counter = {}
    scene_count = 0
    object_count = 0

    for i, (scene_key, path) in enumerate(summaries):
        if args.limit and i >= args.limit:
            print(f"[LIMIT] reached {args.limit} scenes, stopping")
            break

        try:
            perception = load_sample(path, occlusion_threshold=args.threshold)
            perception = replace(
                perception,
                prior_prompt_mode=args.prior_prompt,
                ranking_score=args.ranking_score,
            )
        except Exception as e:
            print(f"  [ERROR] {scene_key}: load failed: {e}")
            continue

        scene_count += 1
        query_target_id = perception.target_molmo_id
        scene_detail_rows = []
        # Candidate-level rows for this scene's partially_occluded targets.
        scene_candidate_rows = []

        try:
            targets = _target_entries(args, path, perception)
        except Exception as e:
            print(f"  [ERROR] {scene_key}: target resolution failed: {e}")
            continue

        for target_entry in targets:
            mid = target_entry["target_id"]
            if mid is None:
                row = {
                    "model": model_name,
                    "prior_prompt": args.prior_prompt,
                    "ranking_score": args.ranking_score,
                    "target_source": target_entry["target_source"],
                    "intent_instruction": target_entry["intent_instruction"],
                    "intent_reason": target_entry["intent_reason"],
                    "intent_candidate_ids": target_entry["intent_candidate_ids"],
                    "intent_vlm_decision": target_entry.get("intent_vlm_decision"),
                    "scene_key": scene_key,
                    "scene_id": perception.scene_id,
                    "target_id": None,
                    "target_label": "none",
                    "is_query_target": False,
                    "annotation": perception.annotation,
                    "branch": None,
                    "grasp_id": None,
                    "grasp_label": None,
                    "is_terminal": None,
                    "reason": "run_intent returned no target object",
                    "status": "intent_no_target",
                }
                row.update(_selected_graspability_fields(None))
                csv_rows.append(row)
                scene_detail_rows.append(row)
                selected_graspability_summary.append(_selected_summary_row(row))
                reason_blocks.append(_reason_block(row, None))
                object_count += 1
                continue

            p = replace(perception, target_molmo_id=mid)

            # 1) Branch classification.
            try:
                branch, reason = classify_branch(p)
                branch_value = branch.value
                status = "ok"
            except Exception as e:
                branch = None
                branch_value = None
                reason = None
                status = f"classify_error: {e}"

            # 2) Single-step vs closed-loop.
            decision = None
            actions_seq = None
            cl_num_steps = None
            cl_success = None
            cl_final_status = None

            if args.closed_loop and branch is not None:
                # Closed-loop mode.
                try:
                    cl_result = run_closed_loop(
                        p,
                        max_steps=args.max_steps,
                        prior_prompt_mode=args.prior_prompt,
                        ranking_score=args.ranking_score,
                    )
                    actions_seq = cl_result.actions
                    cl_num_steps = cl_result.num_steps
                    cl_success = cl_result.success
                    cl_final_status = cl_result.final_status
                    # First action becomes the main-row "decision".
                    decision = actions_seq[0] if actions_seq else None
                except Exception as e:
                    status = f"closed_loop_error: {e}"
            else:
                # Single-step mode.
                if branch == Branch.FULLY_VISIBLE:
                    try:
                        decision = handle_fully_visible(p)
                    except Exception as e:
                        status = f"handler_error: {e}"
                elif branch == Branch.PARTIALLY_OCCLUDED:
                    try:
                        decision = handle_partially_visible(p)
                    except Exception as e:
                        status = f"handler_error: {e}"
                elif branch == Branch.FULLY_OCCLUDED:
                    try:
                        decision = handle_fully_occluded(p)
                    except Exception as e:
                        status = f"handler_error: {e}"

            if mid in perception.molmo_to_node:
                label = perception.node_info[perception.molmo_to_node[mid]]["label"]
            else:
                label = f"object_{mid}_not_in_graph"
            row = {
                "model": model_name,
                "prior_prompt": args.prior_prompt,
                "ranking_score": args.ranking_score,
                "target_source": target_entry["target_source"],
                "intent_instruction": target_entry["intent_instruction"],
                "intent_reason": target_entry["intent_reason"],
                "intent_candidate_ids": target_entry["intent_candidate_ids"],
                "intent_vlm_decision": target_entry.get("intent_vlm_decision"),
                "scene_key": scene_key,
                "scene_id": perception.scene_id,
                "target_id": mid,
                "target_label": label,
                "is_query_target": (mid == query_target_id),
                "annotation": perception.annotation,
                "branch": branch_value,
                "grasp_id":    decision.grasp_id    if decision else None,
                "grasp_label": decision.grasp_label if decision else None,
                "is_terminal": decision.is_terminal if decision else None,
                "reason": reason,
                "status": status,
            }
            # Extra fields in closed-loop mode.
            if args.closed_loop:
                row["cl_success"]      = cl_success
                row["cl_num_steps"]    = cl_num_steps
                row["cl_final_status"] = cl_final_status
                row["cl_action_seq"]   = " -> ".join(
                    f"{a.grasp_id}" for a in actions_seq
                ) if actions_seq else ""
                row["cl_reason_seq"]   = " || ".join(
                    str(getattr(a, "message", "") or "") for a in actions_seq
                ) if actions_seq else ""
            row.update(_selected_graspability_fields(decision))
            csv_rows.append(row)
            scene_detail_rows.append(row)
            selected_graspability_summary.append(_selected_summary_row(row))
            reason_blocks.append(_reason_block(row, decision, actions_seq if args.closed_loop else None))
            object_count += 1
            if branch_value:
                branch_counter[branch_value] = branch_counter.get(branch_value, 0) + 1

            # Candidate-level rows for partially_occluded targets.
            if args.closed_loop and actions_seq:
                # Closed-loop: emit one row per (step, candidate).
                for step_idx, action in enumerate(actions_seq, start=1):
                    if action.details is None:
                        continue   # skip fully_visible steps (no scoring)
                    for cand_mid, info in action.details.items():
                        # Labels come from initial perception (mid set shrinks
                        # across steps but labels are stable).
                        cand_node = perception.molmo_to_node[cand_mid]
                        cand_label = perception.node_info[cand_node]["label"]
                        scene_candidate_rows.append({
                            "model": model_name,
                            "prior_prompt": args.prior_prompt,
                            "ranking_score": args.ranking_score,
                            "target_source": target_entry["target_source"],
                            "target_id": mid,
                            "target_label": label,
                            "step": step_idx,
                            "candidate_id": cand_mid,
                            "candidate_label": cand_label,
                            "P_s": info["P_s"],
                            "P_g": info["P_g"],
                            "P":   info["P"],
                            "IG":  info["IG"],
                            "IG_normalized": info.get("IG_normalized"),
                            "graspability": info.get("graspability"),
                            "graspability_part_id": info.get("graspability_part_id"),
                            "graspability_parts": info.get("graspability_parts"),
                            "cost": info.get("cost"),
                            "score_legacy": info.get("score_legacy"),
                            "score_ig": info.get("score_ig"),
                            "score_ig_graspability": info.get("score_ig_graspability"),
                            "score_theory": info.get("score_theory"),
                            "score": info.get("score"),
                            "vlm_reason": info.get("vlm_reason"),
                            "selected": (cand_mid == action.grasp_id),
                        })
            elif (decision is not None
                    and branch in {Branch.PARTIALLY_OCCLUDED, Branch.FULLY_OCCLUDED}
                    and decision.details):
                # Single-step mode (no step column).
                for cand_mid, info in decision.details.items():
                    cand_node = perception.molmo_to_node[cand_mid]
                    cand_label = perception.node_info[cand_node]["label"]
                    scene_candidate_rows.append({
                        "model": model_name,
                        "prior_prompt": args.prior_prompt,
                        "ranking_score": args.ranking_score,
                        "target_source": target_entry["target_source"],
                        "target_id": mid,
                        "target_label": label,
                        "candidate_id": cand_mid,
                        "candidate_label": cand_label,
                        "P_s": info["P_s"],
                        "P_g": info["P_g"],
                        "P":   info["P"],
                        "IG":  info["IG"],
                        "IG_normalized": info.get("IG_normalized"),
                        "graspability": info.get("graspability"),
                        "graspability_part_id": info.get("graspability_part_id"),
                        "graspability_parts": info.get("graspability_parts"),
                        "cost": info.get("cost"),
                        "score_legacy": info.get("score_legacy"),
                        "score_ig": info.get("score_ig"),
                        "score_ig_graspability": info.get("score_ig_graspability"),
                        "score_theory": info.get("score_theory"),
                        "score": info.get("score"),
                        "vlm_reason": info.get("vlm_reason"),
                        "selected": (cand_mid == decision.grasp_id),
                    })

        detail_json[scene_key] = {
            "scene_id": perception.scene_id,
            "annotation": perception.annotation,
            "query_obj_id": query_target_id,
            "num_objects_tested": len(targets),
            "per_object": scene_detail_rows,
        }

        # Write per-scene scene_<id>.csv.
        if scene_candidate_rows:
            scene_id = perception.scene_id
            out_path = details_dir / f"scene_{scene_id}.csv"
            pd.DataFrame(scene_candidate_rows).to_csv(out_path, index=False)
            print(f"  [details] scene_id={scene_id}: "
                  f"{len(scene_candidate_rows)} candidate rows -> {out_path}")

        # Print scene summary to screen.
        print(f"\n=== {scene_key} (scene_id={perception.scene_id}, "
              f"query_obj_id={query_target_id}) ===")
        for row in scene_detail_rows:
            mark = " *" if row["is_query_target"] else "  "
            label_disp = str(row['target_label'])[:30]
            if args.closed_loop:
                seq = row.get("cl_action_seq", "")
                ok = "✓" if row.get("cl_success") else "✗"
                steps = row.get("cl_num_steps", "-")
                info = f" {ok} {steps} steps [{seq}]"
            else:
                info = (f" => grasp_id={row['grasp_id']}"
                        if row['grasp_id'] is not None else "")
            target_disp = str(row["target_id"]) if row["target_id"] is not None else "None"
            print(f" {mark} target_id={target_disp:>4} "
                  f"({label_disp:<30}) -> {row['branch']}{info}")

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    with open(reason_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(reason_blocks))
        if reason_blocks:
            f.write("\n")
    output = {
        "model": model_name,
        "prior_prompt": args.prior_prompt,
        "ranking_score": args.ranking_score,
        "root": str(root),
        "num_scenes": scene_count,
        "num_objects_total": object_count,
        "branch_summary": branch_counter,
        "selected_graspability_summary": selected_graspability_summary,
        "results": detail_json,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "prior_prompt": args.prior_prompt,
                "ranking_score": args.ranking_score,
                "root": str(root),
                "num_scenes": scene_count,
                "num_objects_total": object_count,
                "branch_summary": branch_counter,
                "selected_graspability_summary": selected_graspability_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDone. model={model_name}")
    print(f"  {scene_count} scenes, {object_count} objects total")
    print(f"  branch summary: {branch_counter}")
    print(f"  summary CSV  -> {csv_path}")
    print(f"  summary JSON -> {json_path}")
    print(f"  selected graspability summary -> {summary_path}")
    print(f"  reasons      -> {reason_path}")
    print(f"  details dir  -> {details_dir}/")


if __name__ == "__main__":
    main()
