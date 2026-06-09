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

# Parse --model early so we can set VLM_MODEL before importing reason.*
# (the VLM client is a module-level singleton; env vars must be set first)
import argparse
import os

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--model", default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.model:
    os.environ["VLM_MODEL"] = _pre_args.model

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from reason.data_loader import load_sample
from reason.branch_judge.classifier import classify_branch
from reason.schemas import Branch
from reason.fully_visible import handle as handle_fully_visible
from reason.partially_visible import handle as handle_partially_visible
from reason.closed_loop import run_closed_loop


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
    parser.add_argument("--model", default=None,
                        help="VLM model name (overrides VLM_MODEL from .env)")
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
    parser.add_argument("--closed-loop", action="store_true",
                        help="Closed-loop mode: full action sequence per target")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max steps in closed-loop (safety cap)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N scenes (debug)")
    args = parser.parse_args()

    # Resolve current model from env (may come from .env or --model).
    model_name = os.environ.get("VLM_MODEL", "unknown_model")
    model_safe = _sanitize_model_name(model_name)
    print(f"[CONFIG] using VLM_MODEL = {model_name}")

    # Per-model output dirs.
    out_dir = Path(args.out_root) / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_dir / "results.csv"
    json_path = Path(args.json) if args.json else out_dir / "branch_results.json"
    details_dir = Path(args.details_dir) if args.details_dir else out_dir / "scene_details"
    details_dir.mkdir(parents=True, exist_ok=True)

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
    detail_json = {}
    branch_counter = {}
    scene_count = 0
    object_count = 0

    for i, (scene_key, path) in enumerate(summaries):
        if args.limit and i >= args.limit:
            print(f"[LIMIT] reached {args.limit} scenes, stopping")
            break

        try:
            perception = load_sample(path, occlusion_threshold=args.threshold)
        except Exception as e:
            print(f"  [ERROR] {scene_key}: load failed: {e}")
            continue

        scene_count += 1
        query_target_id = perception.target_molmo_id
        scene_detail_rows = []
        # Candidate-level rows for this scene's partially_occluded targets.
        scene_candidate_rows = []

        if args.target_id is not None:
            all_mids = [args.target_id]
        else:
            all_mids = sorted(perception.molmo_to_node.keys())

        for mid in all_mids:
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
                    cl_result = run_closed_loop(p, max_steps=args.max_steps)
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

            if mid in perception.molmo_to_node:
                label = perception.node_info[perception.molmo_to_node[mid]]["label"]
            else:
                label = f"object_{mid}_not_in_graph"
            row = {
                "model": model_name,
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
            csv_rows.append(row)
            scene_detail_rows.append(row)
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
                            "target_id": mid,
                            "target_label": label,
                            "step": step_idx,
                            "candidate_id": cand_mid,
                            "candidate_label": cand_label,
                            "P_s": info["P_s"],
                            "P_g": info["P_g"],
                            "P":   info["P"],
                            "IG":  info["IG"],
                            "cost": info.get("cost"),
                            "score": info.get("score"),
                            "selected": (cand_mid == action.grasp_id),
                        })
            elif (decision is not None
                    and branch == Branch.PARTIALLY_OCCLUDED
                    and decision.details):
                # Single-step mode (no step column).
                for cand_mid, info in decision.details.items():
                    cand_node = perception.molmo_to_node[cand_mid]
                    cand_label = perception.node_info[cand_node]["label"]
                    scene_candidate_rows.append({
                        "model": model_name,
                        "target_id": mid,
                        "target_label": label,
                        "candidate_id": cand_mid,
                        "candidate_label": cand_label,
                        "P_s": info["P_s"],
                        "P_g": info["P_g"],
                        "P":   info["P"],
                        "IG":  info["IG"],
                        "cost": info.get("cost"),
                        "score": info.get("score"),
                        "selected": (cand_mid == decision.grasp_id),
                    })

        detail_json[scene_key] = {
            "scene_id": perception.scene_id,
            "annotation": perception.annotation,
            "query_obj_id": query_target_id,
            "num_objects_tested": len(all_mids),
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
            print(f" {mark} target_id={row['target_id']:>3} "
                  f"({label_disp:<30}) -> {row['branch']}{info}")

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    output = {
        "model": model_name,
        "root": str(root),
        "num_scenes": scene_count,
        "num_objects_total": object_count,
        "branch_summary": branch_counter,
        "results": detail_json,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone. model={model_name}")
    print(f"  {scene_count} scenes, {object_count} objects total")
    print(f"  branch summary: {branch_counter}")
    print(f"  summary CSV  -> {csv_path}")
    print(f"  summary JSON -> {json_path}")
    print(f"  details dir  -> {details_dir}/")


if __name__ == "__main__":
    main()
