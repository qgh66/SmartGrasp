"""遍历 sample_data 下所有 gt/summary.json,
对每个场景的每个物体跑分支分类 + handler dispatch。

输出:
    results.csv                       汇总: 每个 (场景, 物体) 一行
    branch_results.json               完整 json
    scene_details/scene_<id>.csv      每个场景一个文件, 记录该场景内
                                      所有 partially_occluded 目标的
                                      候选挡路物详细评分
                                      (P_s/P_g/P/IG/cost/score)

用法:
    python test.py --root sample_data
"""
from dotenv import load_dotenv
load_dotenv()
import json
import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from reason.data_loader import load_sample
from reason.branch_judge.classifier import classify_branch
from reason.schemas import Branch
from reason.fully_visible import handle as handle_fully_visible
from reason.partially_visible import handle as handle_partially_visible
from reason.closed_loop import run_closed_loop


def find_gt_summaries(root: Path):
    """递归找 gt/ 文件夹下的 summary.json。"""
    return [
        (str(p.parent.relative_to(root)), p)
        for p in sorted(root.rglob("summary.json"))
        if p.parent.name == "gt"
    ]


def main():
    parser = argparse.ArgumentParser(description="批量分支分类 + handler dispatch")
    parser.add_argument("--root", default="sample_data", help="场景根目录")
    parser.add_argument("--csv", default="results.csv", help="汇总 csv 路径")
    parser.add_argument("--json", default="branch_results.json", help="汇总 json 路径")
    parser.add_argument("--details-dir", default="scene_details",
                        help="每场景明细 csv 输出目录")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="遮挡边阈值")
    parser.add_argument("--closed-loop", action="store_true",
                        help="开启虚拟闭环模式: 对每个 target 跑完整动作序列")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="闭环最大步数 (防死循环)")
    args = parser.parse_args()

    root = Path(args.root)
    summaries = find_gt_summaries(root)
    if not summaries:
        print(f"[警告] {root} 下没找到 gt/summary.json")
        return

    details_dir = Path(args.details_dir)
    details_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    detail_json = {}
    branch_counter = {}
    scene_count = 0
    object_count = 0
    i = 0

    for scene_key, path in summaries:
        i+=1
        if i == 3:
            break
        try:
            perception = load_sample(path, occlusion_threshold=args.threshold)
        except Exception as e:
            print(f"  [ERROR] {scene_key}: 加载失败 {e}")
            continue

        scene_count += 1
        query_target_id = perception.target_molmo_id
        scene_detail_rows = []
        # 该场景所有 partially_occluded 目标的候选明细 (用于写 scene_<id>.csv)
        scene_candidate_rows = []

        all_mids = sorted(perception.molmo_to_node.keys())

        for mid in all_mids:
            p = replace(perception, target_molmo_id=mid)

            # 1) 分支判断
            try:
                branch, reason = classify_branch(p)
                branch_value = branch.value
                status = "ok"
            except Exception as e:
                branch = None
                branch_value = None
                reason = None
                status = f"classify_error: {e}"

            # 2) 单步 vs 闭环模式
            decision = None
            actions_seq = None         # 闭环动作序列, 只在 closed_loop 模式有值
            cl_num_steps = None
            cl_success = None
            cl_final_status = None

            if args.closed_loop and branch is not None:
                # ====== 闭环模式 ======
                try:
                    cl_result = run_closed_loop(p, max_steps=args.max_steps)
                    actions_seq = cl_result.actions
                    cl_num_steps = cl_result.num_steps
                    cl_success = cl_result.success
                    cl_final_status = cl_result.final_status
                    # 第一步的 decision 作为"首动作", 写入主表
                    decision = actions_seq[0] if actions_seq else None
                except Exception as e:
                    status = f"closed_loop_error: {e}"
            else:
                # ====== 单步模式 (原行为) ======
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

            label = perception.node_info[perception.molmo_to_node[mid]]["label"]
            row = {
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
            # ====== 闭环模式的额外字段 ======
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

            # ====== 收集 partially_occluded 的候选明细 ======
            if args.closed_loop and actions_seq:
                # 闭环模式: 遍历每一步, 每步的所有候选都写一行 (加 step 列)
                for step_idx, action in enumerate(actions_seq, start=1):
                    if action.details is None:
                        continue   # 跳过 fully_visible 步 (无 P_g 等评分)
                    for cand_mid, info in action.details.items():
                        # label 从初始 perception 拿 (各步骤间物体集合减少但 label 不变)
                        cand_node = perception.molmo_to_node[cand_mid]
                        cand_label = perception.node_info[cand_node]["label"]
                        scene_candidate_rows.append({
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
                # 单步模式 (原行为, 不加 step 列)
                for cand_mid, info in decision.details.items():
                    cand_node = perception.molmo_to_node[cand_mid]
                    cand_label = perception.node_info[cand_node]["label"]
                    scene_candidate_rows.append({
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

        # ====== 写该场景的 scene_<id>.csv ======
        if scene_candidate_rows:
            scene_id = perception.scene_id
            out_path = details_dir / f"scene_{scene_id}.csv"
            pd.DataFrame(scene_candidate_rows).to_csv(out_path, index=False)
            print(f"  [details] scene_id={scene_id}: "
                  f"{len(scene_candidate_rows)} candidate rows -> {out_path}")

        # 屏幕打印场景摘要
        print(f"\n=== {scene_key} (scene_id={perception.scene_id}, "
              f"query_obj_id={query_target_id}) ===")
        for row in scene_detail_rows:
            mark = " *" if row["is_query_target"] else "  "
            label_disp = str(row['target_label'])[:30]
            if args.closed_loop:
                seq = row.get("cl_action_seq", "")
                ok = "✓" if row.get("cl_success") else "✗"
                steps = row.get("cl_num_steps", "-")
                info = f" {ok} {steps}步 [{seq}]"
            else:
                info = (f" => grasp_id={row['grasp_id']}"
                        if row['grasp_id'] is not None else "")
            print(f" {mark} target_id={row['target_id']:>3} "
                  f"({label_disp:<30}) -> {row['branch']}{info}")

    pd.DataFrame(csv_rows).to_csv(args.csv, index=False)
    output = {
        "root": str(root),
        "num_scenes": scene_count,
        "num_objects_total": object_count,
        "branch_summary": branch_counter,
        "results": detail_json,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n完成! {scene_count} 个场景, 共测试 {object_count} 个物体")
    print(f"分支统计: {branch_counter}")
    print(f"汇总 CSV  -> {args.csv}")
    print(f"汇总 JSON -> {args.json}")
    print(f"明细目录  -> {details_dir}/")


if __name__ == "__main__":
    main()
