#!/usr/bin/env python
"""Summarize one SmartGrasp FreeGrasp-style benchmark run as Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


CONDITION_DETAILS = {
    "low_without_ambiguity": (
        "低复杂度 / w/o ambiguity",
        "平铺无遮挡，目标类别在场景中唯一",
    ),
    "low_with_ambiguity": (
        "低复杂度 / with ambiguity",
        "平铺无遮挡，同时存在小/中/大三个夹具",
    ),
    "medium_without_ambiguity": (
        "中复杂度 / w/o ambiguity",
        "12 物体堆叠场景，目标是可见上层唯一类别",
    ),
    "medium_with_ambiguity": (
        "中复杂度 / with ambiguity",
        "12 物体堆叠场景，上下层存在同类实例",
    ),
    "high_without_ambiguity": (
        "高复杂度 / w/o ambiguity",
        "先删除上层双色锤，再重新观察唯一的十字螺丝刀",
    ),
    "high_with_ambiguity": (
        "高复杂度 / with ambiguity",
        "先删除电钻，再在另一块电池的干扰下定位被揭露电池",
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--failure-log", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mask-min-iou", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as manifest_file:
        return list(csv.DictReader(manifest_file, delimiter="\t"))


def read_failures(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_result(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "结果文件缺失"
    try:
        with path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)
    except (OSError, json.JSONDecodeError) as error:
        return None, f"无法读取 JSON: {error}"
    if result.get("mode") != "perception_reason_validation_and_delete":
        return result, f"非预期模式: {result.get('mode')}"
    if result.get("graspnet_enabled") is not False:
        return result, "graspnet_enabled 不是 false"
    if result.get("physical_actions_enabled") is not False:
        return result, "physical_actions_enabled 不是 false"
    return result, None


def format_ratio(success: int, total: int) -> str:
    if total <= 0:
        return "N/A"
    return f"{success}/{total} ({success / total:.1%})"


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def instructions_by_target(result: dict[str, Any]) -> dict[str, list[str]]:
    target_instructions: dict[str, list[str]] = defaultdict(list)
    for attempt in result.get("attempts", []):
        target_name = str(attempt.get("target_object_name") or "unknown")
        instruction = str(attempt.get("instruction") or "").strip()
        if instruction and instruction not in target_instructions[target_name]:
            target_instructions[target_name].append(instruction)
    return dict(target_instructions)


def main() -> None:
    arguments = parse_arguments()
    manifest_rows = read_manifest(arguments.manifest)
    failure_lines = read_failures(arguments.failure_log)

    condition_summaries: dict[str, dict[str, int]] = {
        condition_key: defaultdict(int)
        for condition_key in CONDITION_DETAILS
    }
    detail_rows: list[list[str]] = []

    for manifest_row in manifest_rows:
        condition_key = manifest_row["condition"]
        summary = condition_summaries.setdefault(condition_key, defaultdict(int))
        planned_targets = [
            target
            for target in manifest_row["target_names"].split(",")
            if target
        ]
        summary["episodes_planned"] += 1
        summary["targets_planned"] += len(planned_targets)

        result_path = Path(manifest_row["result_path"])
        result, validation_error = load_result(result_path)
        if result is None:
            detail_rows.append([
                CONDITION_DETAILS.get(condition_key, (condition_key, ""))[0],
                manifest_row["episode"],
                manifest_row["repeat"],
                manifest_row["prompt_template"],
                "-",
                "0/" + str(len(planned_targets)),
                "0/" + str(len(planned_targets)),
                "0/" + str(len(planned_targets)),
                validation_error or "失败",
                str(result_path),
            ])
            continue

        summary["episodes_completed"] += 1
        objects = result.get("objects", [])
        if not isinstance(objects, list):
            objects = []

        perception_success = sum(
            bool(item.get("perception_correct")) for item in objects
        )
        reason_success = sum(bool(item.get("reason_correct")) for item in objects)
        removed_success = sum(bool(item.get("removed_from_scene")) for item in objects)
        summary["perception_success"] += perception_success
        summary["reason_success"] += reason_success
        summary["removed_success"] += removed_success
        if len(objects) == len(planned_targets) and removed_success == len(planned_targets):
            summary["episodes_passed"] += 1

        instruction_map = instructions_by_target(result)
        expanded_instructions = []
        for target_name in planned_targets:
            target_prompts = instruction_map.get(target_name) or []
            expanded_instructions.append(
                f"{target_name}: " + (" / ".join(target_prompts) if target_prompts else "-")
            )

        failed_object_details = [
            f"{item.get('target_object_name', 'unknown')}: "
            f"{item.get('failure_reason') or '核验未通过'}"
            for item in objects
            if not item.get("removed_from_scene")
        ]
        if validation_error:
            status = validation_error
        elif len(objects) == len(planned_targets) and not failed_object_details:
            status = "通过"
        else:
            failure_summary = "; ".join(failed_object_details) or "结果数量不完整"
            status = f"未全部通过: {failure_summary}"
        detail_rows.append([
            CONDITION_DETAILS.get(condition_key, (condition_key, ""))[0],
            manifest_row["episode"],
            manifest_row["repeat"],
            manifest_row["prompt_template"],
            "<br>".join(expanded_instructions),
            f"{perception_success}/{len(planned_targets)}",
            f"{reason_success}/{len(planned_targets)}",
            f"{removed_success}/{len(planned_targets)}",
            status,
            str(result_path),
        ])

    report_lines = [
        "# SmartGrasp FreeGrasp-style 六类实验结果",
        "",
        f"- Run ID: `{markdown_escape(arguments.run_id)}`",
        f"- 生成时间: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- 结果清单: `{markdown_escape(arguments.manifest)}`",
        f"- 目标 mask 最小 IoU: `{arguments.mask_min_iou:g}`",
        "- 执行模式: `perception_reason_validation_and_delete`",
        "- GraspNet: 禁用",
        "- 机械臂/Push: 禁用",
        "- PyBullet GUI/录像: 禁用",
        "",
        "## 六类结果汇总",
        "",
        "| 条件 | 场景 | 完成 episode | episode 全通过 | SSR | RSR | 核验后删除 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for condition_key, (condition_label, scene_description) in CONDITION_DETAILS.items():
        summary = condition_summaries[condition_key]
        episodes_planned = summary["episodes_planned"]
        targets_planned = summary["targets_planned"]
        report_lines.append(
            "| "
            + " | ".join([
                markdown_escape(condition_label),
                markdown_escape(scene_description),
                f"{summary['episodes_completed']}/{episodes_planned}",
                format_ratio(summary["episodes_passed"], episodes_planned),
                format_ratio(summary["perception_success"], targets_planned),
                format_ratio(summary["reason_success"], targets_planned),
                format_ratio(summary["removed_success"], targets_planned),
            ])
            + " |"
        )

    report_lines.extend([
        "",
        "SSR 按 `objects[*].perception_correct` 统计，RSR 按 "
        "`objects[*].reason_correct` 统计。本次二者都使用上方记录的 "
        "mask IoU 阈值。“核验后删除”要求 Perception 和 Reason 同时正确。",
        "",
        "## 逐 episode 结果",
        "",
        "| 条件 | Episode | 提示词编号 | 提示词模板 | 实际展开指令 | Perception | Reason | 删除 | 状态 | JSON |",
        "|---|---|---:|---|---|---:|---:|---:|---|---|",
    ])
    for detail_row in detail_rows:
        report_lines.append(
            "| " + " | ".join(markdown_escape(value) for value in detail_row) + " |"
        )

    report_lines.extend([
        "",
        "## 作业失败记录",
        "",
    ])
    if failure_lines:
        report_lines.extend(f"- `{markdown_escape(line)}`" for line in failure_lines)
    else:
        report_lines.append("无。")
    report_lines.append("")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Markdown report written to: {arguments.output}")


if __name__ == "__main__":
    main()
