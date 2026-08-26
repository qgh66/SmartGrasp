#!/usr/bin/env python3
"""Generate a balanced 30-scene C1/C2/C3 industrial grasp benchmark."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from generate_complex_occlusion_benchmark import (
    ASSETS,
    CASES as MODERATE_CASES,
    build_case,
)


DEFINITIONS = {
    "C1": {
        "name": "目标可直接抓取",
        "initial_state": "目标清晰可见，无遮挡或遮挡不影响抓取",
        "occlusion_layers": "0",
        "major_assessment": "避免不必要移障，直接完成抓取",
    },
    "C2": {
        "name": "目标轻度遮挡",
        "initial_state": "目标可识别，但直接抓取受遮挡物影响",
        "occlusion_layers": "1-2",
        "major_assessment": "选择少量、合理的障碍物移除动作",
    },
    "C3": {
        "name": "目标重度遮挡或不可识别",
        "initial_state": "目标大部分被遮挡，无法直接确认目标或抓取部位",
        "occlusion_layers": ">=3",
        "major_assessment": "逐步移除障碍物恢复可见性并最终抓取",
    },
}


def _clear_specs() -> list[dict]:
    anchors = [
        [0.305, -0.020], [0.325, 0.010], [0.345, -0.015], [0.310, 0.025],
        [0.335, -0.030], [0.350, 0.020], [0.300, 0.005], [0.340, 0.030],
        [0.320, -0.010], [0.355, -0.025],
    ]
    seeds = [5101, 5102, 5103, 5104, 5105, 5106, 5177, 5108, 5199, 5110]
    specs = []
    for index, (asset_key, anchor, seed) in enumerate(
        zip(ASSETS, anchors, seeds), start=1
    ):
        asset = ASSETS[asset_key]
        specs.append(
            {
                "case_id": f"c1_{index:02d}_{asset_key}_clear",
                "seed": seed,
                "count": 9,
                "target": asset_key,
                "covers": [],
                "anchor": anchor,
                "instruction": f"直接抓取清晰可见的{asset['zh']}",
            }
        )
    return specs


C3_TARGET_COVERS = [
    ("battery", ["power_drill", "large_clamp", "adjustable_wrench"]),
    ("power_drill", ["ycb_hammer", "large_clamp", "adjustable_wrench"]),
    ("adjustable_wrench", ["power_drill", "medium_clamp", "phillips_screwdriver"]),
    ("phillips_screwdriver", ["battery", "large_clamp", "ycb_hammer"]),
    ("flat_screwdriver", ["small_clamp", "power_drill", "two_color_hammer"]),
    ("ycb_hammer", ["power_drill", "large_clamp", "adjustable_wrench"]),
    ("two_color_hammer", ["power_drill", "medium_clamp", "adjustable_wrench"]),
    ("medium_clamp", ["ycb_hammer", "large_clamp", "phillips_screwdriver"]),
    ("large_clamp", ["power_drill", "ycb_hammer", "two_color_hammer"]),
    ("small_clamp", ["battery", "adjustable_wrench", "phillips_screwdriver"]),
]


def _heavy_specs() -> list[dict]:
    anchors = [
        [0.300, -0.055], [0.350, 0.050], [0.315, 0.055], [0.365, -0.050],
        [0.290, 0.045], [0.355, -0.045], [0.305, 0.060], [0.370, 0.025],
        [0.285, -0.040], [0.345, 0.000],
    ]
    spread_offsets = [(0.027, 0.008), (-0.024, -0.010), (0.004, 0.028)]
    compact_offsets = [(0.008, 0.002), (-0.008, -0.002), (0.000, 0.010)]
    large_footprint_targets = {
        "power_drill",
        "ycb_hammer",
        "two_color_hammer",
        "large_clamp",
    }
    axis_aligned_targets = {
        "power_drill",
        "ycb_hammer",
        "large_clamp",
    }
    spread_yaws = [0.55, -0.75, 1.45]
    compact_yaws = [0.0, 1.570796, -1.570796]
    specs = []
    for index, ((target, covers), anchor) in enumerate(
        zip(C3_TARGET_COVERS, anchors), start=1
    ):
        target_label = ASSETS[target]["zh"]
        cover_labels = "、".join(ASSETS[key]["zh"] for key in covers)
        specs.append(
            {
                "case_id": f"c3_{index:02d}_{target}_three_layer",
                "seed": 5300 + index,
                "count": 10,
                "target": target,
                "covers": covers,
                "anchor": anchor,
                # Large targets need the three layers concentrated around the
                # graspable body; small or slender targets retain a wider
                # spread so that they are not made completely invisible.
                "cover_offsets": (
                    compact_offsets
                    if target in large_footprint_targets
                    else spread_offsets
                ),
                "cover_yaw_offsets": (
                    compact_yaws
                    if target in axis_aligned_targets
                    else spread_yaws
                ),
                "instruction": f"逐步移除{cover_labels}并抓取重度遮挡的{target_label}",
            }
        )
    return specs


def _annotate(
    scene: dict,
    entry: dict,
    difficulty: str,
    config_path: str,
    active_index: int,
) -> None:
    layer_count = len(entry["occluder_names"])
    definition = DEFINITIONS[difficulty]
    entry.update(
        {
            "active_index": active_index,
            "config": config_path,
            "difficulty_category": difficulty,
            "difficulty_name": definition["name"],
            "occlusion_layers": layer_count,
            "initial_state": definition["initial_state"],
            "major_assessment": definition["major_assessment"],
            "position_randomization": "deterministic_seeded_jitter",
        }
    )
    scene["description"] = (
        f"Balanced {difficulty} industrial benchmark; seed={entry['seed']}; "
        f"target={entry['target_name']}."
    )
    scene["benchmark"].update(
        {
            "difficulty_category": difficulty,
            "difficulty_name": definition["name"],
            "occlusion_layers": layer_count,
            "initial_state": definition["initial_state"],
            "major_assessment": definition["major_assessment"],
            "position_randomization": "deterministic_seeded_jitter",
        }
    )
    target_name = entry["target_name"]
    for item in scene["objects"]:
        if item["name"] == target_name:
            item["metadata"]["occlusion_case"] = {
                "C1": "directly_graspable",
                "C2": "lightly_occluded_initially",
                "C3": "heavily_occluded_initially",
            }[difficulty]

    if difficulty == "C3":
        previous_name = target_name
        cover_names = entry["occluder_names"]
        for layer, cover_name in enumerate(cover_names, start=1):
            cover = next(item for item in scene["objects"] if item["name"] == cover_name)
            cover["metadata"]["stack_layer"] = layer
            cover["metadata"]["occludes"] = [previous_name]
            cover["metadata"]["required_removal_rank"] = len(cover_names) - layer + 1
            cover["position"][2] = round(
                max(float(cover["position"][2]), 0.065 + 0.025 * (layer - 1)),
                6,
            )
            previous_name = cover_name


def _write_case(
    root: Path,
    difficulty: str,
    spec: dict,
    active_index: int,
) -> dict:
    scene, entry = build_case(spec)
    relative_config = (
        f"graspnet-workspace/config/difficulty_occlusion_benchmark/"
        f"{difficulty}/{spec['case_id']}.json"
    )
    _annotate(scene, entry, difficulty, relative_config, active_index)
    output_path = root / difficulty / f"{spec['case_id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    active_cases: list[dict] = []
    category_specs = {
        "C1": _clear_specs(),
        # All 11 legacy scenes are C2. The first ten form the balanced set;
        # legacy case 11 remains documented as a holdout.
        "C2": [deepcopy(spec) for spec in MODERATE_CASES[:10]],
        "C3": _heavy_specs(),
    }
    for difficulty in ("C1", "C2", "C3"):
        for spec in category_specs[difficulty]:
            active_cases.append(
                _write_case(
                    output_root,
                    difficulty,
                    spec,
                    active_index=len(active_cases),
                )
            )

    legacy_classification = []
    for spec in MODERATE_CASES:
        legacy_classification.append(
            {
                "case_id": spec["case_id"],
                "difficulty_category": "C2",
                "occlusion_layers": len(spec["covers"]),
                "active_in_balanced_30": spec in MODERATE_CASES[:10],
            }
        )

    counts = {
        difficulty: sum(
            case["difficulty_category"] == difficulty for case in active_cases
        )
        for difficulty in DEFINITIONS
    }
    if counts != {"C1": 10, "C2": 10, "C3": 10}:
        raise RuntimeError(f"unbalanced generated benchmark: {counts}")

    manifest = {
        "benchmark": "balanced_difficulty_occlusion_30_single_camera_gpt55",
        "case_count": len(active_cases),
        "category_counts": counts,
        "difficulty_definitions": DEFINITIONS,
        "legacy_scene_classification": legacy_classification,
        "legacy_extra_holdout": MODERATE_CASES[10]["case_id"],
        "layout_profile": "moderate_compact_right_shifted",
        "workspace_shift_x": 0.2,
        "position_randomization": "deterministic_seeded_jitter",
        "camera_mode": "single_top_rgbd",
        "perception_review_model": "gpt-5.5",
        "intent_model": "gpt-5.5",
        "reason_model": "gpt-5.5",
        "reason_prior_prompt": "graspability",
        "reason_ranking_score": "ig_graspability",
        "task_selection_policy": "reason",
        "cases": active_cases,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(active_cases), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
