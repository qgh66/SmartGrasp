#!/usr/bin/env python3
"""Generate deterministic, moderately occluded industrial-tool scenes."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


ASSETS = {
    "battery": {
        "path": "assets/objects/industrial_tools/battery/Scan.obj",
        "category": "battery",
        "mass": 0.05,
        "base_z": 0.025,
        "stack_z": 0.050,
        "zh": "电池",
        "aliases": ["电池", "battery"],
    },
    "power_drill": {
        "path": "assets/objects/industrial_tools/ycb/035_power_drill/google_16k/textured.obj",
        "category": "power_drill",
        "mass": 0.32,
        "base_z": 0.035,
        "stack_z": 0.067,
        "zh": "电钻",
        "aliases": ["电钻", "充电钻", "power drill"],
    },
    "adjustable_wrench": {
        "path": "assets/objects/industrial_tools/ycb/042_adjustable_wrench/google_16k/textured.obj",
        "category": "wrench",
        "mass": 0.16,
        "base_z": 0.030,
        "stack_z": 0.063,
        "zh": "活动扳手",
        "aliases": ["活动扳手", "扳手", "adjustable wrench"],
    },
    "phillips_screwdriver": {
        "path": "assets/objects/industrial_tools/ycb/043_phillips_screwdriver/google_16k/textured.obj",
        "category": "screwdriver",
        "mass": 0.08,
        "base_z": 0.025,
        "stack_z": 0.070,
        "zh": "十字螺丝刀",
        "aliases": ["十字螺丝刀", "十字起子", "phillips screwdriver"],
    },
    "flat_screwdriver": {
        "path": "assets/objects/industrial_tools/ycb/044_flat_screwdriver/google_16k/textured.obj",
        "category": "screwdriver",
        "mass": 0.08,
        "base_z": 0.025,
        "stack_z": 0.070,
        "zh": "一字螺丝刀",
        "aliases": ["一字螺丝刀", "平口螺丝刀", "flat screwdriver"],
    },
    "ycb_hammer": {
        "path": "assets/objects/industrial_tools/ycb/048_hammer/google_16k/textured.obj",
        "category": "hammer",
        "mass": 0.24,
        "base_z": 0.035,
        "stack_z": 0.074,
        "zh": "灰色锤子",
        "aliases": ["灰色锤子", "锤子", "hammer"],
    },
    "two_color_hammer": {
        "path": "assets/objects/industrial_tools/two_color_hammer/textured.obj",
        "category": "hammer",
        "mass": 0.24,
        "base_z": 0.035,
        "stack_z": 0.074,
        "zh": "双色锤子",
        "aliases": ["双色锤子", "红黑锤子", "two color hammer"],
    },
    "medium_clamp": {
        "path": "assets/objects/industrial_tools/ycb/050_medium_clamp/google_16k/textured.obj",
        "category": "clamp",
        "mass": 0.10,
        "base_z": 0.010,
        "stack_z": 0.042,
        "zh": "中号夹具",
        "aliases": ["中号夹具", "中号夹钳", "medium clamp"],
    },
    "large_clamp": {
        "path": "assets/objects/industrial_tools/ycb/051_large_clamp/google_16k/textured.obj",
        "category": "clamp",
        "mass": 0.16,
        "base_z": 0.030,
        "stack_z": 0.067,
        "zh": "大号夹具",
        "aliases": ["大号夹具", "大号夹钳", "large clamp"],
    },
    "small_clamp": {
        "path": "assets/objects/industrial_tools/small_clamp/textured.obj",
        "category": "clamp",
        "mass": 0.08,
        "base_z": 0.012,
        "stack_z": 0.040,
        "zh": "小号夹具",
        "aliases": ["小号夹具", "小号夹钳", "small clamp"],
    },
}


# Keep the existing camera framing, object spacing, and occlusion geometry, but
# move the complete benchmark workspace away from the robot on the left side of
# the top-camera image.  For this top-down camera, image-right is world +X.
WORKSPACE_SHIFT_X = 0.200


CASES = [
    {
        "case_id": "case_01_flat_screwdriver_partial_wrench",
        "seed": 2401,
        "count": 9,
        "target": "flat_screwdriver",
        "covers": ["adjustable_wrench"],
        "anchor": [0.320, 0.000],
        "instruction": "抓取被活动扳手部分遮挡的一字螺丝刀",
    },
    {
        "case_id": "case_02_battery_partial_drill",
        "seed": 2402,
        "count": 9,
        "target": "battery",
        "covers": ["power_drill"],
        "anchor": [0.240, -0.035],
        "distractor_y_insets": {"ycb_hammer": 0.035},
        "instruction": "抓取被电钻部分遮挡的电池",
    },
    {
        "case_id": "case_03_phillips_partial_hammer",
        "seed": 2403,
        "count": 9,
        "target": "phillips_screwdriver",
        "covers": ["two_color_hammer"],
        "anchor": [0.400, 0.035],
        "instruction": "抓取被双色锤子部分遮挡的十字螺丝刀",
    },
    {
        "case_id": "case_04_wrench_partial_clamp",
        "seed": 2404,
        "count": 9,
        "target": "adjustable_wrench",
        "covers": ["large_clamp"],
        "anchor": [0.320, -0.015],
        "instruction": "抓取被大号夹具部分遮挡的活动扳手",
    },
    {
        "case_id": "case_05_medium_clamp_two_cover",
        "seed": 2405,
        "count": 10,
        "target": "medium_clamp",
        "covers": ["battery", "flat_screwdriver"],
        "anchor": [0.330, 0.040],
        "instruction": "抓取被电池和一字螺丝刀轻度交叉遮挡的中号夹具",
    },
    {
        "case_id": "case_06_small_clamp_partial_medium_clamp",
        "seed": 2406,
        "count": 9,
        "target": "small_clamp",
        "covers": ["medium_clamp"],
        "anchor": [0.390, -0.035],
        "cover_offsets": [(0.018, 0.006)],
        "instruction": "抓取被中号夹具部分遮挡的小号夹具",
    },
    {
        "case_id": "case_07_large_clamp_two_cover",
        "seed": 2407,
        "count": 10,
        "target": "large_clamp",
        "covers": ["flat_screwdriver", "battery"],
        "anchor": [0.250, 0.025],
        "instruction": "抓取被一字螺丝刀和电池轻度交叉遮挡的大号夹具",
    },
    {
        "case_id": "case_08_drill_partial_wrench",
        "seed": 2408,
        "count": 9,
        "target": "power_drill",
        "covers": ["large_clamp"],
        "anchor": [0.360, 0.020],
        "cover_offsets": [(0.000, 0.000)],
        "cover_yaw_offsets": [1.570796],
        "instruction": "抓取被大号夹具部分遮挡的电钻",
    },
    {
        "case_id": "case_09_hammer_partial_clamp",
        "seed": 2409,
        "count": 9,
        "target": "two_color_hammer",
        "covers": ["medium_clamp"],
        "anchor": [0.290, -0.030],
        "cover_offsets": [(0.015, 0.000)],
        "instruction": "抓取被中号夹具部分遮挡的双色锤子",
    },
    {
        "case_id": "case_10_flat_screwdriver_two_cover",
        "seed": 2410,
        "count": 10,
        "target": "flat_screwdriver",
        "covers": ["ycb_hammer", "medium_clamp"],
        "anchor": [0.400, 0.000],
        "cover_offsets": [(0.035, 0.012), (-0.035, -0.015)],
        "instruction": "抓取被灰色锤子和中号夹具轻度遮挡的一字螺丝刀",
    },
    {
        "case_id": "case_11_battery_two_cover",
        "seed": 2411,
        "count": 10,
        "target": "battery",
        "covers": ["adjustable_wrench", "medium_clamp"],
        "anchor": [0.320, 0.040],
        "instruction": "抓取被活动扳手和中号夹具轻度遮挡的电池",
    },
]


DISTRACTOR_CENTERS = [
    (0.140, -0.170),
    (0.320, -0.170),
    (0.500, -0.170),
    (0.680, -0.170),
    (0.140, 0.170),
    (0.320, 0.170),
    (0.500, 0.170),
    (0.680, 0.170),
    (0.140, 0.000),
    (0.680, 0.000),
]


def _object(
    asset_key: str,
    name: str,
    position: list[float],
    yaw: float,
    metadata: dict,
) -> dict:
    asset = ASSETS[asset_key]
    return {
        "name": name,
        "path": asset["path"],
        "position": [round(float(v), 6) for v in position],
        "euler": [0.0, 0.0, round(float(yaw), 6)],
        "scale": 1.0,
        "mass": asset["mass"],
        "lateral_friction": 1.4,
        "spinning_friction": 0.15,
        "metadata": metadata,
    }


def _far_centers(
    occupied_centers: list[tuple[float, float]], rng: random.Random
):
    centers = [
        (center_x + WORKSPACE_SHIFT_X, center_y)
        for center_x, center_y in DISTRACTOR_CENTERS
    ]
    rng.shuffle(centers)
    return [
        center
        for center in centers
        if all(
            math.dist(center, occupied_center) >= 0.145
            for occupied_center in occupied_centers
        )
    ]


def build_case(spec: dict) -> tuple[dict, dict]:
    rng = random.Random(spec["seed"])
    target_key = spec["target"]
    target_asset = ASSETS[target_key]
    target_name = f"{spec['case_id']}_target_{target_key}"
    anchor = (
        spec["anchor"][0]
        + WORKSPACE_SHIFT_X
        + rng.uniform(-0.006, 0.006),
        spec["anchor"][1] + rng.uniform(-0.006, 0.006),
    )
    target_yaw = rng.uniform(-math.pi, math.pi)
    objects = [
        _object(
            target_key,
            target_name,
            [anchor[0], anchor[1], target_asset["base_z"]],
            target_yaw,
            {
                "category": target_asset["category"],
                "instruction_aliases": target_asset["aliases"],
                "role": "target",
                "stack_id": "target_stack",
                "stack_layer": 0,
                "occlusion_case": "partially_occluded_initially",
            },
        )
    ]

    cover_names = []
    # The two covers overlap the target from opposite sides.  Their wider
    # offsets keep a useful part of the target visible instead of creating a
    # vertically stacked three-object pile.
    cover_offsets = spec.get(
        "cover_offsets", [(0.046, 0.018), (-0.046, -0.020)]
    )
    for layer, cover_key in enumerate(spec["covers"], start=1):
        asset = ASSETS[cover_key]
        dx, dy = cover_offsets[layer - 1]
        dx += rng.uniform(-0.004, 0.004)
        dy += rng.uniform(-0.004, 0.004)
        name = f"{spec['case_id']}_cover_{layer}_{cover_key}"
        cover_names.append(name)
        covered_name = target_name
        objects.append(
            _object(
                cover_key,
                name,
                [
                    anchor[0] + dx,
                    anchor[1] + dy,
                    asset["stack_z"] + 0.006 * (layer - 1),
                ],
                (
                    target_yaw
                    + float(spec["cover_yaw_offsets"][layer - 1])
                    + rng.uniform(-0.08, 0.08)
                    if spec.get("cover_yaw_offsets")
                    else rng.uniform(-math.pi, math.pi)
                ),
                {
                    "category": asset["category"],
                    "stack_id": "target_stack",
                    "stack_layer": 1,
                    "occludes": [covered_name],
                    "benchmark_role": "target_occluder",
                },
            )
        )

    # Do not place an identical target instance elsewhere in the scene.  Other
    # members of the same broad category are still allowed, but each asset is
    # used at most once and is spatially separated.
    reserved_assets = {target_key, *spec["covers"]}
    distractor_keys = [key for key in ASSETS if key not in reserved_assets]
    rng.shuffle(distractor_keys)
    centers = _far_centers(
        [tuple(item["position"][:2]) for item in objects], rng
    )
    distractor_count = spec["count"] - len(objects)
    if len(centers) < distractor_count:
        raise ValueError(
            f"{spec['case_id']} has only {len(centers)} separated centers for "
            f"{distractor_count} distractors"
        )
    distractor_index = 0
    while len(objects) < spec["count"]:
        asset_key = distractor_keys[distractor_index]
        asset = ASSETS[asset_key]
        center = centers[distractor_index]
        base_name = f"{spec['case_id']}_distractor_{distractor_index + 1}_{asset_key}"
        base_position = [
            center[0] + rng.uniform(-0.006, 0.006),
            center[1] + rng.uniform(-0.006, 0.006),
            asset["base_z"],
        ]
        distractor_y_inset = float(
            spec.get("distractor_y_insets", {}).get(asset_key, 0.0)
        )
        if distractor_y_inset > 0.0 and abs(base_position[1]) > 0.10:
            base_position[1] -= math.copysign(
                distractor_y_inset, base_position[1]
            )
        objects.append(
            _object(
                asset_key,
                base_name,
                base_position,
                rng.uniform(-math.pi, math.pi),
                {
                    "category": asset["category"],
                    "stack_id": f"distractor_{distractor_index + 1}",
                    "stack_layer": 0,
                    "benchmark_role": "distractor",
                },
            )
        )
        distractor_index += 1

    rng.shuffle(objects)
    target_body_id = next(
        index + 1
        for index, item in enumerate(objects)
        if item["name"] == target_name
    )
    occluder_body_ids = [
        next(
            index + 1
            for index, item in enumerate(objects)
            if item["name"] == name
        )
        for name in cover_names
    ]
    removal_order_names = list(reversed(cover_names))
    removal_order_body_ids = [
        next(
            index + 1
            for index, item in enumerate(objects)
            if item["name"] == name
        )
        for name in removal_order_names
    ]
    scene = {
        "description": (
            f"Deterministic moderate industrial occlusion benchmark; "
            f"seed={spec['seed']}; target={target_name}."
        ),
        "benchmark": {
            "case_id": spec["case_id"],
            "seed": spec["seed"],
            "target_name": target_name,
            "target_body_id": target_body_id,
            "target_asset": target_key,
            "target_occluder_names": cover_names,
            "target_occluder_body_ids": occluder_body_ids,
            "removal_order_names": removal_order_names,
            "removal_order_body_ids": removal_order_body_ids,
            "instruction": spec["instruction"],
            "object_count": len(objects),
            "layout_profile": "moderate_compact_right_shifted",
            "workspace_shift_x": WORKSPACE_SHIFT_X,
        },
        "settle_steps": 720,
        "robot_base_yaw_deg": 180.0,
        "capture_joint_pose_deg": [0.0, 90.0, 45.0, 135.0, 270.0, 72.0],
        "place_target_joint_pose_deg": [-75.0, 90.0, 45.0, 135.0, 270.0, 72.0],
        "object_staging": {"lock_initial_poses_until_grasp": True},
        "continuous_grasp": {"drop_settle_steps": 240, "max_stalled_passes": 2},
        "camera": {
            "position": [0.32 + WORKSPACE_SHIFT_X, 0.0, 0.72],
            "target": [0.32 + WORKSPACE_SHIFT_X, 0.0, 0.07],
            "width": 1280,
            "height": 720,
            "fov": 60.0,
            "near": 0.01,
            "far": 5.0,
        },
        "crop": {"margin": 0.06, "num_points": 20000, "table_z": 0.005},
        "grasp_filter": {
            "max_center_dist": 0.04,
            "bbox_margin": 0.04,
            "min_inner_points": 5,
        },
        "collision_filter": {
            "enabled": True,
            "voxel_size": 0.005,
            "approach_dist": 0.05,
            "collision_thresh": 0.05,
        },
        "topdown_filter": {"enabled": True, "max_angle_deg": 60.0},
        "objects": objects,
    }
    manifest_entry = {
        "case_id": spec["case_id"],
        "seed": spec["seed"],
        "config": f"graspnet-workspace/config/complex_occlusion_benchmark/{spec['case_id']}.json",
        "instruction": spec["instruction"],
        "object_count": len(objects),
        "target_name": target_name,
        "target_body_id": target_body_id,
        "target_asset": target_key,
        "target_label": target_asset["zh"],
        "occluder_names": cover_names,
        "occluder_body_ids": occluder_body_ids,
        "removal_order_names": removal_order_names,
        "removal_order_body_ids": removal_order_body_ids,
        "occluder_labels": [ASSETS[key]["zh"] for key in spec["covers"]],
        "layout_profile": "moderate_compact_right_shifted",
        "workspace_shift_x": WORKSPACE_SHIFT_X,
    }
    return scene, manifest_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--existing-config",
        default=None,
        help="Deprecated compatibility argument; moderate scenes are all regenerated.",
    )
    parser.add_argument(
        "--config-prefix",
        default="graspnet-workspace/config/moderate_occlusion_benchmark",
        help="Repository-relative config directory stored in the manifest.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in CASES:
        scene, entry = build_case(spec)
        entry["config"] = (
            f"{args.config_prefix.rstrip('/')}/{spec['case_id']}.json"
        )
        config_path = output_dir / f"{spec['case_id']}.json"
        config_path.write_text(
            json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(entry)

    Path(args.manifest).write_text(
        json.dumps(
            {
                "benchmark": "moderate_occlusion_single_camera_gpt55_ig_graspability",
                "layout_profile": "moderate_compact_right_shifted",
                "workspace_shift_x": WORKSPACE_SHIFT_X,
                "camera_mode": "single_top_rgbd",
                "perception_review_model": "gpt-5.5",
                "intent_model": "gpt-5.5",
                "reason_model": "gpt-5.5",
                "reason_prior_prompt": "graspability",
                "reason_ranking_score": "ig_graspability",
                "task_selection_policy": "reason",
                "cases": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(CASES)} scenes; manifest has {len(manifest)} cases")


if __name__ == "__main__":
    main()
