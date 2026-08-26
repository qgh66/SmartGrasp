#!/usr/bin/env python3
"""Create paired C3 scenes with clamp occluders swapped out of the target stack."""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_complex_occlusion_benchmark import ASSETS  # noqa: E402


CLAMP_ASSETS = {"small_clamp", "medium_clamp", "large_clamp"}

# Each replacement is already a distractor in its paired original scene.  The
# two physical identities are swapped so the object count and unique-asset set
# remain unchanged while only the target-stack membership changes.
REPLACEMENTS = {
    "c3_01_battery_three_layer": {"large_clamp": "ycb_hammer"},
    "c3_02_power_drill_three_layer": {"large_clamp": "two_color_hammer"},
    "c3_03_adjustable_wrench_three_layer": {"medium_clamp": "ycb_hammer"},
    "c3_04_phillips_screwdriver_three_layer": {"large_clamp": "power_drill"},
    "c3_05_flat_screwdriver_three_layer": {"small_clamp": "ycb_hammer"},
    "c3_06_ycb_hammer_three_layer": {"large_clamp": "two_color_hammer"},
    "c3_07_two_color_hammer_three_layer": {"medium_clamp": "ycb_hammer"},
    "c3_08_medium_clamp_three_layer": {"large_clamp": "adjustable_wrench"},
    "c3_09_large_clamp_three_layer": {},
    "c3_10_small_clamp_three_layer": {},
}


def _asset_from_name(name: str) -> str:
    for asset_key in sorted(ASSETS, key=len, reverse=True):
        if name.endswith(f"_{asset_key}"):
            return asset_key
    raise ValueError(f"Cannot infer asset from object name: {name}")


def _new_case_id(old_case_id: str) -> str:
    match = re.fullmatch(r"c3_(\d{2})_(.+)", old_case_id)
    if not match:
        raise ValueError(f"Unexpected C3 case id: {old_case_id}")
    return f"c3nc_{match.group(1)}_{match.group(2)}"


def _apply_asset(item: dict, asset_key: str) -> None:
    asset = ASSETS[asset_key]
    item["path"] = asset["path"]
    item["scale"] = 1.0
    item["mass"] = asset["mass"]
    item["metadata"]["category"] = asset["category"]


def _rename_with_asset(name: str, old_case_id: str, new_case_id: str, asset_key: str) -> str:
    suffix = name.removeprefix(old_case_id)
    for known_asset in sorted(ASSETS, key=len, reverse=True):
        if suffix.endswith(f"_{known_asset}"):
            suffix = suffix[: -len(known_asset)] + asset_key
            break
    else:
        raise ValueError(f"Cannot replace asset suffix in {name}")
    return new_case_id + suffix


def _swap_cover_with_distractor(
    scene: dict,
    old_case_id: str,
    new_case_id: str,
    clamp_asset: str,
    replacement_asset: str,
    name_map: dict[str, str],
) -> dict:
    cover = next(
        item
        for item in scene["objects"]
        if item.get("metadata", {}).get("benchmark_role") == "target_occluder"
        and _asset_from_name(item["name"]) == clamp_asset
    )
    replacement = next(
        item
        for item in scene["objects"]
        if item.get("metadata", {}).get("benchmark_role") == "distractor"
        and _asset_from_name(item["name"]) == replacement_asset
    )

    cover_old_name = cover["name"]
    replacement_old_name = replacement["name"]
    cover_layer = int(cover["metadata"]["stack_layer"])

    _apply_asset(cover, replacement_asset)
    _apply_asset(replacement, clamp_asset)
    cover["position"][2] = round(
        max(
            float(ASSETS[replacement_asset]["stack_z"]) + 0.006 * (cover_layer - 1),
            0.065 + 0.025 * (cover_layer - 1),
        ),
        6,
    )
    replacement["position"][2] = float(ASSETS[clamp_asset]["base_z"])

    name_map[cover_old_name] = _rename_with_asset(
        cover_old_name, old_case_id, new_case_id, replacement_asset
    )
    name_map[replacement_old_name] = _rename_with_asset(
        replacement_old_name, old_case_id, new_case_id, clamp_asset
    )
    return {
        "removed_cover_asset": clamp_asset,
        "replacement_cover_asset": replacement_asset,
        "cover_body_id": scene["objects"].index(cover) + 1,
        "swapped_distractor_body_id": scene["objects"].index(replacement) + 1,
    }


def build_variant(original_scene: dict, original_entry: dict) -> tuple[dict, dict]:
    scene = deepcopy(original_scene)
    old_case_id = original_entry["case_id"]
    new_case_id = _new_case_id(old_case_id)
    replacement_plan = REPLACEMENTS[old_case_id]

    name_map = {
        item["name"]: item["name"].replace(old_case_id, new_case_id, 1)
        for item in scene["objects"]
    }
    swap_records = []
    for clamp_asset, replacement_asset in replacement_plan.items():
        swap_records.append(
            _swap_cover_with_distractor(
                scene,
                old_case_id,
                new_case_id,
                clamp_asset,
                replacement_asset,
                name_map,
            )
        )

    for item in scene["objects"]:
        old_name = item["name"]
        item["name"] = name_map[old_name]
        metadata = item.get("metadata", {})
        if "occludes" in metadata:
            metadata["occludes"] = [name_map.get(name, name) for name in metadata["occludes"]]

    benchmark = scene["benchmark"]
    benchmark["case_id"] = new_case_id
    benchmark["target_name"] = name_map[benchmark["target_name"]]
    benchmark["target_occluder_names"] = [
        name_map[name] for name in benchmark["target_occluder_names"]
    ]
    benchmark["removal_order_names"] = [
        name_map[name] for name in benchmark["removal_order_names"]
    ]

    cover_assets = [
        _asset_from_name(name) for name in benchmark["target_occluder_names"]
    ]
    if any(asset in CLAMP_ASSETS for asset in cover_assets):
        raise RuntimeError(f"Clamp occluder remains in {new_case_id}: {cover_assets}")

    target_label = original_entry["target_label"]
    cover_labels = "、".join(ASSETS[key]["zh"] for key in cover_assets)
    instruction = f"逐步移除{cover_labels}并抓取重度遮挡的{target_label}"
    benchmark["instruction"] = instruction
    benchmark["paired_original_case_id"] = old_case_id
    benchmark["variant"] = "no_clamp_target_occluders"
    benchmark["cover_asset_swaps"] = swap_records
    scene["description"] = (
        f"Paired C3 no-clamp-occluder variant of {old_case_id}; "
        f"target={benchmark['target_name']}."
    )

    new_entry = deepcopy(original_entry)
    new_entry.update(
        {
            "case_id": new_case_id,
            "paired_original_case_id": old_case_id,
            "variant": "no_clamp_target_occluders",
            "instruction": instruction,
            "target_name": benchmark["target_name"],
            "occluder_names": benchmark["target_occluder_names"],
            "removal_order_names": benchmark["removal_order_names"],
            "occluder_labels": [ASSETS[key]["zh"] for key in cover_assets],
            "cover_asset_swaps": swap_records,
            "scene_no": f"HNC-{old_case_id[3:5]}",
        }
    )
    return scene, new_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--config-prefix",
        default="graspnet-workspace/config/difficulty_occlusion_benchmark/C3_no_clamp_occluders",
    )
    args = parser.parse_args()

    original_manifest_path = Path(args.original_manifest).resolve()
    repo_root = original_manifest_path.parents[2]
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    original_cases = [
        item for item in original_manifest["cases"] if item["difficulty_category"] == "C3"
    ]
    if [item["case_id"] for item in original_cases] != list(REPLACEMENTS):
        raise RuntimeError("Original C3 manifest order does not match the paired replacement plan")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    new_cases = []
    for original_entry in original_cases:
        original_config_path = repo_root / original_entry["config"]
        original_scene = json.loads(original_config_path.read_text(encoding="utf-8"))
        scene, entry = build_variant(original_scene, original_entry)
        config_path = output_dir / f"{entry['case_id']}.json"
        config_path.write_text(
            json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        entry["config"] = f"{args.config_prefix.rstrip('/')}/{entry['case_id']}.json"
        new_cases.append(entry)

    manifest = {
        "benchmark": "hard10_no_clamp_target_occluders_paired",
        "case_count": len(new_cases),
        "category_counts": {"C3": len(new_cases)},
        "paired_original_benchmark": original_manifest["benchmark"],
        "variant": "no_clamp_target_occluders",
        "comparison_control": (
            "Target, camera, XY positions, orientations, object count, object ordering, "
            "and stack structure are preserved. Clamp cover assets are exchanged with "
            "non-clamp distractor assets already present in each scene."
        ),
        "cases": new_cases,
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "generated_cases": len(new_cases),
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "replacement_plan": REPLACEMENTS,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
