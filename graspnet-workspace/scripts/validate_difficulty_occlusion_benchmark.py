#!/usr/bin/env python3
"""Validate the balanced C1/C2/C3 benchmark without running simulation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    workspace = Path(args.workspace).resolve()
    repo_root = workspace.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]

    if len(cases) != 30 or manifest.get("case_count") != 30:
        fail(f"expected 30 cases, got {len(cases)}")
    counts = Counter(case["difficulty_category"] for case in cases)
    if counts != Counter({"C1": 10, "C2": 10, "C3": 10}):
        fail(f"unbalanced categories: {dict(counts)}")
    case_ids = [case["case_id"] for case in cases]
    seeds = [case["seed"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        fail("duplicate case_id")
    if len(seeds) != len(set(seeds)):
        fail("duplicate seed")

    minimum_gap = float("inf")
    category_visibility_contracts = {
        "C1": lambda layers: layers == 0,
        "C2": lambda layers: 1 <= layers <= 2,
        "C3": lambda layers: layers >= 3,
    }
    for case in cases:
        case_id = case["case_id"]
        difficulty = case["difficulty_category"]
        config_path = repo_root / case["config"]
        if not config_path.is_file():
            fail(f"{case_id}: missing config {config_path}")
        scene = json.loads(config_path.read_text(encoding="utf-8"))
        objects = scene["objects"]
        benchmark = scene["benchmark"]
        if not 9 <= len(objects) <= 12:
            fail(f"{case_id}: object count {len(objects)} is outside 9-12")
        if len(objects) != case["object_count"] or len(objects) != benchmark["object_count"]:
            fail(f"{case_id}: inconsistent object_count")
        names = [item["name"] for item in objects]
        if len(names) != len(set(names)):
            fail(f"{case_id}: duplicate object names")
        name_to_body_id = {name: index for index, name in enumerate(names, start=1)}
        if name_to_body_id.get(case["target_name"]) != case["target_body_id"]:
            fail(f"{case_id}: target name/body_id mismatch")
        expected_occluder_ids = [name_to_body_id[name] for name in case["occluder_names"]]
        if expected_occluder_ids != case["occluder_body_ids"]:
            fail(f"{case_id}: occluder name/body_id mismatch")
        expected_removal_ids = [name_to_body_id[name] for name in case["removal_order_names"]]
        if expected_removal_ids != case["removal_order_body_ids"]:
            fail(f"{case_id}: removal order name/body_id mismatch")
        if case["removal_order_names"] != list(reversed(case["occluder_names"])):
            fail(f"{case_id}: removal order is not top-to-bottom")
        layers = int(case["occlusion_layers"])
        if layers != len(case["occluder_names"]):
            fail(f"{case_id}: layer count does not match occluders")
        if not category_visibility_contracts[difficulty](layers):
            fail(f"{case_id}: {layers} layers violate {difficulty}")
        if benchmark["difficulty_category"] != difficulty:
            fail(f"{case_id}: manifest/config difficulty mismatch")

        paths = [item["path"] for item in objects]
        if len(paths) != len(set(paths)):
            fail(f"{case_id}: duplicate industrial asset")
        for relative_path in paths:
            if not (workspace / relative_path).is_file():
                fail(f"{case_id}: missing asset {relative_path}")

        for left_index, left in enumerate(objects):
            left_stack = str(left.get("metadata", {}).get("stack_id", ""))
            for right in objects[left_index + 1 :]:
                right_stack = str(right.get("metadata", {}).get("stack_id", ""))
                if left_stack and left_stack == right_stack:
                    continue
                gap = math.dist(left["position"][:2], right["position"][:2])
                minimum_gap = min(minimum_gap, gap)
                if gap < 0.140:
                    fail(f"{case_id}: independent objects only {gap:.4f}m apart")

        if difficulty == "C3":
            previous_name = case["target_name"]
            for layer, cover_name in enumerate(case["occluder_names"], start=1):
                cover = objects[name_to_body_id[cover_name] - 1]
                metadata = cover["metadata"]
                if metadata.get("stack_layer") != layer:
                    fail(f"{case_id}: wrong stack_layer for {cover_name}")
                if metadata.get("occludes") != [previous_name]:
                    fail(f"{case_id}: broken occlusion chain at {cover_name}")
                expected_rank = layers - layer + 1
                if metadata.get("required_removal_rank") != expected_rank:
                    fail(f"{case_id}: wrong removal rank for {cover_name}")
                previous_name = cover_name

    print(
        json.dumps(
            {
                "valid": True,
                "case_count": len(cases),
                "category_counts": dict(sorted(counts.items())),
                "object_count_range": [
                    min(case["object_count"] for case in cases),
                    max(case["object_count"] for case in cases),
                ],
                "minimum_independent_xy_gap_m": round(minimum_gap, 6),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
