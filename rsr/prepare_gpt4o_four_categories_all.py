from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    ALGORITHMS,
    DEFAULT_DATA_ROOT,
    DEFAULT_INPUT_ROOT,
    PROJECT_ROOT,
    TEST_CASE_BY_DIRECTORY,
)
from .prepare_inputs import (
    build_npz_index,
    export_scene,
    load_selected_images,
    read_metadata,
)


SELECTED_CATEGORIES = (
    "01_hard_ambiguous",
    "02_medium_ambiguous",
    "05_medium_unambiguous",
    "06_hard_unambiguous",
)
EXPECTED_CASES_PER_CATEGORY = 50
BASELINE_CASES_PER_CATEGORY = 10
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "rsr" / "data" / "gpt4o_first10_four_categories"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _select_all(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category_name in SELECTED_CATEGORIES:
        testcase = TEST_CASE_BY_DIRECTORY[category_name]
        subset = metadata[
            (metadata["difficulty"] == testcase.difficulty)
            & (metadata["ambiguious"].astype(bool) == testcase.ambiguous)
        ]
        for (scene_id, query_obj_id), rows in subset.groupby(
            ["sceneId", "queryObjId"],
            sort=True,
        ):
            rows = rows.sort_values("split").copy()
            splits = tuple(int(value) for value in rows["split"].tolist())
            gt_values = {str(value) for value in rows["groundTruthObjIds"].tolist()}
            if len(rows) != 3 or splits != (0, 1, 2) or len(gt_values) != 1:
                continue
            selected.append(
                {
                    "scene_id": int(scene_id),
                    "query_obj_id": int(query_obj_id),
                    "difficulty": testcase.difficulty,
                    "ambiguous": testcase.ambiguous,
                    "rows": rows.to_dict(orient="records"),
                    "testcase": category_name,
                }
            )

    counts = Counter(item["testcase"] for item in selected)
    for category_name in SELECTED_CATEGORIES:
        actual = counts[category_name]
        if actual != EXPECTED_CASES_PER_CATEGORY:
            raise RuntimeError(
                f"{category_name}: expected {EXPECTED_CASES_PER_CATEGORY} "
                f"complete cases, found {actual}"
            )
    return selected


def _case_key(item: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(item["testcase"]),
        int(item["scene_id"]),
        int(item["query_obj_id"]),
    )


def _scan_existing_cases(input_root: Path) -> dict[tuple[str, int, int], Path]:
    existing: dict[tuple[str, int, int], Path] = {}
    for category_name in SELECTED_CATEGORIES:
        for case_dir in sorted((input_root / category_name).glob("scene_*")):
            metadata_path = case_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = _load_json(metadata_path)
            key = (
                category_name,
                int(metadata["scene_id"]),
                int(metadata["query_obj_id"]),
            )
            if key in existing and existing[key] != case_dir:
                raise RuntimeError(
                    f"Duplicate case {key}: {existing[key]} and {case_dir}"
                )
            existing[key] = case_dir
    return existing


def _quarantine_incomplete_case(case_dir: Path, input_root: Path) -> Path:
    quarantine_root = (
        input_root
        / ".incomplete"
        / case_dir.parent.name
    )
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / case_dir.name
    suffix = 1
    while destination.exists() or destination.is_symlink():
        destination = quarantine_root / f"{case_dir.name}.{suffix}"
        suffix += 1
    shutil.move(str(case_dir), str(destination))
    return destination


def _baseline_keys(sample_root: Path) -> set[tuple[str, int, int]]:
    manifest = _load_json(sample_root / "manifest.json")
    by_name = {
        testcase["name"]: testcase
        for testcase in manifest.get("testcases", [])
    }
    keys: set[tuple[str, int, int]] = set()
    for category_name in SELECTED_CATEGORIES:
        scene_ids = by_name[category_name].get("scene_ids", [])
        if len(scene_ids) < BASELINE_CASES_PER_CATEGORY:
            raise RuntimeError(
                f"{category_name}: baseline source has only {len(scene_ids)} cases"
            )
        for scene_id in scene_ids[:BASELINE_CASES_PER_CATEGORY]:
            metadata = _load_json(
                sample_root
                / category_name
                / f"scene_{int(scene_id)}"
                / "metadata.json"
            )
            keys.add(
                (
                    category_name,
                    int(metadata["scene_id"]),
                    int(metadata["query_obj_id"]),
                )
            )
    return keys


def _case_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": int(item["scene_id"]),
        "query_obj_id": int(item["query_obj_id"]),
        "case_directory": str(item["case_directory"]),
    }


def _manifest(
    selected: list[dict[str, Any]],
    *,
    dataset: str,
    baseline_keys: set[tuple[str, int, int]],
    exported_now: list[tuple[str, int, int]],
) -> dict[str, Any]:
    testcases = []
    for category_name in SELECTED_CATEGORIES:
        testcase = TEST_CASE_BY_DIRECTORY[category_name]
        items = [
            item for item in selected
            if item["testcase"] == category_name
        ]
        testcases.append(
            {
                "name": category_name,
                "difficulty": testcase.difficulty,
                "ambiguous": testcase.ambiguous,
                "cases": [_case_record(item) for item in items],
                "scene_ids": [int(item["scene_id"]) for item in items],
            }
        )
    return {
        "schema_version": 1,
        "dataset": dataset,
        "selection_policy": (
            "all complete scene_id+query_obj_id cases in four selected categories"
        ),
        "num_testcases": len(SELECTED_CATEGORIES),
        "cases_per_testcase": EXPECTED_CASES_PER_CATEGORY,
        "num_cases": len(selected),
        "num_annotation_runs": len(selected) * 3,
        "baseline_cases_per_testcase": BASELINE_CASES_PER_CATEGORY,
        "baseline_case_keys": [
            {
                "testcase": testcase,
                "scene_id": scene_id,
                "query_obj_id": query_obj_id,
            }
            for testcase, scene_id, query_obj_id in sorted(baseline_keys)
        ],
        "exported_now": [
            {
                "testcase": testcase,
                "scene_id": scene_id,
                "query_obj_id": query_obj_id,
            }
            for testcase, scene_id, query_obj_id in exported_now
        ],
        "perception_review_model": "gpt-4o",
        "reason_models": ["gpt-4o"],
        "algorithms": [
            {
                "name": algorithm.slug,
                "prior_prompt": algorithm.prior_prompt,
                "ranking_score": algorithm.ranking_score,
            }
            for algorithm in ALGORITHMS
        ],
        "expected_perception_runs": len(selected),
        "expected_reason_runs": len(selected) * 3 * len(ALGORITHMS),
        "testcases": testcases,
    }


def _build_remaining_view(
    full_input_root: Path,
    remaining_input_root: Path,
    remaining: list[dict[str, Any]],
    baseline_keys: set[tuple[str, int, int]],
) -> dict[str, Any]:
    if remaining_input_root.exists() or remaining_input_root.is_symlink():
        if remaining_input_root.is_symlink():
            remaining_input_root.unlink()
        else:
            shutil.rmtree(remaining_input_root)
    remaining_input_root.mkdir(parents=True)

    for item in remaining:
        category_name = str(item["testcase"])
        case_directory = str(item["case_directory"])
        source_case = full_input_root / category_name / case_directory
        target_case = remaining_input_root / category_name / case_directory
        target_case.parent.mkdir(parents=True, exist_ok=True)
        target_case.symlink_to(
            os.path.relpath(source_case, target_case.parent),
            target_is_directory=True,
        )

    manifest = _manifest(
        remaining,
        dataset="remaining_after_first_10_of_four_complete_categories",
        baseline_keys=baseline_keys,
        exported_now=[],
    )
    manifest["cases_per_testcase"] = (
        EXPECTED_CASES_PER_CATEGORY - BASELINE_CASES_PER_CATEGORY
    )
    manifest["expected_perception_runs"] = len(remaining)
    manifest["expected_reason_runs"] = len(remaining) * 3 * len(ALGORITHMS)
    _write_json(remaining_input_root / "manifest.json", manifest)
    return manifest


def prepare(
    data_root: Path,
    sample_root: Path,
    input_root: Path,
    remaining_input_root: Path,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    sample_root = sample_root.resolve()
    input_root = input_root.resolve()
    remaining_input_root = remaining_input_root.resolve()

    parquet_paths = sorted(data_root.glob("train-*.parquet"))
    if len(parquet_paths) != 2:
        raise FileNotFoundError(
            f"Expected two train parquet shards under {data_root}, "
            f"found {parquet_paths}"
        )
    npz_zip = data_root / "npz_file.zip"
    if not npz_zip.exists():
        raise FileNotFoundError(npz_zip)

    selected = _select_all(read_metadata(parquet_paths))
    baseline_keys = _baseline_keys(sample_root)
    selected_by_key = {_case_key(item): item for item in selected}
    missing_baseline = baseline_keys - set(selected_by_key)
    if missing_baseline:
        raise RuntimeError(
            f"Baseline cases missing from complete selection: {sorted(missing_baseline)}"
        )

    input_root.mkdir(parents=True, exist_ok=True)
    existing = _scan_existing_cases(input_root)
    missing = [
        item for item in selected
        if _case_key(item) not in existing
    ]

    for item in selected:
        key = _case_key(item)
        if key in existing:
            item["case_directory"] = existing[key].name
        else:
            item["case_directory"] = (
                f"scene_{item['scene_id']}_query_{item['query_obj_id']}"
            )

    quarantined: list[dict[str, str]] = []
    for item in missing:
        case_dir = (
            input_root
            / str(item["testcase"])
            / str(item["case_directory"])
        )
        if not (case_dir.exists() or case_dir.is_symlink()):
            continue
        metadata_path = case_dir / "metadata.json"
        if metadata_path.exists():
            raise RuntimeError(
                f"Refusing to replace existing complete case directory: {case_dir}"
            )
        destination = _quarantine_incomplete_case(case_dir, input_root)
        quarantined.append(
            {
                "source": str(case_dir),
                "backup": str(destination),
            }
        )
        print(
            f"[prepare] moved incomplete case to {destination}",
            flush=True,
        )

    if missing:
        images = load_selected_images(parquet_paths, missing)
        npz_index = build_npz_index(npz_zip)
        with zipfile.ZipFile(npz_zip, "r") as archive:
            for item in missing:
                scene_id = int(item["scene_id"])
                if scene_id not in npz_index:
                    raise FileNotFoundError(
                        f"No npz member for scene {scene_id}"
                    )
                export_scene(
                    item,
                    images[(scene_id, int(item["query_obj_id"]))],
                    archive,
                    npz_index[scene_id],
                    input_root,
                )

    exported_now = [_case_key(item) for item in missing]
    full_manifest = _manifest(
        selected,
        dataset="all_four_selected_categories_scene_query_cases",
        baseline_keys=baseline_keys,
        exported_now=exported_now,
    )
    _write_json(input_root / "manifest.json", full_manifest)

    remaining = [
        item for item in selected
        if _case_key(item) not in baseline_keys
    ]
    remaining_counts = Counter(item["testcase"] for item in remaining)
    expected_remaining = (
        EXPECTED_CASES_PER_CATEGORY - BASELINE_CASES_PER_CATEGORY
    )
    for category_name in SELECTED_CATEGORIES:
        if remaining_counts[category_name] != expected_remaining:
            raise RuntimeError(
                f"{category_name}: expected {expected_remaining} remaining cases, "
                f"found {remaining_counts[category_name]}"
            )
    remaining_manifest = _build_remaining_view(
        input_root,
        remaining_input_root,
        remaining,
        baseline_keys,
    )
    return {
        "full_input": full_manifest,
        "remaining_input": remaining_manifest,
        "quarantined_incomplete_cases": quarantined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Expand the existing GPT-4o first-10 experiment to all 50 cases "
            "in each of four categories without replacing previous cases."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "input",
    )
    parser.add_argument(
        "--remaining-input-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "remaining_input",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.data_root,
                args.sample_root,
                args.input_root,
                args.remaining_input_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
