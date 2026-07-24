from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    ALGORITHMS,
    ALGORITHM_BY_SLUG,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    REASON_MODELS,
    TEST_CASES,
    safe_model_name,
)


RESULT_CSV_FIELDS = [
    "model", "algorithm", "testcase", "scene_id", "query_obj_id", "difficulty", "ambiguous", "split",
    "instruction", "intent_selected_id", "intent_selected_object", "branch", "reason_grasp_id",
    "reason_grasp_object", "reason_status", "selection_status", "ground_truth_object_ids",
    "predicted_perception_object_id", "predicted_dataset_object_id", "ssr_iou", "rsr_success", "status",
]


def selected_testcases(names: list[str] | None):
    if not names:
        return TEST_CASES
    wanted = set(names)
    selected = [
        testcase
        for testcase in TEST_CASES
        if testcase.slug in wanted or testcase.directory_name in wanted
    ]
    known = {
        name
        for testcase in selected
        for name in (testcase.slug, testcase.directory_name)
    }
    missing = wanted - known
    if missing:
        raise ValueError(f"Unknown testcase names: {sorted(missing)}")
    return tuple(selected)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_object_mask(mask_dir: Path, object_id: int) -> Path | None:
    patterns = (f"mask_{object_id:03d}_*.png", f"{object_id:03d}_*.png")
    for pattern in patterns:
        matches = sorted(mask_dir.glob(pattern))
        if matches:
            return matches[0]
    for name in (f"mask_{object_id:03d}.png", f"{object_id:03d}.png"):
        candidate = mask_dir / name
        if candidate.exists():
            return candidate
    return None


def _point_for_id(perception_summary: dict[str, Any], object_id: int) -> tuple[int, int] | None:
    for point in perception_summary.get("molmo_points", []) or perception_summary.get("object_points", []):
        raw_id = point.get("molmo_id", point.get("object_id"))
        try:
            if int(raw_id) != object_id:
                continue
            return int(point["x"]), int(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def map_perception_id_to_gt(
    object_id: int | None,
    perception_dir: Path,
    instances: np.ndarray,
) -> dict[str, Any]:
    if object_id is None:
        return {"perception_object_id": None, "npz_label": None, "dataset_object_id": None, "method": "missing"}

    mask_path = _find_object_mask(perception_dir / "mask", int(object_id))
    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        if mask.shape == instances.shape:
            labels = instances[mask]
            counts = Counter(int(value) for value in labels if int(value) > 0)
            if counts:
                label, pixels = counts.most_common(1)[0]
                return {
                    "perception_object_id": int(object_id),
                    "npz_label": label,
                    "dataset_object_id": label - 1,
                    "method": "mask_majority_overlap",
                    "overlap_pixels": pixels,
                    "mask_path": str(mask_path),
                }

    summary_path = perception_dir / "summary.json"
    if summary_path.exists():
        point = _point_for_id(_load_json(summary_path), int(object_id))
        if point is not None:
            x, y = point
            if 0 <= y < instances.shape[0] and 0 <= x < instances.shape[1]:
                label = int(instances[y, x])
                if label > 0:
                    return {
                        "perception_object_id": int(object_id),
                        "npz_label": label,
                        "dataset_object_id": label - 1,
                        "method": "point_lookup",
                        "point": {"x": x, "y": y},
                    }

    return {
        "perception_object_id": int(object_id),
        "npz_label": None,
        "dataset_object_id": None,
        "method": "unmapped",
    }


def selected_mask_ssr(
    object_id: int | None,
    perception_dir: Path,
    instances: np.ndarray,
    valid_dataset_ids: list[int],
) -> float:
    """Return the best mask IoU against the valid GT objects for one annotation.

    Missing selections, missing masks, and incompatible mask shapes count as
    zero so SSR uses the same full-denominator rule as the current RSR
    evaluator.
    """
    if object_id is None:
        return 0.0

    mask_path = _find_object_mask(perception_dir / "mask", int(object_id))
    if mask_path is None:
        return 0.0

    predicted_mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    if predicted_mask.shape != instances.shape:
        return 0.0

    best_iou = 0.0
    for dataset_object_id in valid_dataset_ids:
        gt_mask = instances == (int(dataset_object_id) + 1)
        union = np.logical_or(predicted_mask, gt_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(predicted_mask, gt_mask).sum()
        best_iou = max(best_iou, float(intersection / union))
    return best_iou


def evaluate_annotation(
    input_scene: Path,
    perception_scene: Path,
    result_scene: Path,
    split: int,
    model: str,
    algorithm: str,
) -> dict[str, Any]:
    metadata = _load_json(input_scene / "metadata.json")
    split_root = result_scene / "annotations" / f"split_{split}"
    run_scene = split_root / f"scene_{metadata['scene_id']}"
    reason_summary_path = run_scene / "reason" / "summary.json"
    selection_path = split_root / "selection.json"
    perception_dir = perception_scene / "perception"
    instances = np.load(input_scene / "instances_objects.npy")

    predicted_id = None
    status = "missing_reason_output"
    reason_summary = None
    if reason_summary_path.exists():
        reason_summary = _load_json(reason_summary_path)
        predicted_id = reason_summary.get("grasp_object", {}).get("id")
        status = "ok" if predicted_id is not None else "missing_grasp_object"

    selection = _load_json(selection_path) if selection_path.exists() else {}
    intent_selection = selection.get("intent", {}) or {}
    reason_selection = selection.get("reason", {}) or {}

    mapping = map_perception_id_to_gt(predicted_id, perception_dir, instances)
    valid_ids = [int(value) for value in metadata["ground_truth_object_ids"]]
    ssr_iou = selected_mask_ssr(predicted_id, perception_dir, instances, valid_ids)
    correct = int(mapping["dataset_object_id"] in valid_ids) if mapping["dataset_object_id"] is not None else 0
    result = {
        "model": model,
        "algorithm": algorithm,
        "testcase": metadata["testcase"],
        "scene_id": int(metadata["scene_id"]),
        "query_obj_id": int(metadata["query_obj_id"]),
        "difficulty": metadata["difficulty"],
        "ambiguous": bool(metadata["ambiguous"]),
        "split": int(split),
        "instruction": next(
            item["instruction"] for item in metadata["annotations"] if int(item["split"]) == int(split)
        ),
        "intent_selected_id": intent_selection.get("selected_id"),
        "intent_selected_object": intent_selection.get("selected_object"),
        "branch": reason_summary.get("branch") if reason_summary else None,
        "reason_grasp_id": reason_selection.get("grasp_id", predicted_id),
        "reason_grasp_object": reason_selection.get("grasp_object"),
        "reason_status": reason_summary.get("status") if reason_summary else None,
        "selection_status": selection.get("status", "missing"),
        "ground_truth_object_ids": valid_ids,
        "predicted_perception_object_id": predicted_id,
        "predicted_dataset_object_id": mapping["dataset_object_id"],
        "mapping": mapping,
        "ssr_iou": ssr_iou,
        "rsr_success": correct,
        "status": status,
        "reason_summary": str(reason_summary_path),
    }
    result_path = result_scene / "annotations" / f"split_{split}" / "rsr_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_configuration(
    input_root: Path,
    output_root: Path,
    model: str,
    algorithm: str,
    testcases=TEST_CASES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    category_summaries = []
    result_root = output_root / "results" / safe_model_name(model) / algorithm
    report_root = output_root / "reports" / safe_model_name(model) / algorithm
    for testcase in testcases:
        input_category = input_root / testcase.directory_name
        perception_category = output_root / "perception" / testcase.directory_name
        result_category = result_root / testcase.directory_name
        for input_scene in sorted(input_category.glob("scene_*")):
            perception_scene = perception_category / input_scene.name
            result_scene = result_category / input_scene.name
            for split in (0, 1, 2):
                rows.append(
                    evaluate_annotation(
                        input_scene,
                        perception_scene,
                        result_scene,
                        split,
                        model,
                        algorithm,
                    )
                )

        split_rates = {}
        for split in (0, 1, 2):
            split_rows = [
                row for row in rows
                if row["testcase"] == testcase.directory_name and row["split"] == split
            ]
            split_rates[str(split)] = fmean(row["rsr_success"] for row in split_rows) if split_rows else 0.0
        values = list(split_rates.values())
        category_rows = [row for row in rows if row["testcase"] == testcase.directory_name]
        category_summaries.append({
            "testcase": testcase.directory_name,
            "difficulty": testcase.difficulty,
            "ambiguous": testcase.ambiguous,
            "num_scenes": len(category_rows) // 3,
            "num_annotation_runs": len(category_rows),
            "split_rsr": split_rates,
            "ssr_mean_iou": (
                fmean(row["ssr_iou"] for row in category_rows)
                if category_rows else 0.0
            ),
            "rsr_mean": fmean(values),
            "rsr_std": pstdev(values),
            "rsr_pooled": fmean(row["rsr_success"] for row in category_rows) if category_rows else 0.0,
        })

    report_root.mkdir(parents=True, exist_ok=True)
    with (report_root / "rsr_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "metric": "RSR",
        "model": model,
        "algorithm": algorithm,
        "ranking_score": ALGORITHM_BY_SLUG[algorithm].ranking_score,
        "prior_prompt": ALGORITHM_BY_SLUG[algorithm].prior_prompt,
        "definition": "predicted first-grasp object maps to one of groundTruthObjIds",
        "failed_or_missing_runs_count_as_zero": True,
        "num_rows": len(rows),
        "categories": category_summaries,
    }
    (report_root / "rsr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_matrix_csv_reports(
    output_root: Path,
    configurations: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    detail_path = report_root / "rsr_summary_table.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=RESULT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for configuration in configurations:
            source = (
                report_root
                / safe_model_name(configuration["model"])
                / configuration["algorithm"]
                / "rsr_results.csv"
            )
            with source.open("r", newline="", encoding="utf-8") as input_handle:
                writer.writerows(csv.DictReader(input_handle))

    aggregate_fields = [
        "model", "algorithm", "ranking_score", "prior_prompt", "testcase", "difficulty", "ambiguous",
        "num_scenes", "num_annotation_runs", "ssr_mean_iou", "split_0_rsr", "split_1_rsr",
        "split_2_rsr", "rsr_mean", "rsr_std", "rsr_pooled",
    ]
    aggregate_path = report_root / "rsr_aggregate_summary.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        for configuration in configurations:
            for category in configuration["categories"]:
                writer.writerow({
                    "model": configuration["model"],
                    "algorithm": configuration["algorithm"],
                    "ranking_score": configuration["ranking_score"],
                    "prior_prompt": configuration["prior_prompt"],
                    "testcase": category["testcase"],
                    "difficulty": category["difficulty"],
                    "ambiguous": category["ambiguous"],
                    "num_scenes": category["num_scenes"],
                    "num_annotation_runs": category["num_annotation_runs"],
                    "ssr_mean_iou": category["ssr_mean_iou"],
                    "split_0_rsr": category["split_rsr"]["0"],
                    "split_1_rsr": category["split_rsr"]["1"],
                    "split_2_rsr": category["split_rsr"]["2"],
                    "rsr_mean": category["rsr_mean"],
                    "rsr_std": category["rsr_std"],
                    "rsr_pooled": category["rsr_pooled"],
                })
    timeout_fields = [
        "scene_id", "split", "model", "algorithm", "prior_prompt", "ranking_score", "instruction",
        "timeout_seconds_per_query", "elapsed_seconds", "reason_status", "log",
    ]
    timeout_path = report_root / "timeout_failures.csv"
    with timeout_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=timeout_fields, extrasaction="ignore")
        writer.writeheader()
        for path in sorted((output_root / "results").glob("*/*/*/scene_*/annotations/split_*/timeout.json")):
            writer.writerow(_load_json(path))
    return detail_path, aggregate_path, timeout_path


def evaluate_all(
    input_root: Path,
    output_root: Path,
    models: list[str] | tuple[str, ...] = REASON_MODELS,
    algorithms: list[str] | tuple[str, ...] = tuple(item.slug for item in ALGORITHMS),
    testcases=TEST_CASES,
) -> dict[str, Any]:
    configurations = [
        evaluate_configuration(
            input_root,
            output_root,
            model,
            algorithm,
            testcases=testcases,
        )
        for model in models
        for algorithm in algorithms
    ]
    detail_csv, aggregate_csv, timeout_csv = write_matrix_csv_reports(output_root, configurations)
    summary = {
        "metric": "RSR",
        "models": list(models),
        "algorithms": list(algorithms),
        "same_input_manifest_for_all_configurations": str((input_root / "manifest.json").resolve()),
        "num_configurations": len(configurations),
        "detail_csv": str(detail_csv.resolve()),
        "aggregate_csv": str(aggregate_csv.resolve()),
        "timeout_csv": str(timeout_csv.resolve()),
        "configurations": configurations,
    }
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "rsr_matrix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FreeGrasp RSR from SmartGrasp reason outputs.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--algorithm", action="append", choices=sorted(ALGORITHM_BY_SLUG), default=None)
    parser.add_argument(
        "--testcase",
        action="append",
        default=None,
        help="Slug or numbered testcase directory; repeatable.",
    )
    args = parser.parse_args()
    print(json.dumps(
        evaluate_all(
            args.input_root.resolve(),
            args.output_root.resolve(),
            models=args.model or REASON_MODELS,
            algorithms=args.algorithm or [item.slug for item in ALGORITHMS],
            testcases=selected_testcases(args.testcase),
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
