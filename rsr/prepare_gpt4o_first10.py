from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import ALGORITHMS, PROJECT_ROOT, TEST_CASE_BY_DIRECTORY


SELECTED_CATEGORIES = (
    "01_hard_ambiguous",
    "02_medium_ambiguous",
    "05_medium_unambiguous",
    "06_hard_unambiguous",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(source_root: Path, target_root: Path, limit: int) -> dict[str, Any]:
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    source_manifest = _load_json(source_root / "manifest.json")
    source_testcases = {
        item["name"]: item for item in source_manifest.get("testcases", [])
    }

    exported: list[dict[str, Any]] = []
    manifest_testcases = []
    for category_name in SELECTED_CATEGORIES:
        if category_name not in source_testcases:
            raise KeyError(f"Missing {category_name} in {source_root / 'manifest.json'}")
        source_scene_ids = source_testcases[category_name].get("scene_ids", [])
        selected_scene_ids = [int(value) for value in source_scene_ids[:limit]]
        if len(selected_scene_ids) != limit:
            raise RuntimeError(
                f"{category_name}: requested {limit} cases, found {len(selected_scene_ids)}"
            )

        target_category = target_root / category_name
        target_category.mkdir(parents=True, exist_ok=True)
        cases = []
        for scene_id in selected_scene_ids:
            source_scene = source_root / category_name / f"scene_{scene_id}"
            if not source_scene.is_dir():
                raise FileNotFoundError(source_scene)
            metadata = _load_json(source_scene / "metadata.json")
            case_directory = source_scene.name
            target_scene = target_category / case_directory
            if target_scene.exists() or target_scene.is_symlink():
                if target_scene.resolve() != source_scene.resolve():
                    raise FileExistsError(
                        f"Refusing to replace unrelated input path: {target_scene}"
                    )
            else:
                target_scene.symlink_to(
                    os.path.relpath(source_scene, target_scene.parent),
                    target_is_directory=True,
                )
            case = {
                "scene_id": scene_id,
                "query_obj_id": int(metadata["query_obj_id"]),
                "case_directory": case_directory,
            }
            cases.append(case)
            exported.append({"testcase": category_name, **case})

        testcase = TEST_CASE_BY_DIRECTORY[category_name]
        manifest_testcases.append(
            {
                "name": category_name,
                "difficulty": testcase.difficulty,
                "ambiguous": testcase.ambiguous,
                "cases": cases,
                "scene_ids": selected_scene_ids,
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": "first_10_cases_from_rsr_data_input_for_four_categories",
        "source_manifest": str((source_root / "manifest.json").resolve()),
        "selection_policy": "first 10 entries in each source manifest testcase",
        "num_testcases": len(SELECTED_CATEGORIES),
        "cases_per_testcase": limit,
        "num_cases": len(exported),
        "num_annotation_runs": len(exported) * 3,
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
        "expected_perception_runs": len(exported),
        "expected_reason_runs": len(exported) * 3 * len(ALGORITHMS),
        "testcases": manifest_testcases,
    }
    _write_json(target_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a non-destructive first-10 input view for four GPT-4o RSR categories."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "rsr" / "data" / "input",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "rsr"
            / "data"
            / "gpt4o_first10_four_categories"
            / "input"
        ),
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source_root, args.target_root, args.limit),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
