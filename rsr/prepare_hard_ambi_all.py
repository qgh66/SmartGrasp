"""Export every Hard+ambiguous ``(scene_id, query_obj_id)`` case.

Unlike the fixed 6x20 sampler, this exporter intentionally keeps multiple
query objects from the same source scene.  Case directory names therefore
contain both ids so their perception and reason outputs cannot overwrite one
another.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from .config import ALGORITHMS, DEFAULT_DATA_ROOT, PROJECT_ROOT
from .prepare_inputs import (
    build_npz_index,
    export_scene,
    load_selected_images,
    read_metadata,
)


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "rsr" / "data" / "hard_ambi_all" / "input"
TESTCASE = "01_hard_ambiguous"


def _select_all(metadata: pd.DataFrame) -> list[dict]:
    subset = metadata[
        (metadata["difficulty"] == "Hard")
        & (metadata["ambiguious"].astype(bool))
    ]
    selected = []
    for (scene_id, query_obj_id), rows in subset.groupby(
        ["sceneId", "queryObjId"], sort=True
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
                "difficulty": "Hard",
                "ambiguous": True,
                "rows": rows.to_dict(orient="records"),
                "testcase": TESTCASE,
                "case_directory": (
                    f"scene_{int(scene_id)}_query_{int(query_obj_id)}"
                ),
            }
        )
    return selected


def prepare(data_root: Path, input_root: Path) -> dict:
    data_root = data_root.resolve()
    input_root = input_root.resolve()
    parquet_paths = sorted(data_root.glob("train-*.parquet"))
    if len(parquet_paths) != 2:
        raise FileNotFoundError(
            f"Expected two train parquet shards under {data_root}, found {parquet_paths}"
        )
    npz_zip = data_root / "npz_file.zip"
    if not npz_zip.exists():
        raise FileNotFoundError(npz_zip)
    if input_root.exists():
        raise FileExistsError(
            f"{input_root} already exists; remove it explicitly before regenerating"
        )

    metadata = read_metadata(parquet_paths)
    selected = _select_all(metadata)
    images = load_selected_images(parquet_paths, selected)
    npz_index = build_npz_index(npz_zip)
    exported = []
    with zipfile.ZipFile(npz_zip, "r") as archive:
        for item in selected:
            scene_id = item["scene_id"]
            if scene_id not in npz_index:
                raise FileNotFoundError(f"No npz member for scene {scene_id}")
            exported.append(
                export_scene(
                    item,
                    images[(scene_id, item["query_obj_id"])],
                    archive,
                    npz_index[scene_id],
                    input_root,
                )
            )

    cases = [
        {
            "scene_id": item["scene_id"],
            "query_obj_id": item["query_obj_id"],
            "case_directory": item["case_directory"],
        }
        for item in selected
    ]
    manifest = {
        "schema_version": 1,
        "dataset": "all_hard_ambiguous_scene_query_cases",
        "num_cases": len(cases),
        "num_unique_scenes": len({item["scene_id"] for item in cases}),
        "num_annotation_runs": len(cases) * 3,
        "reason_models": ["gpt-5.5"],
        "algorithms": [
            {
                "name": algorithm.slug,
                "prior_prompt": algorithm.prior_prompt,
                "ranking_score": algorithm.ranking_score,
            }
            for algorithm in ALGORITHMS
        ],
        "expected_reason_runs": len(cases) * 3 * len(ALGORITHMS),
        "testcases": [
            {
                "name": TESTCASE,
                "difficulty": "Hard",
                "ambiguous": True,
                "cases": cases,
            }
        ],
    }
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all Hard+ambiguous scene/query cases."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.input_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
