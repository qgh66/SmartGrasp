from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    ALGORITHMS,
    DEFAULT_DATA_ROOT,
    DEFAULT_INPUT_ROOT,
    REASON_MODELS,
    TEST_CASES,
    parse_ground_truth_ids,
)


METADATA_COLUMNS = [
    "sceneId",
    "queryObjId",
    "annotation",
    "groundTruthObjIds",
    "difficulty",
    "ambiguious",
    "split",
]


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _image_bytes(image_field: Any) -> bytes:
    if isinstance(image_field, dict):
        raw = image_field.get("bytes")
        if raw is not None:
            return bytes(raw)
        path = image_field.get("path")
        if path:
            return Path(path).read_bytes()
    raise ValueError(f"Unsupported parquet image field: {type(image_field)!r}")


def _npz_key(npz: np.lib.npyio.NpzFile, candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in npz.files:
            return key
    raise KeyError(f"None of {candidates!r} found in npz keys {npz.files!r}")


def read_metadata(parquet_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in parquet_paths:
        frame = pd.read_parquet(path, columns=METADATA_COLUMNS)
        frame = frame.copy()
        frame["source_parquet"] = str(path.resolve())
        frame["source_shard"] = path.name
        frame["source_row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def select_cases(metadata: pd.DataFrame, per_case: int, seed: int) -> list[dict[str, Any]]:
    grouped = []
    keys = ["sceneId", "queryObjId", "difficulty", "ambiguious"]
    for key, rows in metadata.groupby(keys, sort=True, dropna=False):
        rows = rows.sort_values("split").copy()
        splits = tuple(int(value) for value in rows["split"].tolist())
        annotations = tuple(str(value).strip() for value in rows["annotation"].tolist())
        gt_values = {str(value) for value in rows["groundTruthObjIds"].tolist()}
        if len(rows) != 3 or splits != (0, 1, 2):
            continue
        if len(set(annotations)) != 3:
            continue
        if len(gt_values) != 1:
            continue
        grouped.append({
            "scene_id": int(key[0]),
            "query_obj_id": int(key[1]),
            "difficulty": str(key[2]),
            "ambiguous": bool(key[3]),
            "rows": rows.to_dict(orient="records"),
        })

    rng = random.Random(seed)
    used_scene_ids: set[int] = set()
    selected: list[dict[str, Any]] = []
    for testcase in TEST_CASES:
        candidates = [
            item for item in grouped
            if item["difficulty"] == testcase.difficulty
            and item["ambiguous"] == testcase.ambiguous
            and item["scene_id"] not in used_scene_ids
        ]
        rng.shuffle(candidates)
        chosen = candidates[:per_case]
        if len(chosen) != per_case:
            raise RuntimeError(
                f"{testcase.directory_name}: need {per_case} globally unique scenes, "
                f"only found {len(chosen)}"
            )
        for item in chosen:
            item["testcase"] = testcase.directory_name
            used_scene_ids.add(item["scene_id"])
            selected.append(item)
    return selected


def load_selected_images(
    parquet_paths: list[Path],
    selected: list[dict[str, Any]],
) -> dict[tuple[int, int], bytes]:
    wanted = {(item["scene_id"], item["query_obj_id"]) for item in selected}
    images: dict[tuple[int, int], list[bytes]] = {key: [] for key in wanted}
    scene_ids = sorted({scene_id for scene_id, _ in wanted})
    for path in parquet_paths:
        frame = pd.read_parquet(
            path,
            columns=["sceneId", "queryObjId", "image"],
            filters=[("sceneId", "in", scene_ids)],
        )
        for row in frame.itertuples(index=False):
            key = (int(row.sceneId), int(row.queryObjId))
            if key in wanted:
                images[key].append(_image_bytes(row.image))

    result: dict[tuple[int, int], bytes] = {}
    for key, variants in images.items():
        if len(variants) != 3:
            raise RuntimeError(f"Expected 3 RGB records for scene/query {key}, found {len(variants)}")
        digests = {hashlib.sha256(raw).hexdigest() for raw in variants}
        if len(digests) != 1:
            raise RuntimeError(f"The 3 annotations do not share one RGB image for scene/query {key}")
        result[key] = variants[0]
    return result


def build_npz_index(npz_zip: Path) -> dict[int, str]:
    with zipfile.ZipFile(npz_zip, "r") as archive:
        index = {
            int(Path(member).stem): member
            for member in archive.namelist()
            if member.endswith(".npz") and Path(member).stem.isdigit()
        }
    if not index:
        raise RuntimeError(f"No numeric .npz members found in {npz_zip}")
    return index


def export_scene(
    item: dict[str, Any],
    image_bytes: bytes,
    archive: zipfile.ZipFile,
    npz_member: str,
    input_root: Path,
) -> dict[str, Any]:
    scene_id = int(item["scene_id"])
    scene_dir = input_root / item["testcase"] / item.get(
        "case_directory", f"scene_{scene_id}"
    )
    scene_dir.mkdir(parents=True, exist_ok=False)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_path = scene_dir / "scene_image.png"
    image.save(image_path)

    npz_bytes = archive.read(npz_member)
    (scene_dir / "source.npz").write_bytes(npz_bytes)
    with np.load(io.BytesIO(npz_bytes), allow_pickle=True) as npz:
        depth_key = _npz_key(npz, ("depth", "depth.npy"))
        instances_key = _npz_key(
            npz,
            ("instances_objects", "instances_objects.npy", "instance_mask", "mask"),
        )
        depth = np.asarray(npz[depth_key], dtype=np.float32)
        instances = np.asarray(npz[instances_key])
    np.save(scene_dir / "depth.npy", depth)
    np.save(scene_dir / "instances_objects.npy", instances)

    rows = sorted(item["rows"], key=lambda row: int(row["split"]))
    gt_ids = parse_ground_truth_ids(rows[0]["groundTruthObjIds"])
    annotation_records = []
    for row in rows:
        split = int(row["split"])
        annotation_dir = scene_dir / "annotations" / f"split_{split}"
        annotation_dir.mkdir(parents=True, exist_ok=False)
        instruction = str(row["annotation"]).strip()
        (annotation_dir / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")
        annotation_metadata = {
            "split": split,
            "instruction": instruction,
            "source_parquet": row["source_parquet"],
            "source_shard": row["source_shard"],
            "source_row": int(row["source_row"]),
        }
        _json_dump(annotation_dir / "metadata.json", annotation_metadata)
        annotation_records.append(annotation_metadata)

    metadata_payload = {
        "schema_version": 1,
        "testcase": item["testcase"],
        "scene_id": scene_id,
        "query_obj_id": int(item["query_obj_id"]),
        "difficulty": item["difficulty"],
        "ambiguous": bool(item["ambiguous"]),
        "ambiguious": bool(item["ambiguous"]),
        "ground_truth_object_ids": gt_ids,
        "ground_truth_npz_labels": [value + 1 for value in gt_ids],
        "id_convention": {
            "parquet_object_ids": "zero_based",
            "npz_instance_labels": "one_based_with_zero_as_background",
        },
        "perception_policy": "run once per scene and reuse for all three annotations",
        "rgb": "scene_image.png",
        "depth": "depth.npy",
        "instances_objects": "instances_objects.npy",
        "source_npz": "source.npz",
        "source_npz_member": npz_member,
        "annotations": annotation_records,
    }
    _json_dump(scene_dir / "metadata.json", metadata_payload)
    return metadata_payload


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    input_root = args.input_root.resolve()
    parquet_paths = sorted(data_root.glob("train-*.parquet"))
    if len(parquet_paths) != 2:
        raise FileNotFoundError(f"Expected two train parquet shards under {data_root}, found {parquet_paths}")
    npz_zip = data_root / "npz_file.zip"
    if not npz_zip.exists():
        raise FileNotFoundError(npz_zip)

    if input_root.exists():
        if not args.force:
            raise FileExistsError(f"{input_root} already exists; pass --force to replace only this rsr input tree")
        shutil.rmtree(input_root)
    input_root.mkdir(parents=True)

    metadata = read_metadata(parquet_paths)
    selected = select_cases(metadata, per_case=args.per_case, seed=args.seed)
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

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "scenes_per_testcase": args.per_case,
        "num_testcases": len(TEST_CASES),
        "num_scenes": len(exported),
        "num_annotation_runs": len(exported) * 3,
        "reason_models": list(REASON_MODELS),
        "algorithms": [
            {
                "name": algorithm.slug,
                "prior_prompt": algorithm.prior_prompt,
                "ranking_score": algorithm.ranking_score,
            }
            for algorithm in ALGORITHMS
        ],
        "expected_reason_runs": len(exported) * 3 * len(REASON_MODELS) * len(ALGORITHMS),
        "testcases": [
            {
                "name": testcase.directory_name,
                "difficulty": testcase.difficulty,
                "ambiguous": testcase.ambiguous,
                "scene_ids": [
                    item["scene_id"] for item in exported
                    if item["testcase"] == testcase.directory_name
                ],
            }
            for testcase in TEST_CASES
        ],
    }
    _json_dump(input_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 6 x 20 FreeGrasp RSR inputs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--per-case", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
