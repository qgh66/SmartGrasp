from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from reason.intent_handle import resolve_intent


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_freegrasp_row(data_dir: Path, scene_id: int, query_obj_id: int | None) -> pd.Series:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    rows = df[df["sceneId"].astype(int) == int(scene_id)]
    if query_obj_id is not None:
        rows = rows[rows["queryObjId"].astype(int) == int(query_obj_id)]
    if rows.empty:
        raise ValueError(f"No FreeGrasp row for scene_id={scene_id}, query_obj_id={query_obj_id}")
    return rows.iloc[0]


def _default_summary(scene_id: int, root: Path) -> Path:
    candidates = [
        root / "data" / f"scene_{scene_id}" / "perception" / "summary.json",
        root / "sample_data" / f"scene_{scene_id}" / "perception" / "summary.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No perception summary found for scene_id={scene_id}")


def _sample_summary_paths(root: Path) -> list[Path]:
    return sorted(root.glob("sample_data/scene_*/perception/summary.json"))


def _make_perception_args(
    scene_id: int,
    query_obj_id: int | None,
    point_source: str,
    device: str | None,
    generation_args: argparse.Namespace,
) -> argparse.Namespace:
    from SmartGrasp.perception.perception import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])
    args.scene_id = scene_id
    args.scene_ids = None
    args.serve = False
    args.query_obj_id = query_obj_id
    args.point_source = point_source
    args.device = device
    for name in (
        "molmo_model_id",
        "review_model_id",
        "review_api_key_env",
        "review_base_url",
        "review_timeout",
        "segmentation_backend",
        "sam_model_id",
        "sam_point_grid_radius",
        "sam_prompt_mode",
        "sam_negative_points",
        "proposal_backend",
        "proposal_min_area_ratio",
        "proposal_max_area_ratio",
        "proposal_iou_threshold",
        "proposal_containment_threshold",
        "proposal_border_fraction_threshold",
        "max_proposal_masks",
        "sam2_points_per_side",
        "sam2_crop_n_layers",
        "sam2_pred_iou_thresh",
        "sam2_stability_score_thresh",
        "preserve_unclaimed_sam2",
        "save_candidates",
    ):
        value = getattr(generation_args, name, None)
        if value is not None:
            setattr(args, name, value)
    return args


def _ensure_perception_outputs(
    scene_id: int,
    query_obj_id: int | None,
    freegrasp_dir: Path,
    point_source: str,
    device: str | None,
    force: bool,
    generation_args: argparse.Namespace,
) -> Path:
    point_dir = "perception" if point_source == "molmo" else "gt"
    summary_path = ROOT / "data" / f"scene_{scene_id}" / point_dir / "summary.json"
    if summary_path.exists() and not force:
        return summary_path

    os.environ["SMARTGRASP_DATA_DIR"] = str(freegrasp_dir.resolve())
    from SmartGrasp.perception.perception import run_pipeline

    run_args = _make_perception_args(scene_id, query_obj_id, point_source, device, generation_args)
    run_pipeline(run_args)
    if not summary_path.exists():
        raise FileNotFoundError(f"Perception run did not create {summary_path}")
    return summary_path


def _resolve_from_summary(
    summary_path: Path,
    instruction: str,
    *,
    api_key_env: str,
    base_url: str | None,
    model: str | None,
) -> tuple[dict[str, Any], Any]:
    perception_dir = summary_path.parent
    result = resolve_intent(
        instruction,
        summary_path,
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
    )

    perception_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    perception_label_png = perception_summary.get("perception_label_png") or str(perception_dir / "label_3_final.png")
    selected_id = result.target_object.object_id if result.target_object else None
    candidate_ids = [obj.object_id for obj in result.candidates]
    payload = {
        "scene_id": perception_summary.get("scene_id"),
        "instruction_used": instruction,
        "summary": str(summary_path),
        "perception_outputs": {
            "dir": str(perception_dir),
            "labeled_rgb": perception_label_png,
            "occlusion_graph_json": str(perception_dir / "occlusion_graph.json"),
            "occlusion_graph_png": str(perception_dir / "occlusion_graph.png"),
            "mask_dir": str(perception_dir / "mask"),
            "sam2_auto_label_png": perception_summary.get("sam2_auto_label_png"),
            "sam2_rgb_parts_sheet_png": perception_summary.get("sam2_rgb_parts_sheet_png"),
            "openai_sam2_review_json": perception_summary.get("openai_sam2_review_json"),
            "molmo_sam2_review_json": perception_summary.get("molmo_sam2_review_json"),
        },
        "perception_objects": perception_summary.get("molmo_points", []),
        "intent": {
            "selected_perception_id": selected_id,
            "selected_object": result.target_object.to_json() if result.target_object else None,
            "candidate_perception_ids": candidate_ids,
            "branch": result.branch,
            "occluded_by": list(result.occluded_by),
            "reason": result.reason,
            "raw_vlm_decision": result.vlm_decision,
        },
    }
    return payload, result


def _write_sample_intent_id(summary_path: Path, selected_id: int | None) -> Path:
    scene_dir = summary_path.parent.parent
    output_dir = scene_dir / "intent_id"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "id.txt"
    output_path.write_text(f"{selected_id if selected_id is not None else 'none'}\n", encoding="utf-8")
    return output_path


def _run_sample_mode(args: argparse.Namespace) -> int:
    summary_paths = _sample_summary_paths(ROOT)
    if args.scene_id is not None:
        summary_paths = [
            path
            for path in summary_paths
            if path.parent.parent.name == f"scene_{args.scene_id}"
        ]
    if not summary_paths:
        raise FileNotFoundError("No sample_data/scene_*/perception/summary.json files found.")

    rows = []
    for summary_path in summary_paths:
        perception_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        instruction = args.instruction or str(perception_summary.get("annotation", "")).strip()
        if not instruction:
            raise ValueError(f"Missing annotation in {summary_path}")
        payload, result = _resolve_from_summary(
            summary_path,
            instruction,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=args.model,
        )
        selected_id = result.target_object.object_id if result.target_object else None
        id_path = _write_sample_intent_id(summary_path, selected_id)
        rows.append(
            {
                "scene_id": payload["scene_id"],
                "summary": str(summary_path),
                "instruction": instruction,
                "selected_perception_id": selected_id,
                "id_txt": str(id_path),
            }
        )
        print(
            f"scene_{payload['scene_id']}: selected_perception_id={selected_id} -> {id_path}",
            flush=True,
        )

    print(json.dumps({"mode": "sample", "count": len(rows), "results": rows}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate/read perception outputs and resolve a FreeGrasp query to a perception object id.",
    )
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--query-obj-id", type=int, default=None)
    parser.add_argument("--freegrasp-dir", type=Path, default=ROOT / "freegrasp")
    parser.add_argument("--summary", type=Path, default=None, help="Perception summary.json. Defaults to data/, then sample_data/.")
    parser.add_argument("--instruction", default=None, help="Override FreeGrasp annotation.")
    parser.add_argument("--use-sample", action="store_true", help="Run all sample_data/scene_*/perception summaries and write scene_<id>/intent_id/id.txt.")
    parser.add_argument("--use", choices=["sample"], default=None, help="Alias for --use-sample when set to sample.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate data/scene_<id>/<point-source>/ from freegrasp before running intent.",
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Regenerate outputs even if summary.json already exists.",
    )
    parser.add_argument(
        "--point-source",
        choices=["molmo", "gt-centers"],
        default="molmo",
        help="Generation mode used with --generate.",
    )
    parser.add_argument("--device", default="cuda", help="Device used by perception generation.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--molmo-model-id", default=None)
    parser.add_argument("--review-model-id", default=None)
    parser.add_argument("--review-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--review-base-url", default=None)
    parser.add_argument("--review-timeout", type=float, default=None)
    parser.add_argument(
        "--segmentation-backend",
        choices=["sam2-molmo-langsam", "sam2-anchor", "sam", "langsam", "auto"],
        default="sam2-molmo-langsam",
        help="Perception backend used with --generate. Default matches the sample_data review pipeline.",
    )
    parser.add_argument("--sam-model-id", default=None)
    parser.add_argument("--sam-point-grid-radius", type=int, default=None)
    parser.add_argument("--sam-prompt-mode", choices=["cross", "grid", "ring", "auto"], default=None)
    parser.add_argument("--sam-negative-points", type=int, default=None)
    parser.add_argument("--proposal-backend", choices=["none", "sam2-auto"], default=None)
    parser.add_argument("--proposal-min-area-ratio", type=float, default=None)
    parser.add_argument("--proposal-max-area-ratio", type=float, default=None)
    parser.add_argument("--proposal-iou-threshold", type=float, default=None)
    parser.add_argument("--proposal-containment-threshold", type=float, default=None)
    parser.add_argument("--proposal-border-fraction-threshold", type=float, default=None)
    parser.add_argument("--max-proposal-masks", type=int, default=None)
    parser.add_argument("--sam2-points-per-side", type=int, default=None)
    parser.add_argument("--sam2-crop-n-layers", type=int, default=None)
    parser.add_argument("--sam2-pred-iou-thresh", type=float, default=None)
    parser.add_argument("--sam2-stability-score-thresh", type=float, default=None)
    parser.add_argument("--preserve-unclaimed-sam2", type=int, default=None)
    parser.add_argument("--save-candidates", action="store_true", default=None)
    args = parser.parse_args()

    if not os.environ.get(args.api_key_env) and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"Missing API key. Export {args.api_key_env}=... before running run_intent.py."
        )

    if args.use_sample or args.use == "sample":
        return _run_sample_mode(args)

    if args.scene_id is None:
        raise ValueError("--scene-id is required unless --use-sample is set.")

    row = _load_freegrasp_row(args.freegrasp_dir, args.scene_id, args.query_obj_id)
    query_obj_id = int(row["queryObjId"])
    instruction = args.instruction or str(row["annotation"])

    if args.generate:
        summary_path = _ensure_perception_outputs(
            scene_id=args.scene_id,
            query_obj_id=args.query_obj_id,
            freegrasp_dir=args.freegrasp_dir,
            point_source=args.point_source,
            device=args.device,
            force=args.force_generate,
            generation_args=args,
        )
    else:
        summary_path = args.summary or _default_summary(args.scene_id, ROOT)
    payload, _result = _resolve_from_summary(
        summary_path,
        instruction,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
    )
    payload.update({
        "freegrasp": {
            "query_obj_id": query_obj_id,
            "ground_truth_obj_ids": str(row.get("groundTruthObjIds")),
            "annotation": str(row["annotation"]),
        },
    })
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
