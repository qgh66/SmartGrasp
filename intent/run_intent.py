from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reason.intent_handle import resolve_intent

if load_dotenv is not None:
    load_dotenv()

with open(ROOT / "api_config.json", encoding="utf-8") as _f:
    _cfg = json.load(_f)

RUN_INTENT_MODEL = "gpt-5.5"
RUN_INTENT_BASE_URL: str = _cfg["base_url"]
RUN_INTENT_API_KEY_ENV = "OPENAI_API_KEY"
RUN_INTENT_TIMEOUT = 300.0


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


def _data_summary_paths(root: Path) -> list[Path]:
    return sorted(root.glob("data/scene_*/perception/summary.json"))


def _resolve_from_summary(
    summary_path: Path,
    instruction: str,
    *,
    api_key_env: str,
    base_url: str | None,
    model: str | None,
    timeout: float,
) -> tuple[dict[str, Any], Any]:
    perception_dir = summary_path.parent
    result = resolve_intent(
        instruction,
        summary_path,
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        timeout=timeout,
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
            "final_objects_sheet_png": perception_summary.get("final_objects_sheet_png"),
            "openai_sam2_review_json": perception_summary.get("openai_sam2_review_json"),
            "molmo_sam2_review_json": perception_summary.get("molmo_sam2_review_json"),
        },
        "perception_objects": perception_summary.get("molmo_points", []),
        "intent": {
            "selected_perception_id": selected_id,
            "selected_object": result.target_object.to_json() if result.target_object else None,
            "candidate_perception_ids": candidate_ids,
            "reason": result.reason,
            "raw_vlm_decision": result.vlm_decision,
        },
    }
    return payload, result


def _write_sample_intent_id(summary_path: Path, selected_id: int | None) -> Path:
    scene_dir = summary_path.parent.parent
    output_dir = scene_dir / "intent"
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
            timeout=args.timeout,
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


def _run_data_mode(args: argparse.Namespace) -> int:
    summary_paths = _data_summary_paths(ROOT)
    if args.scene_id is not None:
        summary_paths = [
            path
            for path in summary_paths
            if path.parent.parent.name == f"scene_{args.scene_id}"
        ]
    if not summary_paths:
        raise FileNotFoundError("No data/scene_*/perception/summary.json files found.")

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
            timeout=args.timeout,
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

    print(json.dumps({"mode": "data", "count": len(rows), "results": rows}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate/read perception outputs and resolve a FreeGrasp query to a perception object id.",
    )
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--summary", type=Path, default=None, help="Perception summary.json. Defaults to data/, then sample_data/.")
    parser.add_argument("--instruction", default=None, help="Override FreeGrasp annotation.")
    parser.add_argument("--use-sample", action="store_true", help="Run all sample_data/scene_*/perception summaries and write scene_<id>/intent/id.txt.")
    parser.add_argument("--use", choices=["sample"], default=None, help="Alias for --use-sample when set to sample.")
    parser.add_argument("--api-key-env", default=RUN_INTENT_API_KEY_ENV)
    parser.add_argument("--base-url", default=RUN_INTENT_BASE_URL)
    parser.add_argument("--model", default=RUN_INTENT_MODEL)
    parser.add_argument("--timeout", type=float, default=RUN_INTENT_TIMEOUT)
    parser.add_argument(
        "--mode",
        choices=["data", "sample"],
        default="data",
        help="Use data/scene_*/perception by default, or sample_data with --mode sample.",
    )
    args = parser.parse_args()

    if not os.environ.get(args.api_key_env) and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"Missing API key. Export {args.api_key_env}=... before running python -m intent.run_intent."
        )

    if args.use_sample or args.use == "sample" or args.mode == "sample":
        return _run_sample_mode(args)
    return _run_data_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
