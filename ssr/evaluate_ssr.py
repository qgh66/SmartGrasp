#!/usr/bin/env python3
"""Compute SSR (Segmentation Success Rate) — 连续 IoU 均值.

对每个 scene × split：
  1. reason grasp_id 的 perception mask vs GT mask → IoU
  2. SSR = 所有有效 split 的 IoU 均值

Usage: python ssr/evaluate_ssr.py easy easy-ambi ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS = ["split0", "split1", "split2"]


def load_ground_truth() -> dict[int, dict]:
    """Return {scene_id: {query_obj_id, groundTruthObjIds, annotation: {split: str}}}."""
    df = pd.concat([
        pd.read_parquet(DATA_DIR / "train-00000-of-00002.parquet"),
        pd.read_parquet(DATA_DIR / "train-00001-of-00002.parquet"),
    ], ignore_index=True)

    gt_map = {}
    for sid in sorted(df["sceneId"].unique()):
        sdf = df[df["sceneId"] == sid]
        first_qid = int(sdf["queryObjId"].iloc[0])
        qdf = sdf[sdf["queryObjId"] == first_qid]
        row0 = qdf.iloc[0]

        raw = str(row0["groundTruthObjIds"])
        # groundTruthObjIds 是 0-based，GT mask 文件是 1-based，需要 +1
        gt_ids = [int(x.strip()) + 1 for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]

        annotations = {}
        for _, r in qdf.iterrows():
            annotations[str(int(r["split"]))] = str(r["annotation"])

        gt_map[int(sid)] = {
            "query_obj_id": first_qid,
            "groundTruthObjIds": gt_ids,
            "annotations": annotations,
        }
    return gt_map


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return (np.array(img) > 127).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def find_perception_mask(mask_dir: Path, object_id: int) -> Path | None:
    """Find perception mask for a given object_id (NNN_anchor_*.png)."""
    prefix = f"{object_id:03d}_"
    candidates = sorted(mask_dir.glob(f"{prefix}*.png"))
    return candidates[0] if candidates else None


def find_gt_mask(gt_mask_dir: Path, object_id: int) -> Path | None:
    """Find GT mask for a given object_id (mask_NNN_gt.png or *NNN*.png)."""
    path = gt_mask_dir / f"mask_{object_id:03d}_gt.png"
    if path.exists():
        return path
    candidates = sorted(gt_mask_dir.glob(f"*{object_id:03d}*.png"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="SSR — FreeGrasp definition")
    parser.add_argument("categories", nargs="+", help="Categories (easy, easy-ambi, ...)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    gt_map = load_ground_truth()

    for cat in args.categories:
        cat_dir = DATA_DIR / cat
        if not cat_dir.exists():
            print(f"[SKIP] {cat}: not found")
            continue

        scenes = sorted(d for d in cat_dir.iterdir() if d.is_dir() and d.name.startswith("scene_"))
        if not scenes:
            print(f"[SKIP] {cat}: no scenes")
            continue

        total = 0
        iou_sum = 0.0
        iou_list: list[float] = []
        details = []

        for scene_dir in scenes:
            sid = int(scene_dir.name.replace("scene_", ""))
            gt_info = gt_map.get(sid)
            if not gt_info or not gt_info["groundTruthObjIds"]:
                continue

            gt_obj_ids = gt_info["groundTruthObjIds"]
            # 加载所有 groundTruthObjIds 的 GT masks
            gt_masks: dict[int, np.ndarray] = {}
            for gid in gt_obj_ids:
                p = find_gt_mask(scene_dir / "gt" / "mask", gid)
                if p:
                    m = load_mask(p)
                    if m is not None:
                        gt_masks[gid] = m

            if not gt_masks:
                continue

            mask_dir = scene_dir / "perception" / "mask"

            for split_name in SPLITS:
                reason_csv = scene_dir / "reason" / split_name / "results.csv"
                if not reason_csv.exists():
                    total += 1
                    iou_list.append(0.0)
                    details.append({"scene_id": sid, "split": split_name, "grasp_id": None, "gt_object_ids": gt_obj_ids, "best_gt_id": -1, "iou": 0.0, "error": "reason_csv missing"})
                    continue

                try:
                    df = pd.read_csv(reason_csv)
                except Exception:
                    total += 1
                    iou_list.append(0.0)
                    details.append({"scene_id": sid, "split": split_name, "grasp_id": None, "gt_object_ids": gt_obj_ids, "best_gt_id": -1, "iou": 0.0, "error": "csv read error"})
                    continue
                if df.empty or "grasp_id" not in df.columns:
                    total += 1
                    iou_list.append(0.0)
                    details.append({"scene_id": sid, "split": split_name, "grasp_id": None, "gt_object_ids": gt_obj_ids, "best_gt_id": -1, "iou": 0.0, "error": "no grasp_id column"})
                    continue

                grasp_id = df["grasp_id"].iloc[0]
                if pd.isna(grasp_id):
                    total += 1
                    iou_list.append(0.0)
                    details.append({"scene_id": sid, "split": split_name, "grasp_id": None, "gt_object_ids": gt_obj_ids, "best_gt_id": -1, "iou": 0.0, "error": "intent_no_target"})
                    continue
                grasp_id = int(grasp_id)

                per_mask_path = find_perception_mask(mask_dir, grasp_id)
                if per_mask_path is None:
                    total += 1
                    iou_list.append(0.0)
                    details.append({"scene_id": sid, "split": split_name, "grasp_id": grasp_id, "gt_object_ids": gt_obj_ids, "best_gt_id": -1, "iou": 0.0, "error": "per_mask not found"})
                    continue
                per_mask = load_mask(per_mask_path)
                if per_mask is None:
                    total += 1
                    iou_list.append(0.0)
                    details.append({"scene_id": sid, "split": split_name, "grasp_id": grasp_id, "gt_object_ids": gt_obj_ids, "best_gt_id": -1, "iou": 0.0, "error": "per_mask load fail"})
                    continue

                # grasp_id 的 perception mask vs 所有 groundTruthObjIds 的 GT masks，取 max IoU
                best_iou = 0.0
                best_gt_id = -1
                for gid, gm in gt_masks.items():
                    i = iou(per_mask, gm)
                    if i > best_iou:
                        best_iou = i
                        best_gt_id = gid

                success = best_iou >= 0.5
                total += 1
                iou_sum += best_iou
                iou_list.append(best_iou)

                details.append({
                    "scene_id": sid,
                    "split": split_name,
                    "grasp_id": grasp_id,
                    "gt_object_ids": gt_obj_ids,
                    "best_gt_id": best_gt_id,
                    "iou": round(best_iou, 4),
                })

                if args.verbose:
                    print(f"  scene_{sid} {split_name}: grasp={grasp_id} → best_gt={best_gt_id} IoU={best_iou:.4f}")

        ssr = iou_sum / total if total > 0 else 0.0
        print(f"\n{'='*50}")
        print(f"  {cat}: SSR = {ssr:.4f}  (mean IoU, {total} splits)")
        print(f"{'='*50}")

        out_dir = ROOT / "ssr" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cat}_ssr.json"
        out_path.write_text(json.dumps({
            "category": cat,
            "total_splits": total,
            "ssr_mean_iou": round(ssr, 4),
            "details": details,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
