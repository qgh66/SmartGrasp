#!/usr/bin/env python3
"""Compute RSR (Reasoning Success Rate) — 二值 IoU ≥ 0.5.

对每个 scene × split：
  1. reason grasp_id 的 perception mask vs GT mask → IoU
  2. IoU >= 0.5 → 成功
  3. RSR = 成功数 / 总数

Usage: python rsr_hh/evaluate_rsr_hh.py easy easy-ambi ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssr_hh.evaluate_ssr import (
    load_ground_truth, load_mask, iou,
    find_perception_mask, find_gt_mask,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IOU_THRESHOLD = 0.5
SPLITS = ["split0", "split1", "split2"]
RSR_DIR = ROOT / "rsr_hh"
RESULTS_DIR = RSR_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="RSR — binary IoU ≥ 0.5")
    parser.add_argument("categories", nargs="+")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    gt_map = load_ground_truth()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        ok = 0
        details = []

        for scene_dir in scenes:
            sid = int(scene_dir.name.replace("scene_", ""))
            gt_info = gt_map.get(sid)
            if not gt_info or not gt_info["groundTruthObjIds"]:
                continue

            gt_obj_ids = gt_info["groundTruthObjIds"]
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
                    details.append({"scene_id": sid, "split": split_name, "success": False, "error": "reason_csv missing"})
                    continue

                try:
                    df = pd.read_csv(reason_csv)
                except Exception:
                    total += 1
                    details.append({"scene_id": sid, "split": split_name, "success": False, "error": "csv read error"})
                    continue
                if df.empty or "grasp_id" not in df.columns:
                    total += 1
                    details.append({"scene_id": sid, "split": split_name, "success": False, "error": "no grasp_id column"})
                    continue

                grasp_id = df["grasp_id"].iloc[0]
                if pd.isna(grasp_id):
                    total += 1
                    details.append({"scene_id": sid, "split": split_name, "success": False, "error": "intent_no_target"})
                    continue
                grasp_id = int(grasp_id)

                per_mask_path = find_perception_mask(mask_dir, grasp_id)
                if per_mask_path is None:
                    total += 1
                    details.append({"scene_id": sid, "split": split_name, "success": False, "error": "per_mask not found"})
                    continue
                per_mask = load_mask(per_mask_path)
                if per_mask is None:
                    total += 1
                    details.append({"scene_id": sid, "split": split_name, "success": False, "error": "per_mask load fail"})
                    continue

                best_iou = 0.0
                best_gt_id = -1
                for gid, gm in gt_masks.items():
                    i = iou(per_mask, gm)
                    if i > best_iou:
                        best_iou = i
                        best_gt_id = gid

                success = best_iou >= IOU_THRESHOLD
                total += 1
                if success:
                    ok += 1

                details.append({
                    "scene_id": sid,
                    "split": split_name,
                    "success": success,
                    "iou": round(best_iou, 4),
                })

                if args.verbose:
                    status = "✓" if success else "✗"
                    print(f"  {status} scene_{sid} {split_name}: IoU={best_iou:.4f}")

        rsr = ok / total if total > 0 else 0.0
        print(f"\n{'='*50}")
        print(f"  {cat}: RSR = {ok}/{total} = {rsr:.4f}  (IoU ≥ {IOU_THRESHOLD})")
        print(f"{'='*50}")

        out_path = RESULTS_DIR / f"{cat}_rsr_hh.json"
        out_path.write_text(json.dumps({
            "category": cat,
            "iou_threshold": IOU_THRESHOLD,
            "total": total,
            "success": ok,
            "rsr": round(rsr, 4),
            "details": details,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
