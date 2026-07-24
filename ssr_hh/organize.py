#!/usr/bin/env python3
"""Organize results from data/scene_X/ into data/{category}/scene_X/ structure.

Assumes the following have been generated for each scene:
  data/scene_X/gt/                    (GT perception, once)
  data/scene_X/perception/            (VLM perception, once)
  data/scene_X/intent_split0/         (intent for split 0)
  data/scene_X/intent_split1/         (intent for split 1)
  data/scene_X/intent_split2/         (intent for split 2)
  data/scene_X/reason_split0/         (reason for split 0)
  data/scene_X/reason_split1/         (reason for split 1)
  data/scene_X/reason_split2/         (reason for split 2)

Usage:
  python ssr_hh/organize.py          # dry-run, shows what would be moved
  python ssr_hh/organize.py --run    # actually moves files
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SSR_DIR = ROOT / "ssr_hh"
SPLITS = ["split0", "split1", "split2"]


def load_tasks() -> list[dict]:
    return json.loads((SSR_DIR / "tasks.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize results into category dirs")
    parser.add_argument("--run", action="store_true", help="Actually move files")
    parser.add_argument("--category", default=None, help="Only process one category")
    parser.add_argument("--scene", type=int, default=None, help="Only process one scene")
    args = parser.parse_args()

    tasks = load_tasks()
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]
    if args.scene is not None:
        tasks = [t for t in tasks if t["scene_id"] == args.scene]

    dry = not args.run
    if dry:
        print("=== DRY RUN (use --run to actually move) ===\n")

    success = 0
    skipped = 0
    errors = 0

    for t in tasks:
        sid = t["scene_id"]
        cat = t["category"]
        src_dir = DATA_DIR / f"scene_{sid}"
        dst_dir = DATA_DIR / cat / f"scene_{sid}"

        # ---- gt (once per scene) ----
        src_gt = src_dir / "gt"
        dst_gt = dst_dir / "gt"
        if src_gt.exists():
            if dry:
                print(f"  [gt]    scene_{sid} -> {cat}/scene_{sid}/gt")
            else:
                dst_gt.parent.mkdir(parents=True, exist_ok=True)
                if dst_gt.exists():
                    shutil.rmtree(dst_gt)
                shutil.move(str(src_gt), str(dst_gt))
        else:
            print(f"  [gt]    scene_{sid}: MISSING {src_gt}")
            errors += 1

        # ---- perception (once per scene) ----
        src_per = src_dir / "perception"
        dst_per = dst_dir / "perception"
        if src_per.exists():
            if dry:
                print(f"  [per]   scene_{sid} -> {cat}/scene_{sid}/perception")
            else:
                dst_per.parent.mkdir(parents=True, exist_ok=True)
                if dst_per.exists():
                    shutil.rmtree(dst_per)
                shutil.move(str(src_per), str(dst_per))
        else:
            print(f"  [per]   scene_{sid}: MISSING {src_per}")
            errors += 1

        # ---- intent splits ----
        for split in SPLITS:
            src = src_dir / f"intent_{split}"
            dst = dst_dir / "intent" / split
            if src.exists():
                if dry:
                    print(f"  [intent] scene_{sid}/{split}")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))
            else:
                print(f"  [intent] scene_{sid}/{split}: MISSING")
                errors += 1

        # ---- reason splits ----
        for split in SPLITS:
            src = src_dir / f"reason_{split}"
            dst = dst_dir / "reason" / split
            if src.exists():
                if dry:
                    print(f"  [reason] scene_{sid}/{split}")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))
            else:
                print(f"  [reason] scene_{sid}/{split}: MISSING")
                errors += 1

        success += 1

        # Clean up empty scene dir
        if not dry and src_dir.exists():
            remaining = list(src_dir.iterdir())
            if not remaining:
                src_dir.rmdir()

    print(f"\nDone: {success} scenes organized, {errors} errors, {skipped} skipped")


if __name__ == "__main__":
    main()
