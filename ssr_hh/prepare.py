#!/usr/bin/env python3
"""Generate task list for all 291 scenes with first query + 3 annotation splits.

Output: ssr_hh/tasks.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SSR_DIR = ROOT / "ssr_hh"
SSR_DIR.mkdir(parents=True, exist_ok=True)


def category_name(difficulty: str, ambiguious: bool) -> str:
    base = difficulty.lower()
    return f"{base}-ambi" if ambiguious else base


def main() -> None:
    df0 = pd.read_parquet(DATA_DIR / "train-00000-of-00002.parquet")
    df1 = pd.read_parquet(DATA_DIR / "train-00001-of-00002.parquet")
    df = pd.concat([df0, df1], ignore_index=True)

    tasks = []
    for sid in sorted(df["sceneId"].unique()):
        scene_df = df[df["sceneId"] == sid]
        first_qid = int(scene_df["queryObjId"].iloc[0])
        first_row = scene_df[scene_df["queryObjId"] == first_qid].iloc[0]

        qdf = scene_df[scene_df["queryObjId"] == first_qid]
        annotations: dict[int, str] = {}
        for _, row in qdf.iterrows():
            annotations[int(row["split"])] = str(row["annotation"])

        if len(annotations) != 3:
            print(f"  WARN scene={sid} query={first_qid}: {len(annotations)} splits")

        tasks.append({
            "scene_id": int(sid),
            "query_obj_id": first_qid,
            "difficulty": str(first_row["difficulty"]),
            "ambiguious": bool(first_row["ambiguious"]),
            "category": category_name(str(first_row["difficulty"]), bool(first_row["ambiguious"])),
            "annotations": {
                "0": annotations.get(0, ""),
                "1": annotations.get(1, ""),
                "2": annotations.get(2, ""),
            },
        })

    # Stats
    from collections import Counter
    cat_counts = Counter(t["category"] for t in tasks)
    print(f"Total tasks: {len(tasks)}")
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]}")

    # Write tasks.json
    out_path = SSR_DIR / "tasks.json"
    out_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")

    # Write scene-id lists per category (for batch perception runs)
    lists_dir = SSR_DIR / "scene_lists"
    lists_dir.mkdir(exist_ok=True)
    for cat in sorted(cat_counts):
        ids = [t["scene_id"] for t in tasks if t["category"] == cat]
        (lists_dir / f"{cat}.txt").write_text(" ".join(str(x) for x in ids))
        print(f"  {cat}: {len(ids)} ids -> {lists_dir / f'{cat}.txt'}")


if __name__ == "__main__":
    main()
