#!/usr/bin/env python3
"""Generate task list for all 300 (scene, query) pairs with 3 annotation splits.

Output: ssr/tasks.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SSR_DIR = ROOT / "ssr"
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
        # 取第一个出现的 query 作为 primary（与原逻辑一致）
        first_qid = int(scene_df["queryObjId"].iloc[0])
        qids = sorted(scene_df["queryObjId"].unique())

        for qid in qids:
            qdf = scene_df[scene_df["queryObjId"] == qid]
            row0 = qdf.iloc[0]

            annotations: dict[int, str] = {}
            for _, row in qdf.iterrows():
                annotations[int(row["split"])] = str(row["annotation"])

            if len(annotations) != 3:
                print(f"  WARN scene={sid} query={qid}: {len(annotations)} splits")
                continue

            cat = category_name(str(row0["difficulty"]), bool(row0["ambiguious"]))
            # 多 query 场景中，第一个 query 用 scene_{sid}，其余用 scene_{sid}_q{qid}
            if qid == first_qid:
                dir_name = f"scene_{sid}"
            else:
                dir_name = f"scene_{sid}_q{qid}"

            tasks.append({
                "scene_id": int(sid),
                "query_obj_id": int(qid),
                "difficulty": str(row0["difficulty"]),
                "ambiguious": bool(row0["ambiguious"]),
                "category": cat,
                "directory_name": dir_name,
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
