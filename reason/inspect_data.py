"""Inspect a FreeGraspData sample by (scene_id, query_obj_id)."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


DATA_ROOT = Path("./data")


def inspect(scene_id: int, query_obj_id: int):
    # 1. Read parquet for annotation/image
    df = pd.concat([
        pd.read_parquet(DATA_ROOT / "train-00000-of-00002.parquet"),
        pd.read_parquet(DATA_ROOT / "train-00001-of-00002.parquet"),
    ], ignore_index=True)

    row = df[(df["sceneId"].astype(int) == scene_id) &
             (df["queryObjId"].astype(int) == query_obj_id)]
    if row.empty:
        print(f"No sample for scene={scene_id}, query={query_obj_id}")
        return
    row = row.iloc[0]

    print("=" * 60)
    print(f"scene_id           : {row['sceneId']}")
    print(f"queryObjId         : {row['queryObjId']}")
    print(f"annotation         : {row['annotation']}")
    print(f"groundTruthObjIds  : {row['groundTruthObjIds']}")
    print(f"difficulty         : {row['difficulty']}")
    print(f"ambiguious         : {row['ambiguious']}")
    print(f"split              : {row['split']}")
    print("=" * 60)

    # 2. Save RGB image to disk
    img = Image.open(io.BytesIO(row["image"]["bytes"]))
    rgb_path = Path(f"inspect_scene{scene_id}_query{query_obj_id}_rgb.png")
    img.save(rgb_path)
    print(f"RGB saved to: {rgb_path}")

    # 3. Load and inspect npz
    npz_path = DATA_ROOT / "npz_file" / f"{scene_id}.npz"
    if not npz_path.exists():
        print(f"NPZ not found: {npz_path}")
        return

    npz = np.load(npz_path, allow_pickle=True)
    print(f"\nNPZ keys: {list(npz.files)}")

    instances = np.asarray(npz["instances_objects"])
    print(f"\ninstances_objects shape: {instances.shape}")
    print(f"unique values in instances_objects:")
    for uid in np.unique(instances):
        pct = np.count_nonzero(instances == uid) / instances.size * 100
        print(f"  id={int(uid):3d}  pixels={np.count_nonzero(instances == uid):>8d}  ({pct:5.2f}%)")

    # 4. Check if query_obj_id appears in instances
    print(f"\nIs query_obj_id={query_obj_id} present in instances? "
          f"{query_obj_id in np.unique(instances)}")

    # 5. Visualize: RGB + instance map side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img)
    axes[0].set_title(f"RGB - {row['annotation']}")
    axes[0].axis("off")

    axes[1].imshow(instances, cmap="tab20")
    axes[1].set_title("instances_objects (each color = one object id)")
    axes[1].axis("off")

    viz_path = Path(f"inspect_scene{scene_id}_query{query_obj_id}.png")
    plt.savefig(viz_path, dpi=120, bbox_inches="tight")
    print(f"\nVisualization saved to: {viz_path}")
    plt.close()


if __name__ == "__main__":
    # Default to the suspicious sample
    scene = int(sys.argv[1]) if len(sys.argv) > 1 else 398
    query = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    inspect(scene, query)