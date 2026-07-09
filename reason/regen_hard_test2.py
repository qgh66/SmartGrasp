"""Regenerate only Hard/test_2 with fixed ratio + arrows."""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from PIL import Image

from reason.data_loader import load_sample
from reason.branch_judge.classifier import classify_branch


DATA_ROOT = "./data"
OUT_DIR = Path("./testcase/Hard/test_2")
TARGET_INDEX = 2   # 0-based index within Hard subset


def visualize_occlusion_graph(perception, save_path: Path, target_mid: int):
    g = perception.occlusion_graph
    target_node = perception.molmo_to_node.get(target_mid)

    fig, ax = plt.subplots(figsize=(8, 7))
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    else:
        pos = nx.spring_layout(g, seed=42, k=1.5)
        node_size = 1800

        node_colors = [
            "#ff6b6b" if n == target_node else "#8ecae6"
            for n in g.nodes()
        ]
        labels = {
            n: f"{n}\n(mid={perception.node_info[n]['molmo_id']})"
            for n in g.nodes()
        }

        nx.draw_networkx_nodes(g, pos, node_color=node_colors,
                               node_size=node_size, ax=ax,
                               edgecolors="black", linewidths=1.2)
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=9, ax=ax)

        nx.draw_networkx_edges(
            g, pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=22,
            width=2.0,
            edge_color="#1f4e79",
            node_size=node_size,
            connectionstyle="arc3,rad=0.08",
            ax=ax,
        )

        edge_labels = {
            (u, v): f"{d.get('ratio', 0):.1%}"
            for u, v, d in g.edges(data=True)
        }
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=8, ax=ax,
                                     bbox=dict(boxstyle="round,pad=0.2",
                                               facecolor="white",
                                               edgecolor="none",
                                               alpha=0.8))

    ax.set_title(f"Occlusion Graph (target=red, mid={target_mid})\n"
                 f"arrow A→B = A occludes B")
    ax.axis("off")
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    df = pd.concat([
        pd.read_parquet(f"{DATA_ROOT}/train-00000-of-00002.parquet"),
        pd.read_parquet(f"{DATA_ROOT}/train-00001-of-00002.parquet"),
    ], ignore_index=True)

    hard_df = df[df["difficulty"] == "Hard"].reset_index(drop=True)
    row = hard_df.iloc[TARGET_INDEX]

    scene_id = int(row["sceneId"])
    raw_query = int(row["queryObjId"])

    print(f"Regenerating Hard/test_{TARGET_INDEX}")
    print(f"  scene_id={scene_id}, queryObjId={raw_query}")
    print(f"  annotation: {row['annotation']}")

    perception = load_sample(
        data_root=DATA_ROOT,
        scene_id=scene_id,
        query_obj_id=raw_query,
        occlusion_threshold=0.01,
        auto_download=False,
    )
    branch, reason = classify_branch(perception)
    print(f"  branch: {branch.value}")
    print(f"  reason: {reason}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save RGB
    img = Image.open(io.BytesIO(row["image"]["bytes"]))
    img.save(OUT_DIR / "rgb.png")

    # Save graph
    visualize_occlusion_graph(
        perception, OUT_DIR / "occlusion_graph.png",
        target_mid=perception.target_molmo_id,
    )

    # Save info.json
    info = {
        "row_idx": int(hard_df.index[TARGET_INDEX])
                   if hasattr(hard_df, "index") else TARGET_INDEX,
        "scene_id": scene_id,
        "queryObjId_raw": raw_query,
        "npz_target_id": raw_query + 1,
        "annotation": str(row["annotation"]),
        "groundTruthObjIds": str(row["groundTruthObjIds"]),
        "difficulty": "Hard",
        "ambiguous": bool(row["ambiguious"]),
        "split": int(row["split"]),
        "branch": branch.value,
        "reason": reason,
        "status": "ok",
    }
    with open(OUT_DIR / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # Also print all edge ratios for inspection
    print("\nEdges in graph:")
    g = perception.occlusion_graph
    for u, v, d in g.edges(data=True):
        u_mid = perception.node_info[u]["molmo_id"]
        v_mid = perception.node_info[v]["molmo_id"]
        print(f"  mid={u_mid} -> mid={v_mid}, "
              f"ratio={d.get('ratio'):.2%}, "
              f"pixels={d.get('overlap_pixels')}")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()