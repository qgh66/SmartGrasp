#!/usr/bin/env python3
"""Plot IoU distribution per category."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ssr_hh" / "results"
OUT = ROOT / "ssr_hh" / "plots"
OUT.mkdir(exist_ok=True)

CATEGORIES = ["easy", "easy-ambi", "medium", "medium-ambi", "hard", "hard-ambi"]
COLORS = ["#2ecc71", "#27ae60", "#3498db", "#2980b9", "#e74c3c", "#c0392b"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, cat in enumerate(CATEGORIES):
    path = RESULTS / f"{cat}_ssr_hh.json"
    if not path.exists():
        continue
    data = json.load(open(path))
    ious = [d["iou"] for d in data["details"]]
    
    ax = axes[i]
    ax.hist(ious, bins=20, range=(0, 1), color=COLORS[i], edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(ious), color="black", linestyle="--", label=f"mean={np.mean(ious):.3f}")
    ax.set_title(f"{cat} (n={len(ious)})", fontsize=13)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

fig.suptitle("SSR IoU Distribution by Category", fontsize=16, y=1.01)
plt.tight_layout()
out_path = OUT / "iou_distribution.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
