from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image


def draw_labeled_image_matplotlib(
    image: Image.Image,
    points_with_ids: List[Tuple[int, int, int]],
    out_png_path: str,
) -> str:
    """
    Keep the same visual style as the original project:
    - plt.text at each (x, y)
    - yellow bold text
    - black semi-transparent bbox
    """
    out_dir = os.path.dirname(out_png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    for obj_id, x, y in points_with_ids:
        plt.text(
            x, y, obj_id,
            color="yellow", fontsize=8, fontweight="bold",
            ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )

    # original code sometimes uses title; it's optional for exact look
    plt.axis("off")
    plt.savefig(out_png_path, bbox_inches="tight", dpi=300)
    plt.close()
    return out_png_path