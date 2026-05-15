from __future__ import annotations

import argparse
import os
import sys

from .annotator import MolmoAnnotator


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Molmo annotate: labeled png + points JSON (pixel coords).")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--prompt", required=True, help="Text prompt for Molmo")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--model-id", default="allenai/Molmo-7B-D-0924", help="HF model id")
    args = p.parse_args(argv)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    ann = MolmoAnnotator(model_id=args.model_id)
    res = ann.annotate_to_folder(args.image, args.prompt, out_dir, return_base64=False)

    print(res["labeled_png_path"])
    print(res["json_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())