from __future__ import annotations

import shutil
from pathlib import Path

from datasets import DownloadConfig, load_dataset_builder


REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "freegrasp"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    builder = load_dataset_builder(
        "FBK-TeV/FreeGraspData",
        download_config=DownloadConfig(resume_download=True),
    )
    builder.download_and_prepare()

    cache_root = Path(builder.cache_dir)
    print(f"builder cache_dir: {cache_root}")
    print(f"export dir: {OUT_DIR}")

    copied = 0

    # Export parquet shards and README from the HF hub snapshot cache.
    snapshot_roots = []
    hub_root = cache_root.parents[4] / "datasets--FBK-TeV--FreeGraspData" / "snapshots"
    if hub_root.exists():
        snapshot_roots.extend(sorted(p for p in hub_root.iterdir() if p.is_dir()))
    if cache_root.exists():
        snapshot_roots.append(cache_root)

    seen = set()
    for root in snapshot_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix != ".parquet" and path.name.upper() != "README.MD":
                continue
            if path in seen:
                continue
            seen.add(path)
            target = OUT_DIR / path.name
            if not target.exists():
                shutil.copy2(path, target)
                copied += 1
                print(f"copied: {target}")

    print(f"done, copied {copied} files into {OUT_DIR}")


if __name__ == "__main__":
    main()
