#!/usr/bin/env python
"""Visualize SmartGrasp PLY files with both RGB points and gripper meshes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import open3d as o3d


SMARTGRASP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = SMARTGRASP_ROOT / "data_realworld"
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def _candidate_sort_key(ply_path: Path, data_root: Path) -> tuple[str, int, int, str]:
    """Sort candidates by session timestamp, round, mtime, and path."""
    try:
        relative_parts = ply_path.relative_to(data_root).parts
    except ValueError:
        relative_parts = ply_path.parts

    timestamp = max(
        (part for part in relative_parts if TIMESTAMP_PATTERN.fullmatch(part)),
        default="",
    )
    round_index = max(
        (int(part) for part in relative_parts if part.isdigit()),
        default=-1,
    )
    return timestamp, round_index, ply_path.stat().st_mtime_ns, str(ply_path)


def find_latest_grasp_candidates_ply(data_root: Path) -> Path:
    """Return the newest timestamped grasp-candidate PLY under data_realworld."""
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"SmartGrasp data directory not found: {data_root}")

    candidates = [path for path in data_root.rglob("grasp_candidates.ply") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No grasp_candidates.ply found under: {data_root}")
    return max(candidates, key=lambda path: _candidate_sort_key(path, data_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a SmartGrasp PLY with point cloud and grasp gripper meshes.")
    parser.add_argument(
        "ply_path",
        nargs="?",
        default=None,
        help=(
            "PLY file to visualize. When omitted, automatically use the newest "
            "timestamped data_realworld/**/grasp_candidates.ply."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root searched for the latest candidate PLY (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ply_path is None:
        ply_path = find_latest_grasp_candidates_ply(args.data_root)
        print(f"[visualize-ply] latest candidate PLY: {ply_path}")
    else:
        ply_path = Path(args.ply_path).expanduser().resolve()
        print(f"[visualize-ply] requested PLY: {ply_path}")
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    geometries = []

    if point_cloud.has_points():
        geometries.append(point_cloud)
        print(f"[visualize-ply] point cloud: points={len(point_cloud.points)} colors={point_cloud.has_colors()}")

    if mesh.has_triangles():
        mesh.compute_vertex_normals()
        geometries.append(mesh)
        print(f"[visualize-ply] mesh: vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
    else:
        print("[visualize-ply] mesh: no triangles found; showing point cloud only")

    if not geometries:
        raise RuntimeError(f"No drawable geometry loaded from {ply_path}")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=f"SmartGrasp PLY: {ply_path.name}", width=1280, height=800)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    render_option = visualizer.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.background_color = [1.0, 1.0, 1.0]
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
