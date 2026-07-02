#!/usr/bin/env python
"""Visualize SmartGrasp PLY files with both RGB points and gripper meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def parse_args() -> argparse.Namespace:
    default_path = Path(__file__).resolve().parents[1] / "result" / "grasp_candidates.ply"
    parser = argparse.ArgumentParser(description="Open a SmartGrasp PLY with point cloud and grasp gripper meshes.")
    parser.add_argument("ply_path", nargs="?", default=str(default_path), help="PLY file to visualize.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ply_path = Path(args.ply_path).expanduser().resolve()
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
