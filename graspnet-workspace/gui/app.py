#!/usr/bin/env python
"""Dash web GUI for GraspNet simulation results."""

from __future__ import annotations

import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


ROOT = Path(__file__).resolve().parents[1]
OBJECT_Z_THRESHOLD = 0.005
TABLE_Z = 0.0
TABLE_CLEARANCE = 0.005
MAX_GRASP_CENTER_DIST = 0.04
APPROACH_DEPTH_OFFSET = 0.05

# 全局缓存简化 mesh
_MESH_CACHE = {}


def _simplify_vertex_clustering(verts: np.ndarray, faces: np.ndarray, voxel_size: float):
    """无需外部依赖的顶点聚类简化，保持拓扑大致完好。"""
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces
    vmin = verts.min(axis=0)
    indices = np.floor((verts - vmin) / voxel_size).astype(np.int64)
    voxel_map = {}
    new_verts = []
    new_indices = np.empty(len(verts), dtype=np.int64)
    for i, idx in enumerate(indices):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        if key not in voxel_map:
            voxel_map[key] = len(new_verts)
            new_verts.append(verts[i])
        new_indices[i] = voxel_map[key]
    new_verts = np.array(new_verts, dtype=float)
    new_faces = new_indices[faces]
    # 去除退化面（有重复顶点的面）
    mask = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    new_faces = new_faces[mask]
    # 去除重复面
    if len(new_faces):
        sorted_faces = np.sort(new_faces, axis=1)
        new_faces = np.unique(sorted_faces, axis=0)
    return new_verts, new_faces


def _load_simplified_mesh(obj_path: str, max_verts: int = 5000):
    """加载并拓扑保持地简化 .obj mesh 用于 Plotly 渲染。"""
    if obj_path in _MESH_CACHE:
        return _MESH_CACHE[obj_path]
    import trimesh
    mesh = trimesh.load(obj_path, force='mesh')
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if len(verts) > max_verts:
        # 二分搜索合适的 voxel_size，使顶点数降到 max_verts 附近
        lo, hi = 0.0001, float(mesh.extents.max())
        for _ in range(12):
            mid = (lo + hi) / 2.0
            nv, _ = _simplify_vertex_clustering(verts, faces, mid)
            if len(nv) > max_verts:
                lo = mid
            else:
                hi = mid
        verts, faces = _simplify_vertex_clustering(verts, faces, hi)
    _MESH_CACHE[obj_path] = (verts, faces)
    return verts, faces


def _rotation_matrix_from_orientation(orientation) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    orientation = np.asarray(orientation) if orientation is not None else None
    if orientation is not None and orientation.shape == (4,):
        return Rot.from_quat(orientation).as_matrix()
    elif orientation is not None and orientation.shape == (3, 3):
        return orientation
    return np.eye(3)


def _pca_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(centered.T))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return axes


def _estimate_mesh_orientation_from_points(obj_path: str, obj_pts: np.ndarray):
    if len(obj_pts) < 3:
        return np.eye(3)
    verts, _ = _load_simplified_mesh(obj_path)
    if len(verts) < 3:
        return np.eye(3)
    mesh_axes = _pca_axes(verts)
    object_axes = _pca_axes(obj_pts)
    rot_m = object_axes @ mesh_axes.T
    if np.linalg.det(rot_m) < 0:
        object_axes[:, 2] *= -1
        rot_m = object_axes @ mesh_axes.T
    return rot_m


def _render_mesh_trace(obj_path: str, position, orientation=None, color='#2563eb', opacity=0.85):
    """渲染真实 .obj mesh 作为 Plotly Mesh3d trace。"""
    verts, faces = _load_simplified_mesh(obj_path)
    rot_m = _rotation_matrix_from_orientation(orientation)
    world_verts = (rot_m @ verts.T).T + np.asarray(position, dtype=float)
    return go.Mesh3d(
        x=world_verts[:, 0], y=world_verts[:, 1], z=world_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, flatshading=True,
        name='object', hoverinfo='skip', showscale=False)


def _mesh_position_for_center(obj_path: str, desired_center: np.ndarray, orientation=None) -> np.ndarray:
    verts, _ = _load_simplified_mesh(obj_path)
    mesh_center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    rot_m = _rotation_matrix_from_orientation(orientation)
    return np.asarray(desired_center, dtype=float) - rot_m @ mesh_center


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _unit(vec, fallback):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=float)
    return vec / norm


def _object_points(pc: np.ndarray) -> np.ndarray:
    return pc[pc[:, 2] > OBJECT_Z_THRESHOLD]


def _nearest_object_dist(center: np.ndarray, obj_pts: np.ndarray) -> float:
    if len(obj_pts) == 0:
        return float("inf")
    return float(np.linalg.norm(obj_pts - center, axis=1).min())


def _grasp_approach_dir(rot: np.ndarray) -> np.ndarray:
    # GraspNet official mesh uses local x as depth / approach axis.
    return _unit(rot[:, 0], [0, 0, -1])


def _pre_grasp_pos(center: np.ndarray, rot: np.ndarray, depth: float) -> np.ndarray:
    return center - _grasp_approach_dir(rot) * (max(float(depth), 0.0) + APPROACH_DEPTH_OFFSET)


def _constraint_status(grasp: dict, obj_pts: np.ndarray) -> tuple[bool, list[str], dict]:
    center = np.asarray(grasp["translation"], dtype=float)
    rot = np.asarray(grasp["rotation"], dtype=float)
    pre = _pre_grasp_pos(center, rot, grasp.get("depth", 0.03))
    center_dist = _nearest_object_dist(center, obj_pts)
    min_path_z = min(float(center[2]), float(pre[2]))
    reasons = []
    if center_dist > MAX_GRASP_CENTER_DIST:
        reasons.append("center_not_on_object")
    if min_path_z < TABLE_Z + TABLE_CLEARANCE:
        reasons.append("approach_below_table")
    return not reasons, reasons, {
        "center_object_dist": center_dist,
        "approach_min_z": min_path_z,
    }


def _annotate_constraints(results: list[dict], point_cloud: np.ndarray) -> None:
    obj_pts = _object_points(point_cloud)
    for grasp in results:
        valid, reasons, metrics = _constraint_status(grasp, obj_pts)
        grasp["raw_success"] = bool(grasp.get("success", False))
        grasp["physical_valid"] = valid
        grasp["center_object_dist"] = metrics["center_object_dist"]
        grasp["approach_min_z"] = metrics["approach_min_z"]
        if reasons:
            grasp["failure_reason"] = ", ".join(reasons)
        else:
            grasp.pop("failure_reason", None)


def find_result_files() -> list[Path]:
    candidates: list[Path] = []
    for path in ROOT.rglob("*.json"):
        if path.name.startswith("."):
            continue
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list) or (isinstance(data, dict) and "grasps" in data):
            candidates.append(path)
    return sorted(candidates, key=lambda p: _rel(p))


def infer_viz_path(result_path: Path) -> Path | None:
    stem = result_path.with_suffix("")
    candidates = [
        result_path.with_name(f"{stem.name}_viz_data.pkl"),
        result_path.with_name("results_viz_data.pkl"),
        result_path.with_name("viz_data.pkl"),
        result_path.parent / "results_viz_data.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def normalize_results(data) -> tuple[list[dict], dict]:
    if isinstance(data, dict):
        grasps = data.get("grasps", [])
        meta = {k: v for k, v in data.items() if k != "grasps"}
    else:
        grasps = data
        meta = {"total": len(data), "success": sum(1 for r in data if r.get("success"))}
    normalized = []
    for idx, grasp in enumerate(grasps):
        g = dict(grasp)
        g.setdefault("grasp_index", idx)
        g["success"] = bool(g.get("success", False))
        g["score"] = float(g.get("score", 0.0))
        g["lift_z"] = float(g.get("lift_z", 0.0))
        g["width"] = float(g.get("width", 0.0))
        g["depth"] = float(g.get("depth", 0.0))
        g["translation"] = np.asarray(g.get("translation", [0, 0, 0]), dtype=float)
        g["rotation"] = np.asarray(g.get("rotation", np.eye(3)), dtype=float).reshape(3, 3)
        normalized.append(g)
    return normalized, meta


@lru_cache(maxsize=16)
def load_case(result_path_str: str, viz_path_str: str | None):
    result_path = _resolve(result_path_str)
    if result_path is None or not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path_str}")
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results, meta = normalize_results(data)

    viz_path = _resolve(viz_path_str) if viz_path_str else infer_viz_path(result_path)
    if viz_path is None or not viz_path.exists():
        raise FileNotFoundError(f"Visualization data not found for {result_path}")
    with open(viz_path, "rb") as f:
        viz_data = pickle.load(f)

    point_cloud = np.asarray(viz_data["point_cloud"], dtype=float)
    if point_cloud.ndim == 3:
        point_cloud = point_cloud[0]
    _annotate_constraints(results, point_cloud)
    rgb = np.asarray(viz_data.get("rgb"))
    depth = np.asarray(viz_data.get("depth"))
    trajectories = viz_data.get("grasp_trajectories", [])
    if "object_orientation" not in meta and "object_orientation" in viz_data:
        meta["object_orientation"] = viz_data["object_orientation"]

    return {
        "result_path": result_path,
        "viz_path": viz_path,
        "results": results,
        "meta": meta,
        "point_cloud": point_cloud,
        "rgb": rgb,
        "depth": depth,
        "trajectories": trajectories,
    }


def choose_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def add_points(fig: go.Figure, pc: np.ndarray, sample_count: int):
    obj = _object_points(pc)
    table = pc[pc[:, 2] <= OBJECT_Z_THRESHOLD]
    obj_count = min(max(500, sample_count // 3), len(obj)) if len(obj) else 0
    table_count = max(0, sample_count - obj_count)
    obj_show = choose_points(obj, obj_count) if obj_count else np.empty((0, 3))
    table_show = choose_points(table, min(table_count, len(table))) if len(table) else np.empty((0, 3))

    if len(table_show):
        fig.add_trace(go.Scatter3d(
            x=table_show[:, 0], y=table_show[:, 1], z=table_show[:, 2],
            mode="markers",
            marker={"size": 2, "color": "rgba(155, 164, 176, 0.32)"},
            name=f"table ({len(table_show)})",
            hoverinfo="skip",
        ))
    if len(obj_show):
        fig.add_trace(go.Scatter3d(
            x=obj_show[:, 0], y=obj_show[:, 1], z=obj_show[:, 2],
            mode="markers",
            marker={"size": 3, "color": "#2563eb", "opacity": 0.85},
            name=f"object ({len(obj)})",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>object</extra>",
        ))
    return obj


def add_axis(fig: go.Figure, origin: np.ndarray, rot: np.ndarray, length: float, prefix: str, width: int):
    colors = ["#dc2626", "#16a34a", "#2563eb"]
    names = ["x", "y", "z"]
    for i, color in enumerate(colors):
        end = origin + rot[:, i] * length
        fig.add_trace(go.Scatter3d(
            x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
            mode="lines",
            line={"color": color, "width": width},
            showlegend=False,
            hoverinfo="skip",
            name=f"{prefix} {names[i]}",
        ))


def box_vertices(center: np.ndarray, axes: np.ndarray, size: tuple[float, float, float]) -> np.ndarray:
    hx, hy, hz = np.asarray(size, dtype=float) / 2.0
    corners = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ])
    return center + corners @ axes.T


def box_mesh(center: np.ndarray, axes: np.ndarray, size: tuple[float, float, float],
             color: str, name: str, opacity: float = 0.72):
    vertices = box_vertices(center, axes, size)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ])
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        name=name,
        hoverinfo="skip",
        showscale=False,
    )


def box_edges(center: np.ndarray, axes: np.ndarray, size: tuple[float, float, float],
              color: str = "#0f172a", width: int = 3):
    vertices = box_vertices(center, axes, size)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [vertices[a, 0], vertices[b, 0], None]
        ys += [vertices[a, 1], vertices[b, 1], None]
        zs += [vertices[a, 2], vertices[b, 2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line={"color": color, "width": width},
        name="gripper edges",
        hoverinfo="skip",
        showlegend=False,
    )


def gripper_traces(position: np.ndarray, rot: np.ndarray, width: float, depth: float,
                   name: str = "gripper") -> list:
    # Match graspnetAPI.utils.utils.plot_gripper_pro_max:
    # local x = depth/approach, local y = jaw opening, local z = gripper height.
    opening = max(float(width), 0.002)
    depth = max(float(depth), 0.0)
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    def local_box(sx: float, sy: float, sz: float, offset: tuple[float, float, float]):
        vertices = np.array([
            [0, 0, 0], [sx, 0, 0], [0, 0, sz], [sx, 0, sz],
            [0, sy, 0], [sx, sy, 0], [0, sy, sz], [sx, sy, sz],
        ], dtype=float)
        vertices += np.asarray(offset, dtype=float)
        faces = np.array([
            [4, 7, 5], [4, 6, 7], [0, 2, 4], [2, 6, 4],
            [0, 1, 2], [1, 3, 2], [1, 5, 7], [1, 7, 3],
            [2, 3, 7], [2, 7, 6], [0, 4, 1], [1, 4, 5],
        ])
        return vertices, faces

    parts = [
        local_box(
            depth + depth_base + finger_width,
            finger_width,
            height,
            (-(depth_base + finger_width), -(opening / 2.0 + finger_width), -height / 2.0),
        ),
        local_box(
            depth + depth_base + finger_width,
            finger_width,
            height,
            (-(depth_base + finger_width), opening / 2.0, -height / 2.0),
        ),
        local_box(
            finger_width,
            opening,
            height,
            (-(finger_width + depth_base), -opening / 2.0, -height / 2.0),
        ),
        local_box(
            tail_length,
            finger_width,
            height,
            (-(tail_length + finger_width + depth_base), -finger_width / 2.0, -height / 2.0),
        ),
    ]

    vertices = []
    faces = []
    offset = 0
    for verts, tris in parts:
        vertices.append(verts)
        faces.append(tris + offset)
        offset += len(verts)
    local_vertices = np.vstack(vertices)
    triangles = np.vstack(faces)
    world_vertices = (rot @ local_vertices.T).T + position

    mesh = go.Mesh3d(
        x=world_vertices[:, 0], y=world_vertices[:, 1], z=world_vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color="#38bdf8",
        opacity=0.82,
        flatshading=True,
        name=name,
        hoverinfo="skip",
        showscale=False,
    )

    edge_pairs = set()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_pairs.add(tuple(sorted((int(a), int(b)))))
    xs, ys, zs = [], [], []
    for a, b in sorted(edge_pairs):
        xs += [world_vertices[a, 0], world_vertices[b, 0], None]
        ys += [world_vertices[a, 1], world_vertices[b, 1], None]
        zs += [world_vertices[a, 2], world_vertices[b, 2], None]
    edges = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line={"color": "#0f172a", "width": 3},
        name=f"{name} edges",
        hoverinfo="skip",
        showlegend=False,
    )

    approach_start = position - rot[:, 0] * (tail_length + depth_base + 0.02)
    approach_end = position + rot[:, 0] * max(depth, 0.025)
    approach = go.Scatter3d(
        x=[approach_start[0], approach_end[0]],
        y=[approach_start[1], approach_end[1]],
        z=[approach_start[2], approach_end[2]],
        mode="lines+markers",
        line={"color": "#f97316", "width": 6},
        marker={"size": [3, 6], "color": ["#f97316", "#f59e0b"]},
        name="grasp x/depth axis",
        hoverinfo="skip",
    )
    return [mesh, edges, approach]


def add_selected_gripper(fig: go.Figure, grasp: dict, obj_pts: np.ndarray):
    rot, center, _ = _constrained_grasp_pose(grasp, obj_pts)
    for trace in gripper_traces(center, rot, grasp.get("width", 0.06), grasp.get("depth", 0.03), "selected gripper"):
        fig.add_trace(trace)


def animation_traces(table_pts: np.ndarray, obj_pts: np.ndarray, gripper_pos: np.ndarray,
                     rot: np.ndarray, width: float, depth: float, obj_shift_z: float,
                     stage: str, obj_mesh_path: str | None = None, obj_mesh_pos=None,
                     obj_mesh_orn=None):
    traces = []
    if len(table_pts):
        traces.append(go.Scatter3d(
            x=table_pts[:, 0], y=table_pts[:, 1], z=table_pts[:, 2],
            mode="markers",
            marker={"size": 2, "color": "rgba(155, 164, 176, 0.25)"},
            name="table",
            hoverinfo="skip",
        ))
    else:
        traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="table"))

    if obj_mesh_path and Path(obj_mesh_path).exists() and obj_mesh_pos is not None:
        try:
            traces.append(_render_mesh_trace(obj_mesh_path, obj_mesh_pos, obj_mesh_orn, color="#2563eb"))
        except Exception as e:
            import traceback, sys
            print(f"[GUI] Synthetic mesh render failed: {e}", file=sys.stderr)
            traceback.print_exc()
            # fallback to point cloud
            shifted_obj = obj_pts.copy()
            shifted_obj[:, 2] += obj_shift_z
            if len(shifted_obj):
                traces.append(go.Scatter3d(
                    x=shifted_obj[:, 0], y=shifted_obj[:, 1], z=shifted_obj[:, 2],
                    mode="markers",
                    marker={"size": 3, "color": "#2563eb", "opacity": 0.85},
                    name="object",
                    hoverinfo="skip",
                ))
    else:
        shifted_obj = obj_pts.copy()
        shifted_obj[:, 2] += obj_shift_z
        if len(shifted_obj):
            traces.append(go.Scatter3d(
                x=shifted_obj[:, 0], y=shifted_obj[:, 1], z=shifted_obj[:, 2],
                mode="markers",
                marker={"size": 3, "color": "#2563eb", "opacity": 0.85},
                name="object",
                hoverinfo="skip",
            ))

    traces.extend(gripper_traces(gripper_pos, rot, width, depth))
    traces.append(go.Scatter3d(
        x=[gripper_pos[0]], y=[gripper_pos[1]], z=[gripper_pos[2]],
        mode="markers+text",
        marker={"size": 6, "color": "#f59e0b"},
        text=[stage],
        textposition="top center",
        name="stage",
        hoverinfo="skip",
    ))
    return traces


def make_animation_figure(case, selected_index: int | None, sample_count: int = 2500):
    results = case["results"]
    selected = next((r for r in results if r["grasp_index"] == selected_index), None)
    fig = go.Figure()
    if selected is None:
        fig.update_layout(
            title="Grasp Animation (select a grasp to view)",
            margin={"l": 0, "r": 0, "t": 32, "b": 0},
            annotations=[{"text": "Select a grasp", "showarrow": False, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
        )
        return fig

    pc = case["point_cloud"]
    obj_pts = _object_points(pc)
    table_pts = pc[pc[:, 2] <= OBJECT_Z_THRESHOLD]
    obj_pts = choose_points(obj_pts, min(max(800, sample_count), len(obj_pts))) if len(obj_pts) else np.empty((0, 3))
    table_pts = choose_points(table_pts, min(1200, len(table_pts))) if len(table_pts) else np.empty((0, 3))

    return _animation_synthetic(selected, obj_pts, table_pts, pc, case)


def filtered_results(results: list[dict], top_k: int, outcomes: list[str], score_min: float) -> list[dict]:
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]
    out = []
    for r in ranked:
        outcome = "success" if r["success"] else "failed"
        if outcome in outcomes and r["score"] >= score_min:
            out.append(r)
    return out


def make_3d_figure(case, top_k: int, outcomes: list[str], score_min: float, selected_index: int | None, sample_count: int):
    results = filtered_results(case["results"], top_k, outcomes, score_min)
    fig = go.Figure()
    obj_pts = add_points(fig, case["point_cloud"], sample_count)

    for r in results:
        color = "#16a34a" if r["success"] else "#dc2626"
        size = 8 if r["grasp_index"] == selected_index else 5
        fig.add_trace(go.Scatter3d(
            x=[r["translation"][0]], y=[r["translation"][1]], z=[r["translation"][2]],
            mode="markers",
            marker={"size": size, "color": color, "line": {"color": "#111827", "width": 1}},
            name=f"grasp {r['grasp_index']}",
            hovertemplate=(
                "grasp=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
            ),
            text=[f"{r['grasp_index']} score={r['score']:.3f}"],
        ))
        add_axis(fig, r["translation"], r["rotation"], 0.035, f"grasp {r['grasp_index']}", 5 if r["grasp_index"] == selected_index else 3)

    if len(obj_pts):
        mins = obj_pts.min(axis=0)
        maxs = obj_pts.max(axis=0)
    else:
        pc = case["point_cloud"]
        mins = pc.min(axis=0)
        maxs = pc.max(axis=0)
    center = (mins + maxs) / 2.0
    span = max(float(np.max(maxs - mins)), 0.12)
    pad = span * 0.75

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 26, "b": 0},
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        scene={
            "xaxis": {"title": "X (m)", "range": [center[0] - pad, center[0] + pad]},
            "yaxis": {"title": "Y (m)", "range": [center[1] - pad, center[1] + pad]},
            "zaxis": {"title": "Z (m)", "range": [max(0.0, center[2] - pad), center[2] + pad]},
            "aspectmode": "cube",
            "bgcolor": "#f8fafc",
        },
        title={"text": "Point Cloud and Grasp Poses", "x": 0.02, "font": {"size": 16}},
    )
    return fig


# ======================================================================
# 真实 PyBullet 轨迹动画 (frame_log 驱动)
# ======================================================================

def _animation_from_frame_log(frame_log, selected, obj_pts, table_pts, case=None):
    from scipy.spatial.transform import Rotation as Rot
    frames = []; bounds_all = []

    # 从 case 中读取正确的 obj_path
    obj_path = case.get("meta", {}).get("obj_path") if case else None
    if not obj_path or not Path(obj_path).exists():
        rp = case.get("result_path") if case else None
        if rp:
            for guess in [rp.with_name("textured.obj"),
                          rp.parent / "textured.obj",
                          rp.parent.parent / "textured.obj"]:
                if guess.exists():
                    obj_path = str(guess)
                    break
    obj_mesh_orn = case.get("meta", {}).get("object_orientation") if case else None
    if obj_mesh_orn is None and obj_path and Path(obj_path).exists() and len(obj_pts):
        obj_mesh_orn = _estimate_mesh_orientation_from_points(obj_path, obj_pts)
    has_mesh = bool(obj_path and Path(obj_path).exists())
    mesh_failed = False

    # 用于点云跟随的初始物体位置
    initial_obj_pos = np.array(frame_log[0].get('obj_pos', [0,0,0]), dtype=float) if frame_log else np.zeros(3)

    for fi, f in enumerate(frame_log):
        phase = f.get('phase', '?')
        grp = np.array(f['gripper_pos'])
        gr_orn = f.get('gripper_orn', [0,0,0,1])
        rot_m = Rot.from_quat(gr_orn).as_matrix()
        w_val = f.get('opening', 0.06)
        success = f.get('success', None)
        label = phase
        if success is not None:
            label = 'SUCCESS' if success else 'FAILED'

        traces = []
        if len(table_pts):
            traces.append(go.Scatter3d(
                x=table_pts[:,0], y=table_pts[:,1], z=table_pts[:,2],
                mode='markers', marker={'size':2,'color':'rgba(155,164,176,0.25)'},
                name='table', hoverinfo='skip'))

        obj_p = np.array(f['obj_pos'])
        obj_orn = f.get('obj_orn', [0,0,0,1])
        color = '#16a34a' if success else '#dc2626' if success is not None else '#2563eb'

        if has_mesh and not mesh_failed:
            try:
                traces.append(_render_mesh_trace(obj_path, obj_p, obj_orn, color))
            except Exception as e:
                import traceback, sys
                print(f"[GUI] Mesh render failed at frame {fi}: {e}", file=sys.stderr)
                traceback.print_exc()
                mesh_failed = True
        if not has_mesh or mesh_failed:
            # 降级：让物体点云跟随 obj_p 平移
            if len(obj_pts):
                delta = obj_p - initial_obj_pos
                shifted_pts = obj_pts + delta
                traces.append(go.Scatter3d(
                    x=shifted_pts[:,0], y=shifted_pts[:,1], z=shifted_pts[:,2],
                    mode='markers', marker={'size':3,'color':color,'opacity':0.7},
                    name='object', hoverinfo='skip'))
            else:
                traces.append(go.Scatter3d(
                    x=[obj_p[0]], y=[obj_p[1]], z=[obj_p[2]],
                    mode='markers', marker={'size':12,'color':color,'symbol':'diamond'},
                    name='object', hoverinfo='skip'))

        traces.extend(gripper_traces(grp, rot_m, w_val, selected.get('depth',0.03)))
        traces.append(go.Scatter3d(
            x=[grp[0]], y=[grp[1]], z=[grp[2]+0.06],
            mode='markers+text', marker={'size':4,'color':'#f59e0b'},
            text=[label], textposition='top center', name='stage', hoverinfo='skip'))
        frames.append(go.Frame(data=traces, name=str(fi)))
        bounds_all.extend([obj_p, grp])

    fig = go.Figure(data=frames[0].data, frames=frames)
    ba = np.array(bounds_all); c = ba.mean(axis=0)
    s = max(float(np.ptp(ba, axis=0).max()), 0.1); pad = s*0.6
    
    steps = [{'method': 'animate', 'args': [[str(i)], {'mode': 'immediate',
              'frame': {'duration': 0, 'redraw': True}, 'transition': {'duration': 0}}],
              'label': str(i)} for i in range(len(frames))]
    
    fig.update_layout(
        title=f"PyBullet Replay - grasp {selected['grasp_index']} ({len(frame_log)} frames)",
        margin={'l':0,'r':0,'t':32,'b':0}, paper_bgcolor='#f8fafc',
        scene={'xaxis':{'title':'X','range':[c[0]-pad,c[0]+pad]},
               'yaxis':{'title':'Y','range':[c[1]-pad,c[1]+pad]},
               'zaxis':{'title':'Z','range':[max(0,c[2]-pad),c[2]+pad]},
               'aspectmode':'cube','bgcolor':'#f8fafc'},
        updatemenus=[{'type':'buttons','showactive':False,'x':0.02,'y':1.05,
            'buttons':[{'label':'Play','method':'animate',
                'args':[None,{'frame':{'duration':60,'redraw':True},'fromcurrent':True}]},
                {'label':'Pause','method':'animate',
                'args':[[None],{'frame':{'duration':0,'redraw':False},'mode':'immediate'}]}]}],
        sliders=[{'active':0,'currentvalue':{'prefix':'Frame '},'pad':{'t':28},'steps':steps}],
    )
    return fig


def _object_grasp_center(obj_pts: np.ndarray) -> np.ndarray:
    if len(obj_pts) == 0:
        return np.array([0.3, 0.0, 0.05], dtype=float)
    mins = obj_pts.min(axis=0)
    maxs = obj_pts.max(axis=0)
    return np.array([
        (mins[0] + maxs[0]) / 2.0,
        (mins[1] + maxs[1]) / 2.0,
        max(TABLE_Z + TABLE_CLEARANCE, (mins[2] + maxs[2]) / 2.0),
    ], dtype=float)


def _constrained_grasp_pose(selected: dict, obj_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    raw_rot = np.asarray(selected["rotation"], dtype=float)
    raw_center = np.asarray(selected["translation"], dtype=float)
    if selected.get("physical_valid", True):
        return raw_rot, raw_center, False

    x_axis = np.array([0.0, 0.0, -1.0])
    y_hint = np.asarray(raw_rot[:, 1], dtype=float).copy()
    y_hint[2] = 0.0
    y_axis = _unit(y_hint, [0.0, 1.0, 0.0])
    z_axis = _unit(np.cross(x_axis, y_axis), [1.0, 0.0, 0.0])
    y_axis = _unit(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])
    rot = np.column_stack([x_axis, y_axis, z_axis])
    center = _object_grasp_center(obj_pts)
    return rot, center, True


def _animation_synthetic(selected, obj_pts, table_pts, pc, case):
    """合规演示动画：用 GraspNet 夹爪几何，从桌面上方接近并抓起物体。"""
    rot, center, adjusted = _constrained_grasp_pose(selected, obj_pts)
    width_closed = selected["width"]; width_open = max(0.06, width_closed)
    depth = selected["depth"]
    approach_dir = _grasp_approach_dir(rot)
    pre_grasp = center - approach_dir*(depth+0.05)
    lift_target = center + np.array([0,0,0.20])
    
    # 尝试获取真实 mesh 路径
    obj_path = case.get("meta", {}).get("obj_path") if case else None
    if not obj_path or not Path(obj_path).exists():
        rp = case.get("result_path") if case else None
        if rp:
            for guess in [rp.with_name("textured.obj"),
                          rp.parent / "textured.obj",
                          rp.parent.parent / "textured.obj"]:
                if guess.exists():
                    obj_path = str(guess)
                    break
    obj_mesh_orn = case.get("meta", {}).get("object_orientation") if case else None
    if obj_mesh_orn is None and obj_path and Path(obj_path).exists() and len(obj_pts):
        obj_mesh_orn = _estimate_mesh_orientation_from_points(obj_path, obj_pts)
    has_mesh = bool(obj_path and Path(obj_path).exists())
    
    # For the demonstration, place the mesh by its visual center rather than
    # trusting stale PyBullet object_position metadata from older result files.
    obj_center = center.copy()
    
    frame_specs = []
    for i in range(12):
        f=i/11; frame_specs.append((pre_grasp*(1-f)+center*f, width_open, 0.0, "approach"))
    for i in range(8):
        f=i/7; frame_specs.append((center, width_open*(1-f)+width_closed*f, 0.0, "close"))
    for i in range(16):
        f=i/15
        pos = center*(1-f)+lift_target*f
        frame_specs.append((pos, width_closed, float(pos[2] - center[2]), "lift"))
    
    frames = []
    for idx, (pos,w,shift,stage) in enumerate(frame_specs):
        mesh_pos = None
        if has_mesh:
            mesh_pos = _mesh_position_for_center(obj_path, obj_center + np.array([0.0, 0.0, shift]), obj_mesh_orn)
        frames.append(go.Frame(
            data=animation_traces(table_pts, obj_pts, pos, rot, w, depth, shift, stage, obj_path, mesh_pos, obj_mesh_orn), name=str(idx)))
    fig = go.Figure(data=frames[0].data, frames=frames)
    all_pts = np.vstack([obj_pts if len(obj_pts) else pc, pre_grasp[None,:], lift_target[None,:]])
    c=all_pts.mean(axis=0); s=max(float(np.ptp(all_pts, axis=0).max()),0.1); pad=s*0.7
    title = f"Constrained GraspNet Animation - grasp {selected['grasp_index']}"
    if adjusted:
        title += " (adjusted to object)"
    fig.update_layout(
        title=title,
        margin={'l':0,'r':0,'t':32,'b':0}, paper_bgcolor='#f8fafc',
        scene={'xaxis':{'title':'X','range':[c[0]-pad,c[0]+pad]},
               'yaxis':{'title':'Y','range':[c[1]-pad,c[1]+pad]},
               'zaxis':{'title':'Z','range':[max(0,c[2]-pad),c[2]+pad]},
               'aspectmode':'cube','bgcolor':'#f8fafc'},
        updatemenus=[{'type':'buttons','showactive':False,'x':0.02,'y':1.05,
            'buttons':[{'label':'Play','method':'animate',
                'args':[None,{'frame':{'duration':80,'redraw':True},'fromcurrent':True}]},
                {'label':'Pause','method':'animate',
                'args':[[None],{'frame':{'duration':0,'redraw':False},'mode':'immediate'}]}]}],
        sliders=[{'active':0,'currentvalue':{'prefix':'Frame '},'pad':{'t':28},
                   'steps':[{'method':'animate','args':[[str(i)],{'mode':'immediate',
                    'frame':{'duration':0,'redraw':True},'transition':{'duration':0}}],
                    'label':str(i)} for i in range(len(frames))]}],
    )
    return fig


def make_rgb_figure(rgb):
    fig = go.Figure()
    if rgb is not None and rgb.size:
        fig.add_trace(go.Image(z=rgb[:, :, :3]))
    fig.update_layout(title="RGB", margin={"l": 0, "r": 0, "t": 32, "b": 0}, xaxis_showticklabels=False, yaxis_showticklabels=False)
    return fig


def make_depth_figure(depth):
    fig = go.Figure()
    if depth is not None and depth.size:
        fig.add_trace(go.Heatmap(z=depth, colorscale="Viridis", colorbar={"title": "m"}))
    fig.update_layout(title="Depth", margin={"l": 0, "r": 0, "t": 32, "b": 0}, xaxis_showticklabels=False, yaxis_showticklabels=False)
    return fig


def make_summary(case, selected_index: int | None):
    results = case["results"]
    success = sum(1 for r in results if r["success"])
    total = len(results)
    selected = next((r for r in results if r["grasp_index"] == selected_index), None)
    lines = [
        html.Div([html.Span("Result: ", className="metric-label"), html.Code(_rel(case["result_path"]))]),
        html.Div([html.Span("Viz data: ", className="metric-label"), html.Code(_rel(case["viz_path"]))]),
        html.Div([html.Span("Success: ", className="metric-label"), html.Strong(f"{success}/{total}")]),
        html.Div([html.Span("Point cloud: ", className="metric-label"), html.Strong(f"{len(case['point_cloud']):,} pts")]),
        html.Div([html.Span("Trajectories: ", className="metric-label"), html.Strong(str(len(case["trajectories"])))]),
    ]
    if selected is not None:
        t = selected["translation"]
        lines.extend([
            html.H3(f"Grasp {selected['grasp_index']}", className="panel-subtitle"),
            html.Div([html.Span("Status: ", className="metric-label"), html.Strong("success" if selected["success"] else "failed")]),
            html.Div([html.Span("Raw status: ", className="metric-label"), html.Strong("success" if selected.get("raw_success", selected["success"]) else "failed")]),
            html.Div([html.Span("Physical valid: ", className="metric-label"), html.Strong("yes" if selected.get("physical_valid", True) else "no")]),
            html.Div([html.Span("Score: ", className="metric-label"), html.Strong(f"{selected['score']:.4f}")]),
            html.Div([
                html.Span("Lift delta: ", className="metric-label"),
                html.Strong(f"{selected.get('obj_lift_delta', 0.0):.4f} m"),
            ]),
            html.Div([html.Span("Width: ", className="metric-label"), html.Strong(f"{selected['width']:.4f} m")]),
            html.Div([html.Span("Depth: ", className="metric-label"), html.Strong(f"{selected['depth']:.4f} m")]),
            html.Div([html.Span("Center-object dist: ", className="metric-label"), html.Strong(f"{selected.get('center_object_dist', 0.0):.4f} m")]),
            html.Div([html.Span("Center: ", className="metric-label"), html.Code(f"[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")]),
        ])
        if selected.get("failure_reason"):
            lines.append(html.Div([html.Span("Failure reason: ", className="metric-label"), html.Code(selected["failure_reason"])]))
    return lines


def build_app(default_result: Path | None, default_viz: Path | None) -> Dash:
    result_files = find_result_files()
    if default_result and default_result.exists() and default_result not in result_files:
        result_files.insert(0, default_result)
    initial_result = default_result or (result_files[0] if result_files else None)
    initial_viz = default_viz or (infer_viz_path(initial_result) if initial_result else None)

    app = Dash(__name__, title="GraspNet GUI")
    app.layout = html.Div(className="page", children=[
        html.Div(className="topbar", children=[
            html.H1("GraspNet Result Viewer"),
            html.Div(className="topbar-meta", children="Dash Web GUI"),
        ]),
        html.Div(className="layout", children=[
            html.Div(className="sidebar", children=[
                html.H2("Controls"),
                html.Label("Result file"),
                dcc.Dropdown(
                    id="result-file",
                    options=[{"label": _rel(p), "value": str(p)} for p in result_files],
                    value=str(initial_result) if initial_result else None,
                    clearable=False,
                ),
                html.Label("Viz data"),
                dcc.Input(
                    id="viz-file",
                    value=str(initial_viz) if initial_viz else "",
                    type="text",
                    debounce=True,
                    className="path-input",
                ),
                html.Label("Top K"),
                dcc.Slider(id="top-k", min=1, max=30, step=1, value=10, marks={1: "1", 10: "10", 20: "20", 30: "30"}),
                html.Label("Point samples"),
                dcc.Slider(id="point-samples", min=1000, max=20000, step=1000, value=7000, marks={1000: "1k", 10000: "10k", 20000: "20k"}),
                html.Label("Score min"),
                dcc.Slider(id="score-min", min=-0.5, max=2.0, step=0.05, value=-0.5, marks={-0.5: "-.5", 0: "0", 1: "1", 2: "2"}),
                html.Label("Outcome"),
                dcc.Checklist(
                    id="outcomes",
                    options=[
                        {"label": "Success", "value": "success"},
                        {"label": "Failed", "value": "failed"},
                    ],
                    value=["success", "failed"],
                    inline=True,
                ),
                html.Label("Selected grasp"),
                dcc.Dropdown(id="selected-grasp", clearable=False),
                html.Div(id="summary", className="summary"),
            ]),
            html.Div(className="main", children=[
                dcc.Graph(id="scene-3d", className="scene-graph", config={"displaylogo": False}),
                dcc.Graph(id="animation-3d", className="animation-graph", config={"displaylogo": False}),
                html.Div(className="image-row", children=[
                    dcc.Graph(id="rgb-graph", className="image-graph", config={"displaylogo": False}),
                    dcc.Graph(id="depth-graph", className="image-graph", config={"displaylogo": False}),
                ]),
            ]),
        ]),
    ])

    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; background: #f8fafc; color: #111827; font-family: Inter, Arial, sans-serif; }
            .page { min-height: 100vh; }
            .topbar { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; border-bottom: 1px solid #d9e2ec; background: #ffffff; }
            h1 { font-size: 20px; margin: 0; font-weight: 700; }
            h2 { font-size: 15px; margin: 0 0 14px 0; }
            .topbar-meta { font-size: 13px; color: #52616b; }
            .layout { display: grid; grid-template-columns: 330px minmax(0, 1fr); min-height: calc(100vh - 57px); }
            .sidebar { border-right: 1px solid #d9e2ec; background: #ffffff; padding: 16px; overflow: auto; }
            .main { padding: 12px; min-width: 0; }
            label { display: block; margin: 14px 0 6px; font-size: 12px; font-weight: 700; color: #334155; }
            .path-input { width: 100%; height: 34px; border: 1px solid #cbd5e1; padding: 0 8px; box-sizing: border-box; }
            .scene-graph { height: 65vh; border: 1px solid #d9e2ec; background: #f8fafc; }
            .animation-graph { height: 46vh; border: 1px solid #d9e2ec; background: #f8fafc; margin-top: 12px; }
            .image-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
            .image-graph { height: 27vh; border: 1px solid #d9e2ec; background: #ffffff; }
            .summary { margin-top: 18px; font-size: 13px; line-height: 1.8; color: #1f2937; }
            .summary code { white-space: normal; overflow-wrap: anywhere; background: #eef2f7; padding: 2px 4px; }
            .metric-label { color: #64748b; }
            .panel-subtitle { font-size: 14px; margin: 14px 0 4px; border-top: 1px solid #e2e8f0; padding-top: 12px; }
            @media (max-width: 900px) {
                .layout { grid-template-columns: 1fr; }
                .sidebar { border-right: 0; border-bottom: 1px solid #d9e2ec; }
                .image-row { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

    @app.callback(
        Output("viz-file", "value"),
        Input("result-file", "value"),
    )
    def update_viz_path(result_file):
        rp = _resolve(result_file)
        vp = infer_viz_path(rp) if rp else None
        return str(vp) if vp else ""

    @app.callback(
        Output("selected-grasp", "options"),
        Output("selected-grasp", "value"),
        Input("result-file", "value"),
        Input("viz-file", "value"),
        Input("top-k", "value"),
        Input("outcomes", "value"),
        Input("score-min", "value"),
    )
    def update_selected_options(result_file, viz_file, top_k, outcomes, score_min):
        case = load_case(result_file, viz_file or None)
        rows = filtered_results(case["results"], int(top_k), outcomes or [], float(score_min))
        options = [
            {
                "label": f"{r['grasp_index']} | {'success' if r['success'] else 'failed'} | score {r['score']:.3f}",
                "value": int(r["grasp_index"]),
            }
            for r in rows
        ]
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("scene-3d", "figure"),
        Output("animation-3d", "figure"),
        Output("rgb-graph", "figure"),
        Output("depth-graph", "figure"),
        Output("summary", "children"),
        Input("result-file", "value"),
        Input("viz-file", "value"),
        Input("top-k", "value"),
        Input("outcomes", "value"),
        Input("score-min", "value"),
        Input("selected-grasp", "value"),
        Input("point-samples", "value"),
    )
    def update_figures(result_file, viz_file, top_k, outcomes, score_min, selected_grasp, point_samples):
        case = load_case(result_file, viz_file or None)
        selected = int(selected_grasp) if selected_grasp is not None else None
        scene = make_3d_figure(case, int(top_k), outcomes or [], float(score_min), selected, int(point_samples))
        animation = make_animation_figure(case, selected)
        return scene, animation, make_rgb_figure(case["rgb"]), make_depth_figure(case["depth"]), make_summary(case, selected)

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Dash GUI for GraspNet simulation results")
    parser.add_argument("--results", default=None, help="Path to results.json")
    parser.add_argument("--viz-data", default=None, help="Path to viz_data.pkl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app(_resolve(args.results), _resolve(args.viz_data))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
