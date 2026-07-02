"""Export static grasp-candidate diagnostics for simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from graspnetAPI.utils.utils import plot_gripper_pro_max
except Exception:  # pragma: no cover - keeps non-GraspNet tooling importable.
    plot_gripper_pro_max = None


def _as_array(value, shape=None):
    arr = np.asarray(value, dtype=float)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _choose_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def _choose_points_and_colors(points: np.ndarray, colors: np.ndarray | None, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx], None if colors is None else colors[idx]


def _normalize_rgb_colors(colors: np.ndarray | None, count: int, fallback=(145, 150, 158)) -> np.ndarray:
    if colors is None:
        return np.tile(np.asarray(fallback, dtype=np.uint8), (count, 1))
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Point colors must have shape Nx3, got {colors.shape}")
    if colors.dtype.kind == "f" and colors.max(initial=0) <= 1.0:
        colors = np.rint(colors * 255.0)
    return np.clip(colors, 0, 255).astype(np.uint8, copy=False)


def _gripper_wireframe(center, rot, width, depth, visual_scale=1.0):
    """Return line segments for a compact GraspNet-style parallel gripper."""
    center = _as_array(center)
    rot = _as_array(rot, (3, 3))
    visual_scale = max(float(visual_scale), 0.1)
    opening = max(float(width), 0.004) * visual_scale
    depth = max(float(depth), 0.025) * visual_scale
    height = 0.014 * visual_scale
    base_back = center - rot[:, 0] * 0.035 * visual_scale
    left_tip = center + rot[:, 0] * depth + rot[:, 1] * opening / 2.0
    right_tip = center + rot[:, 0] * depth - rot[:, 1] * opening / 2.0
    left_back = center - rot[:, 0] * 0.015 * visual_scale + rot[:, 1] * opening / 2.0
    right_back = center - rot[:, 0] * 0.015 * visual_scale - rot[:, 1] * opening / 2.0
    left_top = left_back + rot[:, 2] * height
    right_top = right_back + rot[:, 2] * height
    left_tip_top = left_tip + rot[:, 2] * height
    right_tip_top = right_tip + rot[:, 2] * height
    return [
        (base_back, left_back),
        (base_back, right_back),
        (left_back, left_tip),
        (right_back, right_tip),
        (left_top, left_tip_top),
        (right_top, right_tip_top),
        (left_back, left_top),
        (right_back, right_top),
        (left_tip, left_tip_top),
        (right_tip, right_tip_top),
    ]


def _axis_lines(center, rot, length=0.035):
    center = _as_array(center)
    rot = _as_array(rot, (3, 3))
    return [
        (center, center + rot[:, 0] * length, (255, 40, 40)),
        (center, center + rot[:, 1] * length, (40, 255, 40)),
        (center, center + rot[:, 2] * length, (40, 120, 255)),
    ]


def _center_cross_lines(center, size=0.01):
    center = _as_array(center)
    axes = np.eye(3, dtype=float)
    lines = []
    for axis in axes:
        lines.append((center - axis * size, center + axis * size))
    return lines


def _ply_add_line(vertices, edges, start, end, color):
    start = _as_array(start)
    end = _as_array(end)
    color = tuple(int(max(0, min(255, value))) for value in color)
    start_index = len(vertices)
    vertices.append((float(start[0]), float(start[1]), float(start[2]), *color))
    end_index = len(vertices)
    vertices.append((float(end[0]), float(end[1]), float(end[2]), *color))
    edges.append((start_index, end_index, *color))


def _candidate_mesh_color(index: int, score: float) -> tuple[float, float, float]:
    if index == 0:
        return (1.0, 0.08, 0.02)
    score = max(0.0, min(float(score), 1.0))
    return (0.05 + 0.25 * score, 0.15 + 0.35 * score, 1.0)


def _mesh_to_colored_arrays(mesh, fallback_color: tuple[float, float, float]):
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)
    colors = np.asarray(mesh.vertex_colors, dtype=float)
    if len(colors) != len(vertices):
        colors = np.tile(np.asarray(fallback_color, dtype=float), (len(vertices), 1))
    colors = _normalize_rgb_colors(colors, len(vertices))
    return vertices, triangles, colors


def _sample_mesh_surface_points(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, spacing: float = 0.002):
    sampled_points = []
    sampled_colors = []
    for face in faces:
        tri_vertices = vertices[np.asarray(face, dtype=int)]
        tri_colors = colors[np.asarray(face, dtype=int)]
        edge_lengths = [
            np.linalg.norm(tri_vertices[1] - tri_vertices[0]),
            np.linalg.norm(tri_vertices[2] - tri_vertices[1]),
            np.linalg.norm(tri_vertices[0] - tri_vertices[2]),
        ]
        steps = max(1, int(np.ceil(max(edge_lengths) / max(float(spacing), 1e-4))))
        color = np.rint(tri_colors.mean(axis=0)).astype(np.uint8)
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                a = i / steps
                b = j / steps
                c = 1.0 - a - b
                point = a * tri_vertices[0] + b * tri_vertices[1] + c * tri_vertices[2]
                sampled_points.append(point)
                sampled_colors.append(color)
    return sampled_points, sampled_colors


def _set_equal_axes(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = max(float(np.max(maxs - mins)), 0.12)
    radius = span * 0.65
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def export_candidate_png(
    point_cloud,
    candidates,
    results,
    output_path,
    max_points=9000,
    max_candidates=40,
    gripper_visual_scale=1.0,
):
    """Write a PNG showing point cloud, candidate grippers, and executed poses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(point_cloud, dtype=float)
    if points.ndim == 3:
        points = points[0]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampled = _choose_points(points, max_points)
    candidates = list(candidates or [])[:max_candidates]
    results = list(results or [])

    fig = plt.figure(figsize=(14, 9), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(sampled[:, 2] > 0.005, "#8a8f98", "#c7cbd1")
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=2, c=colors, alpha=0.72, depthshade=False)

    for candidate in candidates:
        center = _as_array(candidate.get("execution_translation", candidate.get("translation", [0, 0, 0])))
        rot = _as_array(candidate.get("execution_rotation", candidate.get("rotation", np.eye(3))), (3, 3))
        score = float(candidate.get("score", 0.0))
        width = float(candidate.get("width", 0.06))
        depth = float(candidate.get("depth", 0.03))
        alpha = 0.25 + min(max(score, 0.0), 2.0) * 0.18
        for a, b in _gripper_wireframe(center, rot, width, depth, gripper_visual_scale):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#143df5", alpha=alpha, linewidth=2.0)
        approach_end = center + rot[:, 0] * max(depth, 0.025) * max(float(gripper_visual_scale), 0.1)
        ax.plot(
            [center[0], approach_end[0]],
            [center[1], approach_end[1]],
            [center[2], approach_end[2]],
            color="#143df5",
            alpha=alpha,
            linewidth=1.0,
        )
        ax.scatter([center[0]], [center[1]], [center[2]], s=8, c="#143df5", marker=".", depthshade=False)

    for result in results:
        center = _as_array(result.get("execution_translation", result.get("translation", [0, 0, 0])))
        rot = _as_array(result.get("rotation", np.eye(3)), (3, 3))
        width = float(result.get("width", 0.06))
        depth = float(result.get("depth", 0.03))
        color = "#00a651" if result.get("success") else "#e00022"
        for a, b in _gripper_wireframe(center, rot, width, depth, gripper_visual_scale):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, alpha=0.98, linewidth=2.8)
        ax.scatter([center[0]], [center[1]], [center[2]], s=48, c=color, marker="o", depthshade=False)
        raw = _as_array(result.get("translation", center))
        ax.scatter([raw[0]], [raw[1]], [raw[2]], s=34, c="#111827", marker="x", depthshade=False)
        ax.plot([raw[0], center[0]], [raw[1], center[1]], [raw[2], center[2]], color="#111827", alpha=0.50, linewidth=1.0)

    ax.set_title("Top-down Grasp Candidates")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-62)
    _set_equal_axes(ax, sampled if len(sampled) else points)
    ax.text2D(
        0.02,
        0.96,
        "blue: top-down candidates | red/green: executed poses | black x: raw GraspNet center",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def export_candidate_ply(
    point_cloud,
    candidates,
    output_path,
    point_colors=None,
    max_points=60000,
    max_candidates=20,
    gripper_visual_scale=1.0,
):
    """Write an ASCII PLY with RGB points and GraspNet-style candidate gripper meshes."""
    points = np.asarray(point_cloud, dtype=float)
    if points.ndim == 3:
        points = points[0]
    colors = None if point_colors is None else np.asarray(point_colors)
    sampled, sampled_colors = _choose_points_and_colors(points, colors, max_points)
    sampled_colors = _normalize_rgb_colors(sampled_colors, len(sampled))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = [
        (float(point[0]), float(point[1]), float(point[2]), int(color[0]), int(color[1]), int(color[2]))
        for point, color in zip(sampled, sampled_colors)
    ]
    edges = []
    faces = []
    visual_scale = max(float(gripper_visual_scale), 0.1)

    for idx, candidate in enumerate(list(candidates or [])[:max_candidates]):
        center = _as_array(candidate.get("execution_translation", candidate.get("translation", [0, 0, 0])))
        rot = _as_array(candidate.get("execution_rotation", candidate.get("rotation", np.eye(3))), (3, 3))
        score = float(candidate.get("score", 0.0))
        width = float(candidate.get("width", 0.06))
        depth = float(candidate.get("depth", 0.03))
        color_float = _candidate_mesh_color(idx, score)
        color_u8 = tuple(int(round(value * 255.0)) for value in color_float)
        if plot_gripper_pro_max is not None:
            mesh = plot_gripper_pro_max(
                center,
                rot,
                max(width * visual_scale, 0.004),
                max(depth * visual_scale, 0.005),
                score=score,
                color=color_float,
            )
            mesh_vertices, mesh_faces, mesh_colors = _mesh_to_colored_arrays(mesh, color_float)
            vertex_offset = len(vertices)
            for point, color in zip(mesh_vertices, mesh_colors):
                vertices.append(
                    (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        int(color[0]),
                        int(color[1]),
                        int(color[2]),
                    )
                )
            for face in mesh_faces:
                faces.append((int(face[0]) + vertex_offset, int(face[1]) + vertex_offset, int(face[2]) + vertex_offset))
            surface_points, surface_colors = _sample_mesh_surface_points(mesh_vertices, mesh_faces, mesh_colors)
            for point, color in zip(surface_points, surface_colors):
                vertices.append(
                    (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        int(color[0]),
                        int(color[1]),
                        int(color[2]),
                    )
                )
        else:
            for start, end in _gripper_wireframe(center, rot, width, depth, visual_scale):
                _ply_add_line(vertices, edges, start, end, color_u8)
        for start, end, axis_color in _axis_lines(center, rot, length=max(depth, 0.025) * visual_scale):
            _ply_add_line(vertices, edges, start, end, axis_color)
        for start, end in _center_cross_lines(center, size=0.006 * visual_scale):
            _ply_add_line(vertices, edges, start, end, color_u8)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("comment SmartGrasp RGB point cloud with GraspNet official-style candidate gripper meshes\n")
        file.write("comment units meters; frame camera\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"element face {len(faces)}\n")
        file.write("property list uchar int vertex_indices\n")
        file.write(f"element edge {len(edges)}\n")
        file.write("property int vertex1\n")
        file.write("property int vertex2\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for vertex in vertices:
            file.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} {vertex[3]} {vertex[4]} {vertex[5]}\n")
        for face in faces:
            file.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        for edge in edges:
            file.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]} {edge[4]}\n")
    return output_path


def export_candidate_html(
    point_cloud,
    candidates,
    results,
    output_path,
    max_points=12000,
    max_candidates=50,
    gripper_visual_scale=1.0,
):
    """Write an interactive Plotly HTML view if Plotly is installed."""
    import plotly.graph_objects as go

    points = np.asarray(point_cloud, dtype=float)
    if points.ndim == 3:
        points = points[0]
    sampled = _choose_points(points, max_points)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=sampled[:, 0],
            y=sampled[:, 1],
            z=sampled[:, 2],
            mode="markers",
            marker={
                "size": 2.6,
                "color": sampled[:, 2],
                "colorscale": [
                    [0.0, "#38bdf8"],
                    [0.35, "#22c55e"],
                    [0.7, "#facc15"],
                    [1.0, "#fb7185"],
                ],
                "opacity": 0.9,
            },
            name="point cloud",
        )
    )

    for idx, candidate in enumerate(list(candidates or [])[:max_candidates]):
        center = _as_array(candidate.get("execution_translation", candidate.get("translation", [0, 0, 0])))
        rot = _as_array(candidate.get("execution_rotation", candidate.get("rotation", np.eye(3))), (3, 3))
        score = float(candidate.get("score", 0.0))
        width = float(candidate.get("width", 0.06))
        depth = float(candidate.get("depth", 0.03))
        xs, ys, zs = [], [], []
        for a, b in _gripper_wireframe(center, rot, width, depth, gripper_visual_scale):
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
        line_width = 8 if idx == 0 else 5
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": "rgba(96,165,250,0.96)", "width": line_width},
                name=f"candidate {idx} score={score:.3f}",
                showlegend=idx < 10,
                hovertemplate=f"candidate {idx}<br>score={score:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers+text",
                marker={"size": 5 if idx else 8, "color": "#f97316"},
                text=[str(idx)],
                textposition="top center",
                textfont={"color": "#f8fafc", "size": 14 if idx == 0 else 12},
                name=f"center {idx}",
                showlegend=False,
                hovertemplate=f"candidate {idx} center<br>score={score:.3f}<extra></extra>",
            )
        )

    for result in results or []:
        center = _as_array(result.get("execution_translation", result.get("translation", [0, 0, 0])))
        rot = _as_array(result.get("rotation", np.eye(3)), (3, 3))
        width = float(result.get("width", 0.06))
        depth = float(result.get("depth", 0.03))
        color = "rgba(74,222,128,0.98)" if result.get("success") else "rgba(248,113,113,0.98)"
        xs, ys, zs = [], [], []
        for a, b in _gripper_wireframe(center, rot, width, depth, gripper_visual_scale):
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": color, "width": 6},
                name=f"executed {result.get('grasp_index', '?')}",
            )
        )

    fig.update_layout(
        title="Interactive Grasp Candidates",
        template="plotly_dark",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "#e5e7eb"},
        scene={
            "aspectmode": "cube",
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "bgcolor": "#020617",
            "xaxis": {
                "backgroundcolor": "#0f172a",
                "gridcolor": "#334155",
                "zerolinecolor": "#94a3b8",
                "showbackground": True,
            },
            "yaxis": {
                "backgroundcolor": "#0f172a",
                "gridcolor": "#334155",
                "zerolinecolor": "#94a3b8",
                "showbackground": True,
            },
            "zaxis": {
                "backgroundcolor": "#0f172a",
                "gridcolor": "#334155",
                "zerolinecolor": "#94a3b8",
                "showbackground": True,
            },
            "camera": {"eye": {"x": 1.45, "y": -1.45, "z": 1.15}},
        },
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        legend={"itemsizing": "constant"},
    )
    fig.write_html(str(output_path), include_plotlyjs=True)
    return output_path
