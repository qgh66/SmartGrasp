"""Export static grasp-candidate diagnostics for simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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


def _gripper_wireframe(center, rot, width, depth):
    """Return line segments for a compact GraspNet-style parallel gripper."""
    center = _as_array(center)
    rot = _as_array(rot, (3, 3))
    opening = max(float(width), 0.004)
    depth = max(float(depth), 0.025)
    height = 0.014
    base_back = center - rot[:, 0] * 0.035
    left_tip = center + rot[:, 0] * depth + rot[:, 1] * opening / 2.0
    right_tip = center + rot[:, 0] * depth - rot[:, 1] * opening / 2.0
    left_back = center - rot[:, 0] * 0.015 + rot[:, 1] * opening / 2.0
    right_back = center - rot[:, 0] * 0.015 - rot[:, 1] * opening / 2.0
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


def export_candidate_png(point_cloud, candidates, results, output_path, max_points=9000, max_candidates=40):
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
        for a, b in _gripper_wireframe(center, rot, width, depth):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#143df5", alpha=alpha, linewidth=1.3)
        approach_end = center + rot[:, 0] * max(depth, 0.025)
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
        for a, b in _gripper_wireframe(center, rot, width, depth):
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


def export_candidate_html(point_cloud, candidates, results, output_path, max_points=12000, max_candidates=50):
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
            marker={"size": 2, "color": sampled[:, 2], "colorscale": "Viridis", "opacity": 0.55},
            name="point cloud",
        )
    )

    for idx, candidate in enumerate(list(candidates or [])[:max_candidates]):
        center = _as_array(candidate.get("execution_translation", candidate.get("translation", [0, 0, 0])))
        rot = _as_array(candidate.get("execution_rotation", candidate.get("rotation", np.eye(3))), (3, 3))
        width = float(candidate.get("width", 0.06))
        depth = float(candidate.get("depth", 0.03))
        xs, ys, zs = [], [], []
        for a, b in _gripper_wireframe(center, rot, width, depth):
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": "rgba(20,61,245,0.65)", "width": 3},
                name=f"candidate {idx}",
                showlegend=idx < 3,
            )
        )

    for result in results or []:
        center = _as_array(result.get("execution_translation", result.get("translation", [0, 0, 0])))
        rot = _as_array(result.get("rotation", np.eye(3)), (3, 3))
        width = float(result.get("width", 0.06))
        depth = float(result.get("depth", 0.03))
        color = "rgba(22,163,74,0.95)" if result.get("success") else "rgba(220,38,38,0.95)"
        xs, ys, zs = [], [], []
        for a, b in _gripper_wireframe(center, rot, width, depth):
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
        title="Grasp Candidates and Executed Poses",
        scene={"aspectmode": "cube", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path
