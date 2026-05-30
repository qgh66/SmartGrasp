"""
Phase 1_GUI 可视化模块。

两种模式：
  路径 A (--mode static):   生成 4 合 1 静态 PNG 图，用于技术报告。
  路径 B (--mode interactive): 打开 Open3D 交互窗口，自由旋转查看点云和抓取。
  路径 AB (--mode both):    两者都执行。

用法:
  python -m simulation.visualize \
      --results results/results.json \
      --viz_data results/viz_data.pkl \
      --mode both \
      --out_dir results
"""

import sys
import os
import argparse
import json
import pickle
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "graspnetAPI"))
from graspnetAPI.grasp import GraspGroup, Grasp


# ======================================================================
# 路径 A: 静态图 (Matplotlib)
# ======================================================================

def _plot_rgb(rgb, output_dir):
    """图一: RGB 图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb[:, :, :3])
    ax.set_title("Virtual Camera RGB", fontsize=12)
    ax.axis("off")
    path = os.path.join(output_dir, "fig1_rgb.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [fig1] {path}")


def _plot_depth(depth, output_dir):
    """图二: 深度图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(depth, cmap="plasma")
    ax.set_title("Virtual Camera Depth", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8, label="m")
    path = os.path.join(output_dir, "fig2_depth.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [fig2] {path}")


def _plot_pointcloud_grasps(point_cloud, results, output_dir):
    """图三: 3D 点云 + 抓取姿态。物体区域蓝色、桌面灰色，仅显示最优抓取。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pickle as pkl

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    pc = point_cloud[0]  # (N, 3)

    # 物体点: Z > 0.005 (高于桌面5mm)
    is_object = pc[:, 2] > 0.005
    n_obj = is_object.sum()
    print(f"    [fig3 debug] total={len(pc)}, object_pts={n_obj}, Z_obj=[{pc[is_object,2].min():.3f},{pc[is_object,2].max():.3f}]")

    # 桌面: 降采样灰色
    table_pts = pc[~is_object]
    n_table_show = min(4000, len(table_pts))
    if n_table_show > 0:
        idxs_t = np.random.choice(len(table_pts), n_table_show, replace=False)
        ax.scatter(table_pts[idxs_t, 0], table_pts[idxs_t, 1], table_pts[idxs_t, 2],
                   c="lightgray", s=0.5, alpha=0.4, label="Table")

    # 物体: 蓝色，全部显示
    obj_pts = pc[is_object]
    if len(obj_pts) > 0:
        ax.scatter(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
                   c="#1A73E8", s=1.5, alpha=0.9, label=f"Object ({len(obj_pts)} pts)")

    # 只显示第一个抓取（最优）
    if results:
        r = results[0]
        t = np.array(r["translation"])
        R = np.array(r["rotation"])
        length = 0.06
        for ai, ac in enumerate(["r", "g", "b"]):
            ax.quiver(t[0], t[1], t[2], R[0, ai]*length, R[1, ai]*length, R[2, ai]*length,
                      color=ac, linewidth=2.5, alpha=0.9)
        c = "#00C853" if r["success"] else "#D50000"
        ax.scatter(t[0], t[1], t[2], c=c, marker="o", s=120, zorder=10,
                   edgecolors="black", linewidths=0.5)
        label = "Grasp (Success)" if r["success"] else "Grasp (Failed)"
        ax.text(t[0]+0.02, t[1]+0.02, t[2]+0.03, label, fontsize=8, color=c, fontweight="bold")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])

    # 自动缩放到物体区域
    if len(obj_pts) > 0:
        margin = 0.05
        ax.set_xlim(obj_pts[:, 0].min()-margin, obj_pts[:, 0].max()+margin)
        ax.set_ylim(obj_pts[:, 1].min()-margin, obj_pts[:, 1].max()+margin)
        ax.set_zlim(0, obj_pts[:, 2].max()+margin)

    ax.set_title(f"Point Cloud (Pre-Grasp) + Best Grasp [{n_obj} object pts]\n[Blue=Object, Gray=Table, RGB axes=Grasp Pose]", fontsize=11)
    ax.legend(loc="upper right", fontsize=7)
    path = os.path.join(output_dir, "fig3_pointcloud_grasps.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [fig3] {path}")


def _plot_score_chart(results, output_dir):
    """图四: 抓取得分柱状图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = [r["grasp_index"] for r in results]
    scores = [r["score"] for r in results]
    colors = ["#34A853" if r["success"] else "#EA4335" for r in results]
    bars = ax.bar(indices, scores, color=colors, edgecolor="black", linewidth=0.3)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{s:.2f}", ha="center", fontsize=6, rotation=90)
    ax.set_xlabel("Grasp Rank"); ax.set_ylabel("Score")
    ax.set_title("Grasp Score Ranking  (Green=Success  Red=Failed)", fontsize=12)
    path = os.path.join(output_dir, "fig4_score_ranking.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [fig4] {path}")


def _plot_summary(results, output_dir):
    """图五: 成功率饼图 + 统计文字。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    success = sum(1 for r in results if r["success"])
    fail = len(results) - success
    sizes = [success, fail]
    colors = ["#34A853", "#EA4335"]
    labels = [f"Success ({success})", f"Failed ({fail})"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    avg_lift = np.mean([r["lift_z"] for r in results if r["success"]]) if success > 0 else 0
    stats = (f"Total: {len(results)}\nSuccess: {success}\nFailed: {fail}\n"
             f"Avg Lift (success): {avg_lift:.3f} m")
    ax.text(-1.5, -1.3, stats, fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Grasp Success Rate", fontsize=12)
    path = os.path.join(output_dir, "fig5_success_rate.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [fig5] {path}")


def _plot_scene_overview(viz_data, output_dir):
    """图零: 场景宏观示意图 — 3D 展示桌面、实际物体轮廓、相机位置。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc = viz_data["point_cloud"][0]
    # 物体点: Z > 0.005
    obj_pts = pc[pc[:, 2] > 0.005]

    if len(obj_pts) > 0:
        cx, cy = np.median(obj_pts[:, 0]), np.median(obj_pts[:, 1])
        obj_z_min = np.percentile(pc[:, 2], 10)  # 桌面
        obj_top = obj_pts[:, 2].max()
        obj_half_x = (obj_pts[:, 0].max() - obj_pts[:, 0].min()) / 2
        obj_half_y = (obj_pts[:, 1].max() - obj_pts[:, 1].min()) / 2
    else:
        cx, cy = np.median(pc[:, 0]), np.median(pc[:, 1])
        obj_z_min = 0
        obj_top = 0.05
        obj_half_x = obj_half_y = 0.05

    # 相机实际位置 (0.6m)
    cam = np.array([0.3, 0.0, 0.6])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- 桌面 ---
    tx = [-0.2, 0.8, 0.8, -0.2, -0.2]
    ty = [-0.4, -0.4, 0.4, 0.4, -0.4]
    ax.plot(tx, ty, [obj_z_min]*5, "gray", linewidth=2, alpha=0.6, label="Table (Z=0)")

    # --- 物体包围盒（从点云动态推算） ---
    def plot_edge(a, b, **kw):
        ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], **kw)

    hx, hy = obj_half_x, obj_half_y
    z0, z1 = obj_z_min, obj_top
    cv = np.array([[cx-hx, cy-hy, z0], [cx+hx, cy-hy, z0],
                   [cx+hx, cy+hy, z0], [cx-hx, cy+hy, z0],
                   [cx-hx, cy-hy, z1], [cx+hx, cy-hy, z1],
                   [cx+hx, cy+hy, z1], [cx-hx, cy+hy, z1]])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        plot_edge(cv[i], cv[j], color="#1A73E8", linewidth=2, alpha=0.8,
                  label="Object" if (i,j)==(0,1) else None)

    # --- 相机 ---
    ax.scatter(*cam, c="red", s=180, marker="^", zorder=10)
    ax.text(cam[0]+0.03, cam[1], cam[2]+0.03, "Camera (0.3, 0, 0.6)",
            fontsize=7, color="red")
    ax.plot([cam[0], cx], [cam[1], cy], [cam[2], z0], "r--", linewidth=1.5, alpha=0.5)

    # --- FOV ---
    fov = np.deg2rad(60)
    dz = cam[2] - z0
    hw = dz * np.tan(fov/2) * (640/480)
    hh = dz * np.tan(fov/2)
    corners = [(cx+hw,cy+hh),(cx-hw,cy+hh),(cx-hw,cy-hh),(cx+hw,cy-hh)]
    for dx, dy in corners:
        ax.plot([cam[0], cx+dx*0.6], [cam[1], cy+dy*0.6], [cam[2], z0],
                "gray", linewidth=0.5, alpha=0.3)
    gx = [cx+hw, cx-hw, cx-hw, cx+hw, cx+hw]
    gy = [cy+hh, cy+hh, cy-hh, cy-hh, cy+hh]
    ax.plot(gx, gy, [z0]*5, "gray", linewidth=0.8, alpha=0.3, linestyle="dotted")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Scene Overview (3D)", fontsize=12)
    ax.set_xlim(-0.3, 1.0); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 0.8)
    ax.legend(loc="upper right", fontsize=7)

    path = os.path.join(output_dir, "fig0_scene_overview.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [fig0] {path}")


    ax.set_title("Scene Overview (3D)", fontsize=12)
    ax.set_xlim(-0.3, 1.0); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 1.6)
    ax.legend(loc="upper right", fontsize=7)

    path = os.path.join(output_dir, "fig0_scene_overview.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [fig0] {path}")


# ======================================================================
# 路径 B: 交互式 (Open3D)
# ======================================================================

def generate_interactive(results, viz_data):
    """打开 Open3D 交互窗口，显示点云 + 抓取姿态。"""
    import open3d as o3d

    pc = viz_data["point_cloud"][0]  # (N, 3)

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64))

    # 为每个抓取生成坐标系
    geometries = [pcd]
    for r in results:
        t = np.array(r["translation"])
        R = np.array(r["rotation"])

        # 创建坐标系框
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        # 构建 4x4 变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        frame.transform(T)

        if r["success"]:
            # 绿色坐标系 = 成功
            frame.paint_uniform_color([0, 0.8, 0])
        else:
            # 红色坐标系 = 失败
            frame.paint_uniform_color([0.8, 0, 0])
        geometries.append(frame)

        # 添加小球标记抓取中心
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(t)
        sphere.paint_uniform_color([0, 0.8, 0] if r["success"] else [0.8, 0, 0])
        geometries.append(sphere)

    print(f"[Interactive] 绿色=成功的抓取, 红色=失败的抓取")
    print(f"[Interactive] 鼠标左键旋转, 滚轮缩放, 中键平移. 按 Q 关闭窗口.")
    o3d.visualization.draw_geometries(geometries, window_name="GraspNet Simulation Results")


# ======================================================================
# 主入口
# ======================================================================

def generate_static(results, viz_data, output_dir):
    """生成六张独立静态 PNG 图到 output_dir。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Static] 生成图片到 {output_dir}/ ...")
    _plot_scene_overview(viz_data, output_dir)
    _plot_rgb(viz_data["rgb"], output_dir)
    _plot_depth(viz_data["depth"], output_dir)
    _plot_pointcloud_grasps(viz_data["point_cloud"], results, output_dir)
    _plot_score_chart(results, output_dir)
    _plot_summary(results, output_dir)
    print(f"[Static] 完成")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1_GUI 可视化工具")
    parser.add_argument("--results", type=str, required=True,
                        help="Phase 1 results.json 路径")
    parser.add_argument("--viz_data", type=str, required=True,
                        help="Phase 1 viz_data.pkl 路径")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["static", "interactive", "both"],
                        help="static=PNG图片, interactive=Open3D, both=两者")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="静态图输出目录 (仅 static/both 模式)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.results, "r") as f:
        data = json.load(f)
    with open(args.viz_data, "rb") as f:
        viz_data = pickle.load(f)

    # 兼容两种 results.json 格式
    if 'grasps' in data:
        results = data['grasps']
        print(f"[加载] {data['success']}/{data['total']} 成功, "
              f"点云 ({viz_data['point_cloud'].shape[1]} pts)")
    else:
        results = data
        print(f"[加载] {len(results)} 个抓取, 点云 ({viz_data['point_cloud'].shape[1]} pts)")

    if args.mode in ("static", "both"):
        out_dir = os.path.join(ROOT_DIR, args.out_dir)
        generate_static(results, viz_data, out_dir)

    if args.mode in ("interactive", "both"):
        generate_interactive(results, viz_data)


if __name__ == "__main__":
    main()
