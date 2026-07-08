#!/usr/bin/env python
"""
GraspNet + PyBullet 闭环仿真 Demo。

流程: PyBullet 场景 → 虚拟相机 → 点云 → GraspNet 推理 → 抓取执行 → 评估

用法:
  conda activate smartgrasp
  cd /home/admin128/beilei/graspnet-workspace
  python scripts/demo_closed_loop.py
"""

import sys, os, json, argparse, pickle, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if 'graspnet-workspace/pointnet2' not in p and 'graspnet-workspace/knn' not in p]
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'models'))
sys.path.insert(0, os.path.join(ROOT, 'utils'))
sys.path.insert(0, os.path.join(ROOT, 'graspnet_api'))

import torch
from models.graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from simulation.scene import SimulationScene
from simulation.camera import VirtualCamera
from simulation.robot_gripper import JakaZu3Robotiq85Gripper
from simulation.evaluator import GraspEvaluator


def _resolve_path(path, *, config_dir=None):
    """Resolve a user/config path against common SmartGrasp roots."""
    raw = os.path.expanduser(str(path))
    if os.path.isabs(raw):
        return os.path.abspath(raw)

    repo_root = os.path.dirname(ROOT)
    candidates = []
    if config_dir:
        candidates.append(os.path.join(config_dir, raw))
    candidates.extend([
        os.path.join(ROOT, raw),
        os.path.join(repo_root, raw),
    ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def load_scene_config(config_path):
    """Load an industrial scene JSON and resolve object mesh paths."""
    resolved_config_path = _resolve_path(config_path)
    config_dir = os.path.dirname(resolved_config_path)
    with open(resolved_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    object_specs = config if isinstance(config, list) else config.get("objects", [])
    if not object_specs:
        raise ValueError(f"Scene config has no objects: {resolved_config_path}")

    resolved_specs = []
    for spec in object_specs:
        item = dict(spec)
        item["path"] = _resolve_path(item["path"], config_dir=config_dir)
        resolved_specs.append(item)

    if isinstance(config, list):
        config = {"objects": object_specs}
    config["_path"] = resolved_config_path
    config["_resolved_objects"] = resolved_specs
    return config


def select_target_object(scene, target_name=None):
    """Return (body_id, SceneObject) for the configured target object."""
    if target_name:
        body_id = scene.get_body_id_by_name(target_name)
        return body_id, scene.get_object_info(body_id)

    for body_id, obj in scene.get_object_registry().items():
        if obj.metadata.get("role") == "target":
            return body_id, obj

    body_id = scene.object_ids[0]
    return body_id, scene.get_object_info(body_id)


def crop_to_object(pc, object_points=None, margin=0.05, num_points=20000, table_z=0.005):
    """按物体点云的 xy 包围盒裁剪整幅点云，保留物体及其周围一圈桌面。

    GraspNet 需要支撑面上下文，纯物体点会生成不出抓取；但整桌点云会让网络
    把大量抓取生成在远处平坦桌面上、偏离物体。这里取物体点（z>table_z）的
    xy 范围外扩 margin，裁掉范围外的桌面点，再重采样到 num_points。

    Args:
        pc: (1, N, 3) 点云。
        margin: 物体 xy 包围盒外扩量（米）。
        num_points: 输出点数。
        table_z: 判定物体点的高度阈值。
    Returns:
        (1, num_points, 3) 裁剪并重采样后的点云。
    """
    cloud = pc[0]
    if object_points is not None and len(object_points) > 0:
        obj = np.asarray(object_points)
    else:
        obj = cloud[cloud[:, 2] > table_z]
    if len(obj) == 0:
        return pc
    lo = obj[:, :2].min(axis=0) - margin
    hi = obj[:, :2].max(axis=0) + margin
    mask = ((cloud[:, 0] >= lo[0]) & (cloud[:, 0] <= hi[0]) &
            (cloud[:, 1] >= lo[1]) & (cloud[:, 1] <= hi[1]))
    cropped = cloud[mask]
    if len(cropped) == 0:
        return pc
    idx = np.random.choice(len(cropped), num_points, replace=len(cropped) < num_points)
    return cropped[idx][np.newaxis].astype(np.float32)


def filter_grasps_to_object(gg, object_points, max_center_dist=0.04, bbox_margin=0.04):
    """Keep GraspNet candidates whose centers are close to the target object."""
    if object_points is None or len(object_points) == 0 or len(gg) == 0:
        return gg, {"enabled": False, "kept": len(gg), "total": len(gg)}

    obj = np.asarray(object_points, dtype=np.float32)
    translations = np.asarray(gg.translations, dtype=np.float32)
    lo = obj.min(axis=0) - float(bbox_margin)
    hi = obj.max(axis=0) + float(bbox_margin)
    bbox_mask = np.all((translations >= lo) & (translations <= hi), axis=1)

    # Compute nearest target-object point distance in chunks to avoid a large
    # temporary matrix when GraspNet returns many candidates.
    min_dists = np.full(len(translations), np.inf, dtype=np.float32)
    for start in range(0, len(translations), 256):
        chunk = translations[start:start + 256]
        dists = np.linalg.norm(chunk[:, None, :] - obj[None, :, :], axis=2)
        min_dists[start:start + len(chunk)] = dists.min(axis=1)

    dist_mask = min_dists <= float(max_center_dist)
    keep_mask = bbox_mask | dist_mask
    kept = int(keep_mask.sum())
    stats = {
        "enabled": True,
        "kept": kept,
        "total": int(len(gg)),
        "max_center_dist": float(max_center_dist),
        "bbox_margin": float(bbox_margin),
        "best_center_dist": float(min_dists.min()) if len(min_dists) else None,
    }
    if kept == 0:
        return gg, stats
    filtered = gg[keep_mask]
    filtered.sort_by_score()
    return filtered, stats


def parse_args():
    p = argparse.ArgumentParser(description='GraspNet + PyBullet 闭环仿真')
    p.add_argument('--obj', default='/home/admin128/beilei/obj_phase3/002/textured.obj',
                   help='物体 .obj 路径')
    p.add_argument('--scene-config', default=None,
                   help='多物体工业场景 JSON 配置；指定后优先使用配置中的 objects')
    p.add_argument('--target-object', default=None,
                   help='多物体场景中的目标物体 name；不指定时使用 metadata.role=target 或第一个物体')
    p.add_argument('--ckpt', default=None,
                   help='Checkpoint 路径 (默认自动查找)')
    p.add_argument('--top_k', type=int, default=10, help='评估前 K 个抓取')
    p.add_argument('--scale', type=float, default=1.0,
                   help='物体缩放因子（图形学单位 mesh 需缩到米制小物体尺寸）')
    p.add_argument('--gui', action='store_true', help='打开 PyBullet GUI')
    p.add_argument('--device', default='cuda:0', help='推理设备')
    p.add_argument('--output', default='results/grasp_simulation.json', help='输出文件')
    return p.parse_args()


def main():
    args = parse_args()

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    # 查找 checkpoint
    ckpt = args.ckpt
    if ckpt is None:
        candidates = [
            os.path.join(ROOT, 'checkpoints', 'checkpoint-rs.tar'),
            '/home/admin128/beilei/graspnet-baseline/checkpoints/checkpoint-rs.tar',
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt = c
                break
    if ckpt is None or not os.path.exists(ckpt):
        raise FileNotFoundError(f'Checkpoint not found. 请将 checkpoint-rs.tar 放到 {ROOT}/checkpoints/')
    print(f'[Checkpoint] {ckpt}')

    # ==============================================================
    # 1. 加载 GraspNet
    # ==============================================================
    print('[1/5] 加载 GraspNet...')
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02,
                   hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt_data['model_state_dict'])
    net.to(device)
    net.eval()
    print(f'  ✅ epoch {ckpt_data.get("epoch", "?")}')

    # ==============================================================
    # 2. 创建 PyBullet 场景
    # ==============================================================
    print('[2/5] 创建 PyBullet 场景...')
    scene = SimulationScene(gui=args.gui)
    scene.connect()
    scene.load_plane()

    scene_config = None
    if args.scene_config:
        scene_config = load_scene_config(args.scene_config)
        scene.load_objects(scene_config["_resolved_objects"])
        obj_id, target_object = select_target_object(scene, args.target_object)
        print(f'  ✅ 场景加载完成: {len(scene.object_ids)} 个物体, 目标: {target_object.name}')
    else:
        # 单物体兼容路径：保留旧的 --obj 用法，方便单模型调试。
        from scipy.spatial.transform import Rotation as R
        r = R.random()
        orientation = tuple(r.as_quat()[[0, 1, 2, 3]])

        obj_id = scene.load_object(
            args.obj,
            position=(0.3, 0.0, 0.05),
            orientation=orientation,
            scale=args.scale,
            mass=0.03,
            name="target_object",
            metadata={"role": "target"},
        )
        target_object = scene.get_object_info(obj_id)

    # 等待物体稳定
    settle_steps = int(scene_config.get("settle_steps", 300)) if scene_config else 300
    for _ in range(settle_steps):
        scene.step()
    obj_pos, obj_orn = scene.get_object_pose(obj_id)
    print(f'  ✅ 目标物体稳定, 位置: {np.round(obj_pos, 3)}')

    # ==============================================================
    # 3. 虚拟相机 + 点云
    # ==============================================================
    print('[3/5] 相机拍摄 + 点云生成...')
    camera_cfg = scene_config.get("camera", {}) if scene_config else {}
    cam = VirtualCamera(position=tuple(camera_cfg.get("position", (0.3, 0.0, 0.5))),
                        target=tuple(camera_cfg.get("target", (0.3, 0.0, 0.05))),
                        near=float(camera_cfg.get("near", 0.01)),
                        far=float(camera_cfg.get("far", 5.0)),
                        width=int(camera_cfg.get("width", 1280)),
                        height=int(camera_cfg.get("height", 720)),
                        fov=float(camera_cfg.get("fov", 60.0)))
    rgb, depth, seg = cam.capture()
    pc = cam.generate_point_cloud(depth, num_points=20000).numpy()
    object_clouds = cam.generate_object_point_clouds(depth, seg, scene.object_ids)
    object_point_counts = {body_id: int(len(pts)) for body_id, pts in object_clouds.items()}
    n_obj_pts = (pc[0, :, 2] > 0.005).sum()
    print(f'  ✅ 点云: {pc.shape}, 物体点数: {n_obj_pts}')

    # 按物体点云 xy 包围盒裁剪，裁掉远处大片桌面，只保留物体及周围一圈支撑面。
    # 参照参考流程的 crop_pointcloud（此处无 VLM，直接用物体点 bbox + margin），
    # 避免 GraspNet 在平坦桌面上生成大量落在桌面、偏离物体的抓取。
    crop_cfg = scene_config.get("crop", {}) if scene_config else {}
    target_points = object_clouds.get(int(obj_id))
    pc = crop_to_object(
        pc,
        object_points=target_points,
        margin=float(crop_cfg.get("margin", 0.05)),
        num_points=int(crop_cfg.get("num_points", 20000)),
        table_z=float(crop_cfg.get("table_z", 0.005)),
    )
    n_obj_pts = (pc[0, :, 2] > 0.005).sum()
    print(f'  ✂️  裁剪后点云: {pc.shape}, 物体点数: {n_obj_pts}')

    # ==============================================================
    # 4. GraspNet 推理
    # ==============================================================
    print('[4/5] GraspNet 推理...')
    cloud_tensor = torch.from_numpy(pc.astype(np.float32)).to(device)
    with torch.no_grad():
        end_points = net({'point_clouds': cloud_tensor})
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    gg.sort_by_score()
    print(f'  ✅ {len(gg)} 个候选, top score={gg[0].score:.4f}')

    grasp_filter_stats = {}
    if scene_config:
        filter_cfg = scene_config.get("grasp_filter", {})
        gg, grasp_filter_stats = filter_grasps_to_object(
            gg,
            target_points,
            max_center_dist=float(filter_cfg.get("max_center_dist", 0.04)),
            bbox_margin=float(filter_cfg.get("bbox_margin", 0.04)),
        )
        print(
            f'  🎯 目标过滤: {grasp_filter_stats.get("kept")}/'
            f'{grasp_filter_stats.get("total")} 个候选, '
            f'最近中心距={grasp_filter_stats.get("best_center_dist")}'
        )

    # ==============================================================
    # 5. 物理仿真评估
    # ==============================================================
    print(f'[5/5] 仿真评估 Top-{args.top_k}...')
    # JAKA Zu3 + Robotiq-85，纯 PyBullet IK（不启用 MoveIt）
    gripper = JakaZu3Robotiq85Gripper(planner=None)
    gripper.load()
    evaluator = GraspEvaluator(object_id=obj_id, gripper=gripper,
                               point_cloud=pc[0], gui=args.gui)
    results = evaluator.evaluate(gg, top_k=args.top_k)

    n_ok = sum(1 for r in results if r['success'])
    for i, r in enumerate(results):
        status = '✅' if r['success'] else '❌'
        print(f'  {status} Grasp {i}: score={r["score"]:.3f}, '
              f'lift_z={r["lift_z"]:.3f}m, w={r["width"]:.4f}m')

    print(f'\n📊 结果: {n_ok}/{len(results)} 成功')

    # 构建结果
    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    out = {
        'total': len(results),
        'success': n_ok,
        'obj_path': target_object.path,
        'scene_config': scene_config.get("_path") if scene_config else None,
        'target_object_name': target_object.name,
        'target_body_id': int(obj_id),
        'objects': scene.get_object_poses(),
        'object_point_counts': object_point_counts,
        'grasp_filter': grasp_filter_stats,
        'object_position': list(obj_pos),
        'object_orientation': list(obj_orn),
        'gripper': gripper.metadata(),
        'grasps': [],
    }
    for r in results:
        g = {}
        for k, v in r.items():
            if k == 'frame_log':
                g[k] = v
            elif isinstance(v, np.ndarray):
                g[k] = v.tolist()
            elif isinstance(v, (np.bool_, bool)):
                g[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                g[k] = int(v)
            elif isinstance(v, (np.floating,)):
                g[k] = float(v)
            else:
                g[k] = v
        out['grasps'].append(g)

    # 递归安全序列化
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    out = _json_safe(out)
    json.dump(out, open(args.output, 'w'), indent=2, ensure_ascii=False)
    print(f'💾 结果已保存: {args.output}')

    # 保存 viz_data.pkl — 确保 flush 避免 stdout 混入
    import sys as _sys; _sys.stdout.flush(); _sys.stderr.flush()
    viz_path = args.output.replace('.json', '_viz_data.pkl')
    with open(viz_path, 'wb') as _f:
        pickle.dump({'rgb': rgb, 'depth': depth, 'point_cloud': pc,
                     'seg': seg, 'object_point_counts': object_point_counts,
                     'objects': scene.get_object_poses(),
                     'grasp_filter': grasp_filter_stats,
                     'target_body_id': int(obj_id),
                     'target_object_name': target_object.name,
                     'scene_config': scene_config.get("_path") if scene_config else None,
                     'obj_path': target_object.path,
                     'object_orientation': list(obj_orn)}, _f)
    _sys.stdout.flush()
    print(f'💾 可视化数据已保存: {viz_path}')

    gripper.remove()
    scene.disconnect()


if __name__ == '__main__':
    main()
