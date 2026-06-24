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


def crop_to_object(pc, margin=0.05, num_points=20000, table_z=0.005):
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


def parse_args():
    p = argparse.ArgumentParser(description='GraspNet + PyBullet 闭环仿真')
    p.add_argument('--obj', default='/home/admin128/beilei/obj_phase3/002/textured.obj',
                   help='物体 .obj 路径')
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

    # 随机朝向
    from scipy.spatial.transform import Rotation as R
    r = R.random()
    orientation = tuple(r.as_quat()[[0, 1, 2, 3]])

    obj_id = scene.load_object(args.obj, position=(0.3, 0.0, 0.05),
                                orientation=orientation, scale=args.scale, mass=0.03)
    # 等待物体稳定
    for _ in range(300):
        scene.step()
    obj_pos, obj_orn = scene.get_object_pose(obj_id)
    print(f'  ✅ 物体加载完成, 位置: {np.round(obj_pos, 3)}')

    # ==============================================================
    # 3. 虚拟相机 + 点云
    # ==============================================================
    print('[3/5] 相机拍摄 + 点云生成...')
    cam = VirtualCamera(position=(0.3, 0.0, 0.5), target=(0.3, 0.0, 0.05),
                        near=0.01, far=5.0)
    rgb, depth, _ = cam.capture()
    pc = cam.generate_point_cloud(depth, num_points=20000).numpy()
    n_obj_pts = (pc[0, :, 2] > 0.005).sum()
    print(f'  ✅ 点云: {pc.shape}, 物体点数: {n_obj_pts}')

    # 按物体点云 xy 包围盒裁剪，裁掉远处大片桌面，只保留物体及周围一圈支撑面。
    # 参照参考流程的 crop_pointcloud（此处无 VLM，直接用物体点 bbox + margin），
    # 避免 GraspNet 在平坦桌面上生成大量落在桌面、偏离物体的抓取。
    pc = crop_to_object(pc, margin=0.05, num_points=20000)
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
        'obj_path': args.obj,
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
                     'obj_path': args.obj, 'object_orientation': list(obj_orn)}, _f)
    _sys.stdout.flush()
    print(f'💾 可视化数据已保存: {viz_path}')

    gripper.remove()
    scene.disconnect()


if __name__ == '__main__':
    main()
