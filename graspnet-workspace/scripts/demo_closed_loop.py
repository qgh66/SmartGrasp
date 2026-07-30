#!/usr/bin/env python
"""
GraspNet + PyBullet 闭环仿真 Demo。

流程: PyBullet 场景 → 虚拟相机 → 点云 → GraspNet 推理 → 抓取执行 → 评估

用法:
  conda activate smartgrasp
  cd /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace
  python scripts/demo_closed_loop.py
"""

import sys, os, json, argparse, pickle, random, atexit, numpy as np

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
from simulation.gripper_factory import GRIPPER_MODELS, create_gripper
from simulation.evaluator import GraspEvaluator
from simulation.object_mapping import (
    decode_body_ids,
    load_object_mask,
    match_scene_object_by_mask,
)
from simulation.perception_input import (
    export_perception_input,
    generate_capture_scene_id,
    run_perception_for_scene,
    run_pipeline_for_scene,
)
from utils.collision_detector import ModelFreeCollisionDetector

MAX_GRIPPER_OPENING = 0.085


def load_graspnet_model(checkpoint_path, device):
    """Load GraspNet only when its GPU memory is actually needed."""
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()
    return net, checkpoint.get("epoch", "?")


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


def point_cloud_from_reason_part_mask(
    camera,
    depth,
    segmentation,
    body_id,
    mask_path,
):
    """Back-project Reason's selected part mask, restricted to one body ID."""
    body_ids = decode_body_ids(segmentation)
    part_mask, diagnostics = load_object_mask(
        mask_path,
        target_shape=body_ids.shape,
    )
    body_mask = body_ids == int(body_id)
    region_mask = part_mask & body_mask
    region_pixels = int(np.count_nonzero(region_mask))
    if region_pixels == 0:
        raise RuntimeError(
            "Reason part mask does not overlap the selected PyBullet body: "
            f"body_id={int(body_id)}, mask={diagnostics['mask_path']}"
        )

    masked_depth = np.where(region_mask, depth, np.nan)
    region_points = camera.backproject_depth(masked_depth).astype(np.float32)
    body_pixels = int(np.count_nonzero(body_mask))
    diagnostics.update(
        {
            "source": "reason_part_mask",
            "selected_body_id": int(body_id),
            "body_pixels": body_pixels,
            "intersection_pixels": region_pixels,
            "reference_coverage": float(
                region_pixels / diagnostics["mask_pixels"]
            ),
            "body_coverage": float(
                region_pixels / body_pixels
            ) if body_pixels else 0.0,
            "point_count": int(len(region_points)),
        }
    )
    return region_points, diagnostics


def filter_grasps_to_object(
    gg,
    object_points,
    max_center_dist=0.04,
    bbox_margin=0.04,
    min_inner_points=5,
    enforce_center_distance=False,
):
    """Keep candidates whose closing region contains visible target points."""
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

    inner_counts = np.zeros(len(gg), dtype=np.int32)
    for index, grasp in enumerate(gg):
        local = (obj - grasp.translation) @ grasp.rotation_matrix
        within_height = np.abs(local[:, 2]) <= float(grasp.height) / 2.0
        within_depth = (
            (local[:, 0] >= float(grasp.depth) - 0.06)
            & (local[:, 0] <= float(grasp.depth))
        )
        executable_width = min(float(grasp.width), MAX_GRIPPER_OPENING)
        between_fingers = np.abs(local[:, 1]) <= executable_width / 2.0
        inner_counts[index] = int(np.count_nonzero(within_height & within_depth & between_fingers))

    inner_mask = inner_counts >= int(min_inner_points)
    center_mask = (
        min_dists <= float(max_center_dist)
        if enforce_center_distance
        else np.ones(len(gg), dtype=bool)
    )
    keep_mask = bbox_mask & inner_mask & center_mask
    kept = int(keep_mask.sum())
    stats = {
        "enabled": True,
        "kept": kept,
        "total": int(len(gg)),
        "max_center_dist": float(max_center_dist),
        "bbox_margin": float(bbox_margin),
        "min_inner_points": int(min_inner_points),
        "center_distance_enforced": bool(enforce_center_distance),
        "best_inner_point_count": int(inner_counts.max()) if len(inner_counts) else 0,
        "best_center_dist": float(min_dists.min()) if len(min_dists) else None,
    }
    filtered = gg[keep_mask]
    filtered.sort_by_score()
    return filtered, stats


def filter_collision_grasps(gg, scene_points, config):
    """Remove candidates whose fingers or approach path intersect the scene."""
    stats = {"enabled": bool(config.get("enabled", True)), "total": int(len(gg))}
    if not stats["enabled"] or len(gg) == 0:
        stats.update({"kept": int(len(gg)), "reason": "disabled_or_no_candidates"})
        return gg, stats

    points = np.asarray(scene_points, dtype=np.float32)
    detector = ModelFreeCollisionDetector(
        points,
        voxel_size=float(config.get("voxel_size", 0.005)),
    )
    collision_mask = detector.detect(
        gg,
        approach_dist=float(config.get("approach_dist", 0.05)),
        collision_thresh=float(config.get("collision_thresh", 0.05)),
    )
    keep_mask = ~np.asarray(collision_mask, dtype=bool)
    filtered = gg[keep_mask]
    filtered.sort_by_score()
    stats.update({
        "kept": int(len(filtered)),
        "removed": int((~keep_mask).sum()),
        "voxel_size": float(config.get("voxel_size", 0.005)),
        "approach_dist": float(config.get("approach_dist", 0.05)),
        "collision_thresh": float(config.get("collision_thresh", 0.05)),
    })
    return filtered, stats


def prefer_topdown_grasps(gg, config):
    """Keep safe downward approaches and rank them by GraspNet score."""
    stats = {"enabled": bool(config.get("enabled", True)), "total": int(len(gg))}
    if not stats["enabled"] or len(gg) == 0:
        stats.update({"kept": int(len(gg)), "reason": "disabled_or_no_candidates"})
        return gg, stats

    approach_axes = np.asarray(gg.rotation_matrices, dtype=np.float32)[:, :, 0]
    downward_cosines = approach_axes @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    max_angle_deg = float(config.get("max_angle_deg", 60.0))
    safe_mask = downward_cosines >= np.cos(np.deg2rad(max_angle_deg))
    if safe_mask.any():
        safe = gg[safe_mask]
        # The angle threshold is a safety gate, not a grasp-quality score.
        # Ranking by angle first made nearly vertical low-quality grasps displace
        # substantially better candidates from a small top-k evaluation.
        preferred = safe
        preferred.sort_by_score()
        stats.update({"kept": int(len(preferred)), "fallback": False})
    else:
        preferred = gg
        preferred.sort_by_score()
        stats.update({"kept": int(len(preferred)), "fallback": True})
    stats["max_angle_deg"] = max_angle_deg
    return preferred, stats


def parse_args():
    p = argparse.ArgumentParser(description='GraspNet + PyBullet 闭环仿真')
    p.add_argument('--obj', default='assets/objects/industrial_tools/ycb/050_medium_clamp/google_16k/textured.obj',
                   help='物体 .obj 路径')
    p.add_argument('--scene-config', default=None,
                   help='多物体工业场景 JSON 配置；指定后优先使用配置中的 objects')
    p.add_argument('--target-object', default=None,
                   help='多物体场景中的目标物体 name；不指定时使用 metadata.role=target 或第一个物体')
    p.add_argument('--target-mask', default=None,
                   help='Reason Object ID 对应的 Perception 整物体 mask；按可见 segmentation 最大 IoU 匹配目标')
    p.add_argument('--target-mask-min-iou', type=float, default=0.01,
                   help='Perception mask 与 PyBullet segmentation 匹配的最小 IoU')
    p.add_argument('--scene-id', type=int, default=None,
                   help='手工指定本次拍照 ID；不指定时按 1、2、3... 自动递增')
    p.add_argument('--instruction', default=None,
                   help='用户指令；提供后自动为本次拍照生成 ID 并保存 Perception 输入')
    p.add_argument('--run-perception-after-capture', action='store_true',
                   help='导出同帧输入后，把本次拍照 ID 传给 perception/run_perception.sh')
    p.add_argument('--run-pipeline-after-capture', action='store_true',
                   help='导出同帧输入后运行 Perception+Intent+Reason，并将 Reason Object ID 映射为当前 PyBullet 目标')
    p.add_argument('--task-closed-loop', action='store_true',
                   help='语义目标物理闭环：遮挡动作后重新拍摄和推理，只有最终目标抓取成功才结束')
    p.add_argument(
        '--occlusion-action',
        choices=['auto', 'push', 'grasp-away'],
        default='auto',
        help='闭环中的遮挡物处理：自动选择、推动，或抓走后松爪（默认: auto）',
    )
    p.add_argument('--max-task-rounds', type=int, default=6,
                   help='语义目标闭环最大感知-动作轮数（默认: 6）')
    p.add_argument('--push-distance', type=float, default=0.05,
                   help='推动遮挡物的水平距离，单位米（默认: 0.05）')
    p.add_argument('--push-direction', type=float, nargs=3, default=None,
                   metavar=('X', 'Y', 'Z'),
                   help='推动方向；默认根据物体相对工作区中心自动向外选择')
    p.add_argument('--reobserve-settle-steps', type=int, default=120,
                   help='中间动作后、重新拍摄前等待的仿真步数（默认: 120）')
    p.add_argument('--use-reason-part-mask', action='store_true',
                   help='完整 Pipeline 模式下使用 Reason 选择的 part mask 聚焦 GraspNet 裁剪和候选过滤；默认仍使用整物体点云')
    p.add_argument('--target-objects', nargs='+', default=None,
                   help='顺序抓取目标名称列表，例如 battery flat_screwdriver')
    p.add_argument('--all-objects', action='store_true',
                   help='启用顺序抓取与搬运流程；配合 --target-objects 指定目标及顺序')
    p.add_argument('--continuous-grasp', action='store_true',
                   help='连续清场模式：每次成功投放并松爪后重新拍摄，重试先前被遮挡的剩余物体')
    p.add_argument('--drop-after-grasp', action='store_true',
                   help='批量模式下搬运到配置的投放位姿后松开夹爪，让物体自然落下；连续清场模式默认启用')
    p.add_argument('--drop-settle-steps', type=int, default=None,
                   help='松爪后等待物体下落的仿真步数；默认读取场景 continuous_grasp.drop_settle_steps')
    p.add_argument('--max-stalled-passes', type=int, default=None,
                   help='连续清场连续多少轮无成功后停止；默认读取场景 continuous_grasp.max_stalled_passes')
    p.add_argument('--ckpt', default=None,
                   help='Checkpoint 路径 (默认自动查找)')
    p.add_argument('--top_k', type=int, default=10, help='评估前 K 个抓取')
    p.add_argument('--test-all-candidates', action='store_true',
                   help='物理评估所有过滤后候选，而不是只评估 top_k')
    p.add_argument('--test-all-raw-candidates', action='store_true',
                   help='诊断模式：跳过候选过滤并物理评估全部原始候选')
    p.add_argument('--stop-on-success', action='store_true',
                   help='依次评估全部过滤后候选，找到首个真实成功抓取后停止')
    p.add_argument('--assisted-grasp', action='store_true',
                   help='确认目标被夹住后建立高保持力约束，运输到目标位姿后保持吸附')
    p.add_argument('--seed', type=int, default=1,
                   help='NumPy/PyTorch 随机种子，用于复现实验（默认: 1）')
    p.add_argument('--gui-speed', type=float, default=1.0,
                   help='GUI 动画速度倍率，越小越慢（默认: 1）')
    p.add_argument('--initial-pose-hold-seconds', type=float, default=3.0,
                   help='全部抓取开始前保持初始关节位姿的秒数（默认: 3）')
    p.add_argument('--max-candidates-per-object', type=int, default=30,
                   help='批量模式每个物体最多执行的候选数（默认: 30）')
    p.add_argument('--gripper-model', choices=GRIPPER_MODELS, default='robotiq85',
                   help='执行夹爪模型；默认 robotiq85，稳定简化模型为 box_parallel')
    p.add_argument('--scale', type=float, default=1.0,
                   help='物体缩放因子（图形学单位 mesh 需缩到米制小物体尺寸）')
    p.add_argument('--gui', action='store_true', help='打开 PyBullet GUI')
    p.add_argument('--record-video', action='store_true',
                   help='录制原生 PyBullet GUI 窗口动画为 MP4（需要 --gui）')
    p.add_argument('--video-output', default=None,
                   help='PyBullet GUI MP4 输出路径')
    p.add_argument('--device', default='cuda:0', help='推理设备')
    p.add_argument('--output', default='results/grasp_simulation.json', help='输出文件')
    return p.parse_args()


def main():
    args = parse_args()

    instruction = str(args.instruction or "").strip()
    if args.scene_id is not None and not instruction:
        raise ValueError("--scene-id requires a non-empty --instruction")
    if args.run_perception_after_capture and not instruction:
        raise ValueError(
            "--run-perception-after-capture requires a non-empty --instruction"
        )
    if args.run_pipeline_after_capture and not instruction:
        raise ValueError(
            "--run-pipeline-after-capture requires a non-empty --instruction"
        )
    if args.task_closed_loop and not args.run_pipeline_after_capture:
        raise ValueError(
            "--task-closed-loop requires --run-pipeline-after-capture"
        )
    if args.task_closed_loop and not args.scene_config:
        raise ValueError("--task-closed-loop requires --scene-config")
    if args.max_task_rounds <= 0:
        raise ValueError("--max-task-rounds must be greater than zero")
    if args.push_distance <= 0:
        raise ValueError("--push-distance must be greater than zero")
    if args.reobserve_settle_steps < 0:
        raise ValueError("--reobserve-settle-steps must be non-negative")
    if args.drop_settle_steps is not None and args.drop_settle_steps < 0:
        raise ValueError("--drop-settle-steps must be non-negative")
    if args.push_direction is not None and np.linalg.norm(args.push_direction) < 1e-8:
        raise ValueError("--push-direction must be a non-zero vector")
    if args.run_perception_after_capture and args.run_pipeline_after_capture:
        raise ValueError(
            "--run-perception-after-capture and "
            "--run-pipeline-after-capture are mutually exclusive"
        )
    if args.run_pipeline_after_capture and args.target_mask:
        raise ValueError(
            "--run-pipeline-after-capture resolves its own target mask and "
            "cannot be combined with --target-mask"
        )
    if args.run_pipeline_after_capture and args.target_object:
        raise ValueError(
            "--run-pipeline-after-capture resolves its own target object and "
            "cannot be combined with --target-object"
        )
    if args.run_pipeline_after_capture and args.all_objects:
        raise ValueError(
            "--run-pipeline-after-capture is currently a single-target flow "
            "and cannot be combined with --all-objects"
        )
    if args.run_pipeline_after_capture and args.continuous_grasp:
        raise ValueError(
            "--run-pipeline-after-capture is currently a single-target flow "
            "and cannot be combined with --continuous-grasp"
        )
    if args.all_objects and args.continuous_grasp:
        raise ValueError(
            "--all-objects and --continuous-grasp are independent batch modes"
        )
    if args.drop_after_grasp and not (
        args.all_objects or args.continuous_grasp
    ):
        raise ValueError(
            "--drop-after-grasp requires --all-objects or "
            "--continuous-grasp"
        )
    if args.use_reason_part_mask and not args.run_pipeline_after_capture:
        raise ValueError(
            "--use-reason-part-mask requires --run-pipeline-after-capture"
        )

    if args.all_objects or args.continuous_grasp:
        from demo_all_objects import run_all_objects
        return run_all_objects(args)
    if args.task_closed_loop:
        from demo_task_closed_loop import run_task_closed_loop
        return run_task_closed_loop(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f'[Seed] {args.seed}')

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    # 查找 checkpoint
    ckpt = args.ckpt
    if ckpt is None:
        candidates = [os.path.join(ROOT, 'checkpoints', 'checkpoint-rs.tar')]
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
    if args.run_pipeline_after_capture:
        net = None
        print('  ⏸️ 完整 Pipeline 模式：延后加载，避免与 SAM2 叠加显存')
    else:
        net, epoch = load_graspnet_model(ckpt, device)
        print(f'  ✅ epoch {epoch}')

    # ==============================================================
    # 2. 创建 PyBullet 场景
    # ==============================================================
    print('[2/5] 创建 PyBullet 场景...')
    scene = SimulationScene(gui=args.gui)
    scene.connect()
    scene.load_plane()

    scene_config = None
    staging_enabled = False
    target_selection = None
    if args.scene_config:
        scene_config = load_scene_config(args.scene_config)
        scene.load_objects(scene_config["_resolved_objects"])
        staging_enabled = bool(
            scene_config.get("object_staging", {}).get(
                "lock_initial_poses_until_grasp",
                False,
            )
        )
        if staging_enabled:
            scene.stage_objects_at_initial_poses()
            print(
                "  🔒 初始堆叠已暂时锁定；目标进入抓取评估前恢复动态质量"
            )
        if args.target_mask or args.run_pipeline_after_capture:
            obj_id = None
            target_object = None
            print(
                f'  ✅ 场景加载完成: {len(scene.object_ids)} 个物体, '
                '目标等待 Perception/Reason mask 映射'
            )
        else:
            obj_id, target_object = select_target_object(scene, args.target_object)
            target_selection = {
                "source": "object_name_or_scene_default",
                "selected_body_id": int(obj_id),
                "selected_object_name": target_object.name,
            }
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
        target_selection = {
            "source": "single_object_scene",
            "selected_body_id": int(obj_id),
            "selected_object_name": target_object.name,
        }

    # 等待物体稳定
    settle_steps = int(scene_config.get("settle_steps", 300)) if scene_config else 300
    for _ in range(settle_steps):
        scene.step()
    if obj_id is not None:
        obj_pos, obj_orn = scene.get_object_pose(obj_id)
        print(f'  ✅ 目标物体稳定, 位置: {np.round(obj_pos, 3)}')
    else:
        print('  ✅ 场景稳定，等待相机 segmentation 完成目标映射')

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
    perception_input = None
    perception_output_dir = None
    reason_output_dir = None
    reason_target = None
    capture_scene_id = args.scene_id
    if instruction and capture_scene_id is None:
        capture_scene_id = generate_capture_scene_id()
    if capture_scene_id is not None:
        perception_input = export_perception_input(
            scene_id=capture_scene_id,
            rgb=rgb,
            depth=depth,
            segmentation=seg,
            instruction=instruction,
        )
        print(
            f'  💾 Perception 同帧输入: scene_id={capture_scene_id} '
            f'-> {perception_input["input_dir"]}'
        )
        if args.run_perception_after_capture:
            perception_output_dir = run_perception_for_scene(
                capture_scene_id,
                run_reason=False,
            )
            print(
                f'  ✅ Perception 完成: scene_id={capture_scene_id} '
                f'-> {perception_output_dir}'
            )
        elif args.run_pipeline_after_capture:
            reason_target = run_pipeline_for_scene(capture_scene_id)
            perception_output_dir = reason_target["perception_output_dir"]
            reason_output_dir = reason_target["reason_output_dir"]
            print(
                f'  ✅ Perception + Intent + Reason 完成: '
                f'scene_id={capture_scene_id}, '
                f'branch={reason_target["branch"]}, '
                f'object_id={reason_target["object_id"]}, '
                f'label={reason_target["object_label"]}'
            )
    pc = cam.generate_point_cloud(depth, num_points=20000).numpy()
    object_clouds = cam.generate_object_point_clouds(depth, seg, scene.object_ids)
    if reason_target is not None:
        obj_id, target_object, target_selection = match_scene_object_by_mask(
            scene,
            seg,
            reason_target["object_mask_path"],
            minimum_iou=args.target_mask_min_iou,
        )
        target_selection.update(
            {
                "reason_scene_id": int(reason_target["scene_id"]),
                "reason_branch": reason_target["branch"],
                "reason_object_id": int(reason_target["object_id"]),
                "reason_object_label": reason_target["object_label"],
                "reason_summary_path": reason_target["reason_summary_path"],
                "occlusion_graph_path": reason_target[
                    "occlusion_graph_path"
                ],
            }
        )
        obj_pos, obj_orn = scene.get_object_pose(obj_id)
        print(
            f'  🔗 Reason Object {reason_target["object_id"]} '
            f'-> PyBullet body_id={obj_id}, '
            f'name={target_object.name}, '
            f'IoU={target_selection["selected_iou"]:.4f}'
        )
    elif args.target_mask:
        obj_id, target_object, target_selection = match_scene_object_by_mask(
            scene,
            seg,
            args.target_mask,
            minimum_iou=args.target_mask_min_iou,
        )
        obj_pos, obj_orn = scene.get_object_pose(obj_id)
        print(
            f'  🔗 Reason Object Mask -> body_id={obj_id}, '
            f'name={target_object.name}, '
            f'IoU={target_selection["selected_iou"]:.4f}'
        )
    object_point_counts = {body_id: int(len(pts)) for body_id, pts in object_clouds.items()}
    n_obj_pts = (pc[0, :, 2] > 0.005).sum()
    print(f'  ✅ 点云: {pc.shape}, 物体点数: {n_obj_pts}')

    target_points = object_clouds.get(int(obj_id))
    grasp_region_points = target_points
    grasp_region_source = "whole_object"
    part_mask_diagnostics = None
    if args.use_reason_part_mask:
        part_mask_info = reason_target.get("grasp_part_mask") or {}
        part_mask_path = reason_target.get("grasp_part_mask_path")
        if not part_mask_path:
            raise RuntimeError(
                "Reason did not produce grasp_part_mask.path for "
                "--use-reason-part-mask"
            )
        try:
            part_object_id = int(part_mask_info.get("object_id"))
        except (TypeError, ValueError):
            part_object_id = None
        if (
            part_object_id != int(reason_target["object_id"])
            or not bool(part_mask_info.get("validated"))
        ):
            raise RuntimeError(
                "Reason part mask is not validated for the selected object: "
                f"selected_object_id={reason_target['object_id']}, "
                f"part_object_id={part_object_id}, "
                f"validated={part_mask_info.get('validated')}"
            )
        grasp_region_points, part_mask_diagnostics = (
            point_cloud_from_reason_part_mask(
                cam,
                depth,
                seg,
                obj_id,
                part_mask_path,
            )
        )
        part_mask_diagnostics["part_id"] = part_mask_info.get("part_id")
        part_mask_diagnostics["object_id"] = part_object_id
        grasp_region_source = "reason_part_mask"
        print(
            f'  🎯 Part mask 定位: part_id={part_mask_info.get("part_id")}, '
            f'points={len(grasp_region_points)}, '
            f'body_coverage={part_mask_diagnostics["body_coverage"]:.3f}'
        )
        if part_mask_diagnostics["body_coverage"] >= 0.95:
            print(
                '  ⚠️ Part mask 覆盖了至少 95% 的可见物体，'
                '本轮定位效果接近整物体 mask'
            )

    # 按抓取区域点云 xy 包围盒裁剪，裁掉远处大片桌面，只保留目标区域及支撑面。
    # 参照参考流程的 crop_pointcloud（此处无 VLM，直接用物体点 bbox + margin），
    # 避免 GraspNet 在平坦桌面上生成大量落在桌面、偏离物体的抓取。
    crop_cfg = scene_config.get("crop", {}) if scene_config else {}
    pc = crop_to_object(
        pc,
        object_points=grasp_region_points,
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
    if net is None:
        net, epoch = load_graspnet_model(ckpt, device)
        print(f'  ✅ 延后加载完成, epoch {epoch}')
    camera_points = cam.world_to_camera_points(pc[0]).astype(np.float32)
    cloud_tensor = torch.from_numpy(camera_points[np.newaxis]).to(device)
    with torch.no_grad():
        end_points = net({'point_clouds': cloud_tensor})
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    gg = cam.camera_grasps_to_world(gg)
    gg.sort_by_score()
    if len(gg) == 0:
        raise RuntimeError('GraspNet produced no grasp candidates for the target crop')
    print(f'  ✅ {len(gg)} 个候选, top score={gg[0].score:.4f}')

    raw_gg = GraspGroup(gg.grasp_group_array.copy())
    grasp_filter_stats = {}
    collision_filter_stats = {}
    topdown_filter_stats = {}
    if scene_config and not args.test_all_raw_candidates:
        filter_cfg = scene_config.get("grasp_filter", {})
        gg, grasp_filter_stats = filter_grasps_to_object(
            gg,
            grasp_region_points,
            max_center_dist=float(filter_cfg.get("max_center_dist", 0.04)),
            bbox_margin=float(filter_cfg.get("bbox_margin", 0.04)),
            min_inner_points=int(filter_cfg.get("min_inner_points", 5)),
            enforce_center_distance=args.use_reason_part_mask,
        )
        grasp_filter_stats["region_source"] = grasp_region_source
        print(
            f'  🎯 抓取区域过滤 ({grasp_region_source}): '
            f'{grasp_filter_stats.get("kept")}/'
            f'{grasp_filter_stats.get("total")} 个候选, '
            f'最近中心距={grasp_filter_stats.get("best_center_dist")}'
        )
        if args.use_reason_part_mask and len(gg) == 0:
            raise RuntimeError(
                "No GraspNet candidates overlap the selected Reason part mask"
            )
        collision_cfg = scene_config.get("collision_filter", {})
        gg, collision_filter_stats = filter_collision_grasps(gg, pc[0], collision_cfg)
        print(
            f'  collision filter: {collision_filter_stats.get("kept")}/'
            f'{collision_filter_stats.get("total")} candidates'
        )

        topdown_cfg = scene_config.get("topdown_filter", {})
        gg, topdown_filter_stats = prefer_topdown_grasps(gg, topdown_cfg)
        print(
            f'  top-down preference: {topdown_filter_stats.get("kept")} candidates, '
            f'fallback={topdown_filter_stats.get("fallback", False)}'
        )
    elif args.test_all_raw_candidates:
        gg = raw_gg
        grasp_filter_stats = {"enabled": False, "reason": "test_all_raw_candidates"}
        collision_filter_stats = {"enabled": False, "reason": "test_all_raw_candidates"}
        topdown_filter_stats = {"enabled": False, "reason": "test_all_raw_candidates"}

    # ==============================================================
    # 5. 物理仿真评估
    # ==============================================================
    evaluate_all = (
        args.test_all_candidates
        or args.test_all_raw_candidates
        or args.stop_on_success
    )
    evaluation_count = len(gg) if evaluate_all else args.top_k
    evaluation_description = (
        f'最多 {evaluation_count} 个候选（首个成功后停止）'
        if args.stop_on_success
        else f'{evaluation_count} 个候选'
    )
    print(f'[5/5] 仿真评估 {evaluation_description}...')
    # JAKA Zu3 + Robotiq-85，纯 PyBullet IK（不启用 MoveIt）
    capture_joint_pose_deg = (
        scene_config.get("capture_joint_pose_deg") if scene_config else None
    )
    place_target_joint_pose_deg = (
        scene_config.get("place_target_joint_pose_deg") if scene_config else None
    )
    gripper = create_gripper(
        args.gripper_model,
        planner=None,
        initial_joint_pose_deg=capture_joint_pose_deg,
        robot_base_yaw_deg=(
            scene_config.get("robot_base_yaw_deg", 0.0) if scene_config else 0.0
        ),
        gui_motion_step_delay=(0.003 / args.gui_speed) if args.gui else 0.0,
    )
    gripper.load()
    video_recorder = None
    if args.record_video:
        from simulation.video_recorder import PyBulletVideoRecorder
        video_path = args.video_output or args.output.replace('.json', '_pybullet.mp4')
        video_recorder = PyBulletVideoRecorder(video_path)
        video_recorder.start()
        atexit.register(video_recorder.close)
        print(f'  🎥 PyBullet GUI 录制: {video_path}')
    target_activated_from_staging = scene.activate_staged_object(obj_id)
    if target_activated_from_staging:
        print(
            f'  🔓 目标 {target_object.name} 已恢复动态质量 '
            f'({target_object.mass:.3f} kg)'
        )
    evaluator = GraspEvaluator(
        object_id=obj_id,
        gripper=gripper,
        point_cloud=grasp_region_points,
        gui=args.gui,
        assisted_grasp=args.assisted_grasp,
        # The closing-region filter already validates target occupancy and is
        # less biased than center-to-surface distance for slender objects.
        validate_target_center=not bool(scene_config),
        scene_object_ids=scene.object_ids,
        place_target_joint_pose_deg=place_target_joint_pose_deg,
        gui_speed=args.gui_speed,
    )
    results = evaluator.evaluate(
        gg,
        top_k=evaluation_count,
        stop_on_success=args.stop_on_success,
    )

    n_ok = sum(1 for r in results if r['success'])
    for i, r in enumerate(results):
        status = '✅' if r['success'] else '❌'
        placement = r.get('placement') or {}
        transport_text = (
            f', transported={placement.get("object_followed_to_place", False)}'
            if place_target_joint_pose_deg is not None else ''
        )
        failure_text = (
            f', reason={r["failure_reason"]}' if r.get('failure_reason') else ''
        )
        print(f'  {status} Grasp {i}: score={r["score"]:.3f}, '
              f'lift_delta={r.get("obj_lift_delta", 0.0):.3f}m, '
              f'w={r["width"]:.4f}m{transport_text}{failure_text}')

    print(f'\n📊 结果: {n_ok}/{len(results)} 成功')

    # 构建结果
    final_target_selected = bool(
        reason_target is None
        or reason_target.get("branch") == "fully_visible"
        or (
            reason_target.get("target_object_id") is not None
            and int(reason_target["target_object_id"])
            == int(reason_target["object_id"])
        )
    )
    task_success = bool(n_ok > 0 and final_target_selected)
    if task_success:
        task_status = "final_target_grasped"
    elif n_ok > 0:
        task_status = "intermediate_occluder_action_succeeded"
    else:
        task_status = "physical_action_failed"
    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    out = {
        'total': len(results),
        'success': n_ok,
        'task_success': task_success,
        'task_status': task_status,
        'obj_path': target_object.path,
        'scene_config': scene_config.get("_path") if scene_config else None,
        'target_object_name': target_object.name,
        'target_body_id': int(obj_id),
        'target_selection': target_selection,
        'perception_input': perception_input,
        'perception_output_dir': (
            str(perception_output_dir)
            if perception_output_dir is not None
            else None
        ),
        'reason_output_dir': (
            str(reason_output_dir)
            if reason_output_dir is not None
            else None
        ),
        'reason_target': reason_target,
        'grasp_region': {
            'source': grasp_region_source,
            'point_count': int(len(grasp_region_points)),
            'part_mask': part_mask_diagnostics,
        },
        'graspnet_input_frame': 'camera',
        'candidate_execution_frame': 'world',
        'objects': scene.get_object_poses(),
        'object_point_counts': object_point_counts,
        'grasp_filter': grasp_filter_stats,
        'collision_filter': collision_filter_stats,
        'topdown_filter': topdown_filter_stats,
        'seed': int(args.seed),
        'capture_joint_pose_deg': capture_joint_pose_deg,
        'place_target_joint_pose_deg': place_target_joint_pose_deg,
        'gui_speed': float(args.gui_speed),
        'test_all_candidates': bool(args.test_all_candidates),
        'test_all_raw_candidates': bool(args.test_all_raw_candidates),
        'use_reason_part_mask': bool(args.use_reason_part_mask),
        'stop_on_success': bool(args.stop_on_success),
        'assisted_grasp': bool(args.assisted_grasp),
        'object_staging_enabled': staging_enabled,
        'target_activated_from_staging': target_activated_from_staging,
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
                     'perception_input': perception_input,
                     'perception_output_dir': (
                         str(perception_output_dir)
                         if perception_output_dir is not None
                         else None
                     ),
                     'reason_output_dir': (
                         str(reason_output_dir)
                         if reason_output_dir is not None
                         else None
                     ),
                     'reason_target': reason_target,
                     'grasp_region': {
                         'source': grasp_region_source,
                         'point_count': int(len(grasp_region_points)),
                         'part_mask': part_mask_diagnostics,
                     },
                     'grasp_region_points': grasp_region_points,
                     'objects': scene.get_object_poses(),
                      'grasp_filter': grasp_filter_stats,
                     'collision_filter': collision_filter_stats,
                     'topdown_filter': topdown_filter_stats,
                     'seed': int(args.seed),
                     'use_reason_part_mask': bool(args.use_reason_part_mask),
                     'target_body_id': int(obj_id),
                     'target_object_name': target_object.name,
                     'scene_config': scene_config.get("_path") if scene_config else None,
                     'obj_path': target_object.path,
                     'object_orientation': list(obj_orn)}, _f)
    _sys.stdout.flush()
    print(f'💾 可视化数据已保存: {viz_path}')

    if video_recorder is not None:
        video_recorder.close()
        atexit.unregister(video_recorder.close)
        print(f'🎥 PyBullet GUI 视频已保存: {video_recorder.output_path}')
    gripper.remove()
    scene.disconnect()


if __name__ == '__main__':
    main()
