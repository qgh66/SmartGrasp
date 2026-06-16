# GraspNet Workspace

独立、可运行的 **GraspNet + PyBullet 仿真** 工作区。

---

## 目录结构

```
graspnet-workspace/
├── models/              # GraspNet 核心模型
│   ├── graspnet.py      # GraspNet 网络定义 (Stage1 + Stage2 + pred_decode)
│   ├── backbone.py      # PointNet2 backbone (特征提取)
│   ├── modules.py       # ApproachNet, CloudCrop, OperationNet, ToleranceNet
│   ├── loss.py          # 训练损失函数
│   └── graspnet_baseline.py  # 高层封装（加载 → 推理 → 碰撞检测）
│
├── pointnet2/           # PointNet++ CUDA 算子
│   ├── pointnet2_modules.py   # SA/FP 模块
│   ├── pointnet2_utils.py     # CylinderQuery, 采样等
│   ├── pytorch_utils.py       # 辅助层
│   └── _ext_src/              # CUDA 源码
│
├── knn/                 # KNN CUDA 算子
│
├── utils/               # 工具函数
│   ├── collision_detector.py  # ModelFreeCollisionDetector
│   ├── data_utils.py          # CameraInfo, create_point_cloud_from_depth_image
│   ├── label_generation.py    # 抓取标签处理
│   └── loss_utils.py          # 常量、公式
│
├── dataset/             # 数据集加载
│   └── graspnet_dataset.py    # GraspNetDataset
│
├── graspnet_api/        # graspnetAPI (GraspGroup 等)
│
├── simulation/          # PyBullet 仿真模块
│   ├── scene.py         # SimulationScene (场景管理)
│   ├── camera.py        # VirtualCamera (RGB-D 虚拟相机)
│   ├── gripper.py       # ParallelJawGripper (平行二指夹爪)
│   ├── evaluator.py     # GraspEvaluator (抓取物理评估)
│   ├── run_sim.py       # 主入口（Phase 1: 单物体仿真）
│   └── visualize.py     # 可视化工具
│
├── scripts/             # 运行脚本
│   ├── demo_inference.sh   # 纯推理 demo
│   ├── demo_simulation.sh  # 仿真 demo
│   └── train.sh            # 训练脚本
│
└── checkpoints/         # 预训练模型权重（需手动放入）
```

---

## 快速开始

### 1. 环境准备

所有程序运行于 **`smartgrasp`** conda 环境：

```bash
conda activate smartgrasp
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
```

### 2. 安装 PyBullet

```bash
conda activate smartgrasp
pip install pybullet
```

### 3. 放置 checkpoint

将训练好的 checkpoint 放入 `checkpoints/`：

```bash
cp /home/admin128/beilei/graspnet-baseline/checkpoints/checkpoint-rs.tar checkpoints/
```

---

## 运行示例

### 纯推理 Demo（无需 PyBullet）

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
bash scripts/run_inference.sh
```

### 仿真 Demo（GraspNet + PyBullet）

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
bash scripts/run_grasp_simulation.sh
```

可选参数通过环境变量传入：

```bash
OBJ_PATH=/path/to/textured.obj \
CKPT=/path/to/checkpoint-rs.tar \
DEVICE=cuda:0 \
TOP_K=5 \
OUTPUT=results_grasp_sim.json \
bash scripts/run_grasp_simulation.sh
```

### Reveal Push 仿真（默认 JAKA ZU3 可视化）

`scripts/demo_reveal_push.py` 的真实位置是：

```text
/home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace/scripts/demo_reveal_push.py
```

因此运行前必须进入 `graspnet-workspace/`。如果在 `SmartGrasp/` 根目录运行，会报：

```text
python: can't open file '/home/admin128/sangxiyuan/SmartGrasp/scripts/demo_reveal_push.py'
```

正确命令：

```bash
conda activate smartgrasp
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

bash scripts/run_reveal_push_jaka.sh
```

自定义距离和输出：

```bash
DISTANCE=0.05 \
OUTPUT=results/reveal_push_jaka.json \
bash scripts/run_reveal_push_jaka.sh
```

当前默认 `--robot-model jaka`，会使用本机已有的 JAKA ZU3 + Robotiq URDF：

```text
/home/admin128/Desktop/liboyan/Trans_MP/lby_moveit/src/robotiq_test/config/gazebo_jaka_zu3_robotiq.urdf
```

这不是从网络下载的模型。代码会在运行时把 URDF 里的 `package://...` mesh 路径映射到本机已有 mesh。

当前设计：

- JAKA ZU3 + Robotiq 用于网页回放可视化和 IK 跟随。
- 原简化夹爪碰撞体仍负责实际 push 接触，保证 Reveal push 验证稳定。
- 结果文件的 `frame_log` 会包含 `robot_model=jaka_zu3_robotiq` 和 `robot_links`，GUI 会画出 JAKA 关节骨架和 TCP。

如需回退到原简化夹爪：

```bash
bash scripts/run_reveal_push_simple.sh
```

使用真实 RGB-D + mask 数据：

```bash
RGB=/path/to/aligned_rgb.jpg \
DEPTH=/path/to/aligned_depth.npy \
MASK=/path/to/object_mask.png \
INTRINSICS=/path/to/camera_intrinsics.json \
OUTPUT=results/reveal_push_real_rgbd.json \
bash scripts/run_reveal_push_real_rgbd.sh
```

启动网页查看 JAKA push 回放：

```bash
bash scripts/run_reveal_push_gui.sh
```

指定结果文件和端口：

```bash
RESULTS=results/reveal_push_jaka.json \
VIZ_DATA=results/reveal_push_jaka_viz_data.pkl \
PORT=8051 \
bash scripts/run_reveal_push_gui.sh
```

---

## GraspNet 输入/输出说明

### 输入

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `point_clouds` | `torch.FloatTensor` | `(B, N, 3)` | B=1, N=20000 点云 |

### 输出（pred_decode 后)

| 字段 | 形状 | 说明 |
|---|---|---|
| `grasp_score` | `(K,)` | 抓取得分 (0~1) |
| `grasp_width` | `(K,)` | 夹爪宽度 (米) |
| `grasp_height` | `(K,)` | 夹爪高度 (米) |
| `grasp_depth` | `(K,)` | 抓取深度 (米) |
| `rotation_matrix` | `(K, 9)` | 旋转矩阵 (row-major) |
| `translation` | `(K, 3)` | 抓取中心位置 |
| `obj_ids` | `(K,)` | 物体 ID (-1=未知) |

K 为候选抓取数量（通常数百个）。

### 坐标约定

- 抓取坐标系: **local X 轴 = 接近/depth 方向**，local Y 轴 = 夹爪开合方向，local Z 轴 = 夹爪高度方向
- 世界坐标系: Z 向上（高度），Y 向前，X 向右
- 坐标系变换: GUI 和 evaluator 使用 `R @ local_vertices + center`，其中 local X 是 GraspNet 官方夹爪几何的 depth/approach 轴

---

## 关键模块说明

### `GraspNet` 网络 (graspnet.py)

```
Stage1: PointNet2 → ApproachNet
  - 输出: objectness (前景/背景) + 每点 300 个 viewpoint 得分
  - 选取每点得分最高的 viewpoint 作为 approach vector

Stage2: CloudCrop → OperationNet + ToleranceNet
  - CloudCrop: 按 viewpoint 对点云做圆柱体分组
  - OperationNet: 预测 12 个角度 × 4 个深度的抓取得分
  - ToleranceNet: 预测抓取容差

pred_decode: 解析网络输出 → GraspGroup (抓取位姿 + 几何参数)
```

### `ModelFreeCollisionDetector` (collision_detector.py)

- 体素化点云，判断抓取区域是否有足够的空白空间
- 如果抓取区域内被占据的体素比例过大 → 标记为碰撞

### `VirtualCamera` (simulation/camera.py)

- 在 PyBullet 场景中拍摄 RGB-D 图像
- 将深度图反投影为三维点云
- 支持 ER_TINY_RENDERER (CPU 渲染, DIRECT 模式可用)

### `GraspEvaluator` (simulation/evaluator.py)

抓取执行五个步骤:
1. 夹爪移动到接近位置（后退 offset）
2. 沿接近方向前进到抓取位置
3. 闭合手指 + 创建固定约束
4. 向上提升（检查物体是否跟随）
5. 判定成功/失败（物体 Z > 0.10m）

### 启动jaka真实模型的仿真：
```
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

conda run --no-capture-output -n smartgrasp python gui/app.py \
  --host 0.0.0.0 \
  --port 8051 \
  --results results/reveal_push_jaka.json \
  --viz-data results/reveal_push_jaka_viz_data.pkl
```