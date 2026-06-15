# SmartGrasp - Execution Module (执行模块)

本目录包含了 SmartGrasp 视觉语言任务导向抓取系统（Vision-Language Task-Oriented Grasping）的核心**执行模块（Execution Module）**。

执行模块作为机器人的“算法大脑”，负责接收上游感知（Perception）与推理（Reasoning）模块传来的信号与目标/遮挡物 ID。本模块会根据场景的遮挡状态，精准调用底层 3D 视觉算法与 GraspNet 大模型，最终输出物理机械臂可直接执行的 6-DoF 控制指令。

Reveal Push 仿真的实现、数据来源、判定标准和当前限制见
[`PUSH_SIMULATION_SUMMARY.md`](PUSH_SIMULATION_SUMMARY.md)。

---

## 📂 目录结构与文件说明 (Directory Structure)

`execution/`
* `execution_api.py` : [核心API] 完全可见分支：目标物体的 6D 抓取
* `reveal_api.py` : [核心API] 遮挡排雷分支：障碍物的微动推开与闭环触发
* `grasp_generator.py` : [算法层] GraspNet 大模型调用与最优位姿筛选计算
* `pointcloud_utils.py` : [工具层] 3D 点云处理（RGB-D + Mask 转换局部点云）
* `test_pipeline.py` : [测试层] 单目标/Mock 数据管线联调测试脚本
* `batch_test.py` : [测试层] 批量自动化验证脚本（遍历测试真实场景数据）
* `results/` : [输出层] 存放抓取测试生成的位姿结果与日志
* `__pycache__/` : Python 编译缓存目录

---

## 🧠 核心功能模块 (Core Modules)

本模块严格按照《SmartGrasp 项目架构》分为两大执行分支，以应对不同的场景复杂度：

### 1. 抓取模块 (Fully Visible 分支) -> `execution_api.py`
* **适用场景**：当目标物体完全可见（处于有向遮挡关系图的最顶层）时调用。
* **工作流**：
  1. 接收上游传来的 Target ID 和 2D Mask。
  2. 调用 `pointcloud_utils.py`，结合相机内参（Intrinsics）将深度图与 Mask 反投影为干净的**局部 3D 点云**。
  3. 调用 `grasp_generator.py`（封装了预训练的 GraspNet-1Billion 网络），生成 1024 个候选姿态。
  4. 根据物理稳定性（Force-closure 得分等）筛选出 **Top-1 最优 6-DoF 抓取位姿**（包含 Rotation, Translation, Width）。

### 2. 微动探测模块 (Occluded 分支) -> `reveal_api.py`
* **适用场景**：当目标物体被部分遮挡（Partially Occluded）或完全不可见（Fully Occluded）时调用。
* **工作流**：
  1. 接收推理模块基于信息增益（InfoGain）算出的“得分最高的遮挡物 ID”（挡路石）及其 3D 几何中心点。
  2. 采用极简的数学控制策略替代复杂大模型：输出默认垂直向下的夹爪姿态，并针对遮挡物在 X 轴执行 3-5 厘米的**轻微拨动（push）**或**小幅抓放（pick_and_place）**，以维持复杂堆叠场景的物理稳定性。
  3. 动作生成后，向全局输出 `request_reloop: True` 信号，触发系统重新拍摄 RGB-D 照片进入下一轮闭环（RE-LOOPS）。

---

## 🛠️ 测试与验证 (Testing & Validation)

在不依赖外部 PyBullet 物理仿真环境的情况下，您可以直接在本目录下对算法的大脑逻辑进行独立闭环测试。

**环境要求**：请确保处于 `smartgrasp` Conda 虚拟环境中。

### 1. 运行单步管线测试
测试基于伪造数据（Mock RGB-D & Mask）的 GraspNet 调用流程：
`python test_pipeline.py`
*(成功后会在终端输出被筛选出的最优 6D 抓取位姿参数)*

### 2. 运行批量验证测试
测试多张真实物体 Mask 在算法管线中的鲁棒性：
`python batch_test.py`
*(脚本会自动遍历处理多组掩码数据，结果保存在 `results/` 目录下)*

### 3. 测试微动探测逻辑
测试排雷推开坐标计算与闭环信号生成：
`python reveal_api.py`

### 4. 运行 Reveal Push 物理仿真
将 `reveal_api.py` 生成的 5 cm +X 扰动计划交给 PyBullet 执行，并保存
Dash GUI 可回放的逐帧轨迹：

```bash
cd ../graspnet-workspace
python scripts/demo_reveal_push.py \
  --distance 0.05 \
  --output results/reveal_push.json
```

该脚本默认使用 PyBullet 的小方块。使用真实遮挡物模型时传入
`--obj /path/to/object.obj`。

### 5. 使用真实 RGB-D 数据运行 Push 仿真

真实数据模式要求四个严格对齐的输入：

- RGB 图像
- 原始深度图（PNG/TIFF 或 `.npy`）
- 要推动物体的全分辨率二值 Mask
- 相机内参 JSON，包含 `fx`、`fy`、`cx`、`cy`

#### 第一步：使用 RealSense 采集同一帧 RGB-D

`smartgrasp` 环境没有安装 `pyrealsense2`，本机的 `calib` 环境可以直接采集：

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

conda run -n calib python scripts/capture_realsense_rgbd.py \
  --output-dir real_rgbd_capture \
  --prefix push_scene
```

该命令一次性保存：

```text
real_rgbd_capture/push_scene_rgb.png
real_rgbd_capture/push_scene_depth.npy
real_rgbd_capture/push_scene_depth.png
real_rgbd_capture/push_scene_intrinsics.json
real_rgbd_capture/push_scene_capture.json
```

其中深度已经由 RealSense SDK 对齐到彩色图。`capture.json` 中的
`depth_scale_for_demo` 是后续命令需要使用的 `--depth-scale`，D435
通常为 `1000`。

#### 第二步：从刚采集的 RGB 生成要推动物体的 Mask

必须对 `push_scene_rgb.png` 本身做分割，不能复用其他时刻图片的 Mask。
例如当前 Grounded-SAM-2 的 `mysteps.py` 会输出模具 Mask：

```bash
cd /home/admin128/Gsam2/Grounded-SAM-2

python mysteps.py \
  --input-path /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace/real_rgbd_capture/push_scene_rgb.png \
  --output-dir /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace/real_rgbd_capture/segmentation
```

如果要推动的遮挡物就是该模具，使用输出的
`push_scene_rgb_01_mold_mask.png`。如果要推动的是其他物体，则必须使用
该物体自己的 Mask。Mask 中的白色区域表示送入 PyBullet 的物体表面。

#### 第三步：由真实 RGB-D 建模并运行 Push

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
conda activate smartgrasp

python scripts/demo_reveal_push.py \
  --rgb real_rgbd_capture/push_scene_rgb.png \
  --depth real_rgbd_capture/push_scene_depth.npy \
  --mask real_rgbd_capture/segmentation/push_scene_rgb_01_mold_mask.png \
  --intrinsics real_rgbd_capture/push_scene_intrinsics.json \
  --depth-scale 1000 \
  --mass 0.05 \
  --friction 0.7 \
  --distance 0.05 \
  --output results/reveal_push_rgbd.json
```

如果深度图单位是毫米，使用 `--depth-scale 1000`；如果已经是米，使用
`--depth-scale 1`。脚本会从真实深度拟合桌面平面，将目标可见点云构造成
单视角凸包碰撞代理，再在 PyBullet 中执行 push。

#### 第四步：在网页端查看真实 RGB-D 仿真回放

```bash
python gui/app.py \
  --host 0.0.0.0 \
  --port 8051 \
  --results results/reveal_push_rgbd.json \
  --viz-data results/reveal_push_rgbd_viz_data.pkl
```

浏览器打开 `http://127.0.0.1:8051`。当前 `8050` 已被其他程序占用时，
使用这里的 `8051`。

这种模式中 RGB、深度、Mask 和物体可见表面来自真实传感器；物体隐藏面、
质量、摩擦系数、夹爪和运动过程仍属于仿真近似。单张 RGB 图片或归一化
高度图不能替代真实深度图。

---

## 🔗 后续集成 (Integration Note)

本目录为**纯算法逻辑层**。在项目的第 3 个月（Month 3）进行全系统闭环测试时，本目录下的 API 将被主控流水线（如 `graspnet-workspace/scripts/demo_closed_loop.py`）动态导入。主程序将根据感知层传来的 `is_occluded` 状态，对 `execution_api.py` 和 `reveal_api.py` 进行交通路由，并在 PyBullet 仿真引擎中渲染最终的机械臂 3D 物理执行动画。

完整的真实 RGB-D 采集、Mask 生成、Push 仿真和网页回放命令见上面的
“使用真实 RGB-D 数据运行 Push 仿真”章节。
