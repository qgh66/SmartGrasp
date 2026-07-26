# SmartGrasp Grasp Execution Module

## 正式完整 Pipeline：拍照到真实抓取

正式入口是项目根目录的 `run_pipeline.py`。它按下面的顺序完整运行一次：

```text
尾号 72659 的 RealSense 拍摄 RGB-D
→ Perception（背景 Mask、SAM2、VLM、遮挡图）
→ Intent（从英文指令选择目标对象）
→ Reason（选择当前要抓的对象和 SAM2 part）
→ GraspNet（只使用 Reason 指定的二值 part mask）
→ JAKA + Robotiq 实际抓取、放置并返回拍摄位
```

### 运行前准备

1. 确认 JAKA 工作区安全、急停已经解除、控制器和夹爪可用。
2. 确认尾号 `72659` 的 RealSense 已连接；当前完整序列号为
   `243122072659`。
3. 确认根目录存在 `api_config.json`：

```json
{
  "base_url": "<OpenAI-compatible API URL>",
  "api_key": "<API Key>"
}
```

4. 确认以下文件存在：

```text
graspnet-workspace/checkpoints/checkpoint-rs.tar
graspnet-workspace/calibration/hand_eye_tcp_camera.json
```

### 正式运行命令

必须从项目根目录启动，并使用 `smartgrasp` 环境：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp

python -u run_pipeline.py \
  --instruction "grasp the screwdriver on the left"
```

这条命令会真实移动机械臂和夹爪，不是测试或 dry-run。默认使用：

```text
calibration-mode = hand_eye
top-k = 50
candidate-index = 0
camera serial suffix = 72659
```

每次运行创建一个不带 `scene_` 前缀的时间戳目录：

```text
data_realworld/<YYYYMMDD_HHMMSS>/
├── input/          # RGB、raw/npy 深度、相机参数、拍摄 TCP 位姿、GraspNet 结果
├── perception/     # SAM2 mask、背景 mask、遮挡图、summary.json
├── intent/         # id.txt
└── reason/         # results.csv、summary.json、reason.txt
```

完整主日志写入：

```text
logs/realworld_<YYYYMMDD_HHMMSS>.log
```

Intent 只执行一次并把 `intent/id.txt` 交给 Reason。Reason 输出的
`selected_object_graspability_part_id` 会映射到：

```text
perception/mask_sam2/part_NNN.png
```

正式 Grasp 只接受这张二值 mask；mask 缺失时会停止，不会退化成 RGB 裁剪图
或整幅点云。如果一次 GraspNet 推理的候选全部被安全过滤，pipeline 最多只
重试一次推理，不降低 TCP 最低高度阈值，也不会在没有有效候选时移动机械臂。

### 分阶段停止

下面的参数用于检查中间产物：

```bash
# 只拍照，拍摄阶段仍会把机械臂移动到拍摄位并打开夹爪
python -u run_pipeline.py \
  --instruction "grasp the screwdriver on the left" \
  --capture-only

# 拍照并运行 Perception 后停止
python -u run_pipeline.py \
  --instruction "grasp the screwdriver on the left" \
  --perception-only

# 运行到 Reason 后停止
python -u run_pipeline.py \
  --instruction "grasp the screwdriver on the left" \
  --reason-only

# 完成 Perception、Intent、Reason，但跳过 GraspNet/JAKA 抓取
python -u run_pipeline.py \
  --instruction "grasp the screwdriver on the left" \
  --no-grasp
```

复用已经拍摄的场景、重新执行后续阶段：

```bash
python -u run_pipeline.py \
  --scene-dir /home/admin128/qiuguanhe/SmartGrasp/data_realworld/20260726_225018 \
  --instruction "grasp the screwdriver on the left"
```

使用 `--scene-dir` 并执行真实抓取时，必须保证物体、相机和拍摄时的相对位置
没有变化，并且 `input/capture_tcp_pose.json` 与该 RGB-D 帧匹配。

---

这个分支还保存了基于 **GraspNet + PyBullet** 的单物体抓取仿真，以及基于
**JAKA Zu3 + Robotiq-85 + RealSense** 的独立真实抓取工具。仿真结果可通过
Dash GUI 查看，真实抓取结果可通过 PNG、HTML、PLY 和 JSON 工件检查。

下面介绍的 PyBullet 仿真采用 **JAKA Zu3 机械臂 + Robotiq-85 二指夹爪**，
用 PyBullet IK 驱动机械臂、Robotiq-85 欠驱动夹爪做**真实摩擦夹持**（不再
用固定约束“吸附”物体），整体流程对齐参考实现 `environment_sim.py` 的
`grasp()` 原语：张开 → 移到目标上方 → 直线下插 → 闭合 → 直线抬回 →
按夹爪关节角判定是否夹到实体。这个仿真小节只抓单个物体，不包含正式
pipeline 中的 VLM Intent/Reason。

正式完整 pipeline 的入口在根目录 `run_pipeline.py`；GraspNet、标定、仿真和
真实执行的底层代码主要位于 `graspnet-workspace/`，感知代码位于
`perception/`。

## 目录结构

```text
SmartGrasp/
├── perception/                 # 感知 pipeline（Molmo/SAM 等）
├── execution/                  # 执行层入口和请求处理
├── graspnet-workspace/         # GraspNet 仿真、真实抓取、标定和可视化
├── result/                     # 当前真实抓取输出（候选 JSON/PNG/HTML/PLY）
├── realworld_data/             # RealSense 多步骤采集数据
├── smartgrasp.full.yml         # conda 环境导出文件
└── smartgrasp.full.no_pip.yml
```

`graspnet-workspace/` 中最重要的文件是：

```text
graspnet-workspace/
├── scripts/
│   ├── demo_closed_loop.py     # 闭环仿真主入口
│   ├── realworld_grasp.py      # 真实 RGB-D -> GraspNet -> 可选 JAKA 执行
│   ├── demo_inference.py       # 单次 GraspNet 推理示例
│   ├── test_scripted_grasp.py  # PyBullet 脚本抓取测试
│   ├── collect_handeye_chessboard.py # 手眼标定数据采集
│   ├── solve_handeye_chessboard.py   # 手眼标定求解
│   ├── check_chessboard_height.py    # 检查棋盘高度
│   ├── print_jaka_tcp_pose.py        # 读取 JAKA TCP 位姿
│   └── jaka_motion_worker.py          # JAKA/夹爪动作 worker
├── capture_realsense.py         # 单帧 RealSense RGB-D 采集
├── capture_realsense_scenes.py  # 多步骤场景采集
├── visualize_ply.py             # Open3D 查看 grasp_candidates.ply
├── simulation/
│   ├── run_sim.py               # 直接运行 GraspNet + PyBullet 仿真
│   ├── scene.py                 # PyBullet 场景和物体 mesh
│   ├── camera.py                # 虚拟 RGB-D 相机和点云反投影
│   ├── robot_gripper.py         # JAKA Zu3 + Robotiq-85 适配器
│   ├── evaluator.py             # 抓取执行与物理评估
│   ├── candidate_visualizer.py  # PNG/HTML/PLY 候选可视化导出
│   └── planning/moveit_bridge.py # 可选 ROS2/MoveIt 规划桥接
├── config/
│   ├── realworld_config.yaml    # 真实抓取、相机和机器人默认配置
│   └── industrial_scene.json    # 工业场景配置
├── calibration/
│   └── hand_eye_tcp_camera.json # 当前手眼标定结果
├── assets/                      # 物体和机器人 mesh/URDF
├── checkpoints/                 # GraspNet checkpoint
├── log/                         # 真实抓取试验日志
├── results/                     # 仿真结果和 GUI 数据
├── models/, pointnet2/, knn/    # GraspNet 网络及 CUDA/PyTorch 扩展
├── graspnet_api/                # GraspGroup 等接口
├── utils/                       # 点云、标定、夹爪和运动工具
├── gui/app.py                   # Dash GUI
├── ros2_moveit/                 # JAKA Zu3 MoveIt 规划脚本
└── vendor/                      # Robotiq/串口等 vendored 依赖
```

## 环境准备

所有命令建议在 `smartgrasp` conda 环境中运行：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace
```

检查 GUI 和仿真依赖是否可导入：

```bash
python -c "import pybullet, dash, plotly, trimesh, scipy; print('deps OK')"
```

如果缺少 `dash`、`plotly`、`trimesh`、`pybullet` 等包，需要先安装到 `smartgrasp` 环境中。

## 需要准备的外部文件

大文件不建议直接提交到 Git。运行 milestone demo 时至少需要：

1. GraspNet checkpoint，例如：


```text
/home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/checkpoints/checkpoint-rs.tar
```

2. 一个待抓取物体 mesh，例如：

```text
graspnet-workspace/assets/objects/industrial_tools/two_color_hammer/textured.obj
```

运行时也可以通过 `--obj` 指定其他 `.obj` 文件。

3. JAKA Zu3 与 Robotiq-85 的 URDF（已随仓库 `graspnet-workspace/assets/robots/` 提供，无需另外准备）：

```text
graspnet-workspace/assets/robots/jaka_zu3/gazebo_jaka_zu3_robotiq.urdf
```

> 注意 mesh 的单位：仿真按米制处理。图形学单位的 mesh（如 `duck.obj` 等）需要用
> `--scale` 缩到约 5~8 cm 的桌面小物体尺寸，否则会因为太大/太小而抓不到。`banana.obj`
> 本身就是米制（约 22 cm），用默认 `--scale 1.0` 即可。

## 运行闭环仿真

当前直接从 `graspnet-workspace` 运行闭环仿真入口，输出写到 `results/`：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace

python scripts/demo_closed_loop.py \
  --obj assets/objects/industrial_tools/two_color_hammer/textured.obj \
  --ckpt checkpoints/checkpoint-rs.tar \
  --top_k 5 \
  --device cuda:0 \
  --output results/grasp_simulation.json
```

如果需要调整物体 mesh 尺寸，可用 `--scale` 透传给主入口：

```bash
python scripts/demo_closed_loop.py \
  --obj assets/objects/industrial_tools/two_color_hammer/textured.obj \
  --ckpt checkpoints/checkpoint-rs.tar \
  --top_k 5 \
  --scale 1.0
```

如果只是调试流程、没有可用 GPU，可以传入 `--device cpu` 跑小规模测试（较慢）。除非明确需要调度提交，否则不默认使用 SLURM。

主要命令行参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--obj` | — | 物体 `.obj` 路径 |
| `--ckpt` | 自动查找 | GraspNet checkpoint，找不到时报错 |
| `--top_k` | 10 | 评估打分最高的前 K 个抓取 |
| `--scale` | 1.0 | 物体缩放因子（图形学单位 mesh 需缩小） |
| `--device` | cuda:0 | 推理设备 `cuda:0` / `cpu` |
| `--gui` | 关 | 打开 PyBullet 图形窗口 |
| `--output` | results/grasp_simulation.json | 结果 JSON 路径（相对 `graspnet-workspace/`） |

这个脚本会依次完成：

1. 在 PyBullet 中搭建桌面，并按配置加载单物体 mesh（默认固定朝向；设置 `GRASP_RANDOM_ORIENTATION=1` 或 `--random-orientation` 后随机朝向），等其稳定后记录位姿。
2. 用虚拟相机（1280×720）拍 RGB-D，将 depth 反投影为世界坐标系点云。
3. **按物体点云的 xy 包围盒裁剪点云**（裁掉远处大片桌面，只留物体及周围一圈支撑面，对应参考流程的 crop_pointcloud 简化版），再送入 GraspNet 生成候选抓取。
4. 加载 **JAKA Zu3 + Robotiq-85**，对 top-k 抓取逐个执行物理仿真（见下一节），逐帧记录夹爪和物体位姿。
5. 保存 JSON 结果和 GUI 可视化数据。

默认输出文件统一为：

```text
graspnet-workspace/results/grasp_simulation.json
graspnet-workspace/results/grasp_simulation_viz_data.pkl
graspnet-workspace/results/grasp_simulation_candidates.png
graspnet-workspace/results/grasp_simulation_candidates.html
```

其中：

- `grasp_simulation.json`：每个抓取的分数、位姿、宽度、深度、是否成功、判定相关诊断字段、逐帧动画轨迹日志，以及执行用的夹爪 metadata。
- `grasp_simulation_viz_data.pkl`：RGB、depth、点云、物体路径、物体姿态等 GUI 需要的数据。
- `grasp_simulation_candidates.png/html`：候选抓取和实际执行姿态的静态/交互诊断图。

如果显式设置 `GRASP_RECORD_VIDEO=1`，还会生成：

```text
graspnet-workspace/results/grasp_simulation_pybullet.mp4
```

## 抓取执行流程（JAKA Zu3 + Robotiq-85）

抓取执行在 `simulation/evaluator.py` 的 `GraspEvaluator` 中，对 GraspNet 输出的每个候选执行一次完整的物理抓取，流程对齐参考实现 `environment_sim.py` 的 `grasp()`：

1. **抓取点修正**：抓取中心沿 z 下压 2 cm 并用桌面高度兜底（`center.z = max(center.z - 0.02, TABLE_Z)`），与参考一致。
2. **几何护栏**：抓取中心到物体点云的最近距离若超过 `MAX_GRASP_CENTER_DIST` 则判失败（`grasp_center_not_on_object`），避免隔空抓取。（抓取中心的桌面高度限制已按需求取消。）
3. **接近**：张开夹爪 → 移到目标正上方 `over` 点 → 直线下插到预抓取点 → 沿 approach 方向推进到抓取中心。约定 GraspNet 的 local X 轴为 approach/depth 方向、local Y 轴为夹爪开合方向。
4. **闭合**：Robotiq-85 欠驱动夹爪慢速闭合，靠 `JOINT_GEAR` mimic 约束 + 高摩擦指垫做**真实摩擦夹持**（不创建固定约束“吸附”物体）。
5. **抬升**：夹爪沿 z 直线抬回。
6. **判定**：`success = gripper.is_gripper_closed()` —— 夹爪主关节角未完全合死即视为夹到实体（完全照参考）。物体能否被带起是物理自然结果，仅作诊断字段记录，不作为额外判据。

机械臂运动用 PyBullet IK（`JakaZu3Robotiq85Gripper`，`planner=None`），默认**不启用** MoveIt。

相关常量（`simulation/evaluator.py`）：

```text
TABLE_Z = 0.0                 # 桌面高度
TABLE_CLEARANCE = 0.005       # 桌面余量
MAX_GRASP_CENTER_DIST = 0.04  # 抓取中心到物体点云的最大允许距离（米）
```

夹爪闭合力等参数在 `simulation/robot_gripper.py`（如 `GRIPPER_MOTOR_FORCE = 8.0`，与参考一致）。

## 如何评估 / 查看评估结果

`results.json` 顶层给出整体评估：

- `total`：实际评估的抓取数（≤ `--top_k`）。
- `success`：成功抓取数（即 `is_gripper_closed()` 判定为 True 的数量）。
- `gripper`：执行用的夹爪 metadata（`model: jaka_zu3_robotiq85`、是否启用 MoveIt 等）。

每个 `grasps[i]` 的关键字段：

| 字段 | 含义 |
|------|------|
| `success` | 该抓取是否成功（= `grasped_by_gripper`） |
| `grasped_by_gripper` | 夹爪关节角判定是否夹到实体（判定依据） |
| `score` | GraspNet 打分 |
| `translation` / `rotation` | 抓取中心位姿（已含下压修正） |
| `width` / `depth` | 抓取宽度 / 深度 |
| `obj_z_before` / `obj_z_after` / `obj_lift_delta` | 抓取前后物体高度及位移（**诊断用**：判断物体是否真被带起） |
| `failure_reason` | 被护栏拦下时的原因（如 `grasp_center_not_on_object`），正常执行的抓取无此字段 |
| `frame_log` | 逐帧（approach/close/lift/done）的夹爪与物体位姿，供 GUI 回放 |

命令行快速查看一次评估结果：

```bash
python -c "
import json,numpy as np
d=json.load(open('graspnet-workspace/results/grasp_simulation.json'))
print('gripper:', d['gripper']['model'], '| success', d['success'], '/', d['total'])
for g in d['grasps']:
    s='OK' if g['success'] else 'x'
    print(f\"  {s} g{g['grasp_index']} grasped={g.get('grasped_by_gripper')} \"
          f\"lift_delta={round(g.get('obj_lift_delta',0),4)} reason={g.get('failure_reason','-')}\")
"
```

要按真实“物体被提起”更严格地评估，可在上面用 `obj_lift_delta`（lift 后物体相对上升量）自行加判据；当前默认判定与参考保持一致，只看夹爪关节角。

## 启动 Dash GUI

GUI 不会重新运行 GraspNet，也不会重新跑 PyBullet。它只读取已经保存好的：

```text
results.json
results_viz_data.pkl
```

启动命令：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace

python gui/app.py \
  --host 0.0.0.0 \
  --port 8050 \
  --results results/grasp_simulation.json \
  --viz-data results/grasp_simulation_viz_data.pkl
```

如果是在服务器本机浏览器打开：

```text
http://127.0.0.1:8050
```

如果是在自己的电脑访问远程服务器，需要先做端口转发。VS Code / Cursor 一般会自动提示转发 `8050`；也可以在本地终端手动执行：

```bash
ssh -L 8050:127.0.0.1:8050 <user>@<server>
```

然后在本地浏览器打开：

```text
http://127.0.0.1:8050
```

## GUI 页面内容

GUI 主要包含四块：

- `Point Cloud and Grasp Poses`：显示桌面、物体点云、候选抓取中心和方向。这里故意不渲染完整夹爪，避免第一张图过乱。
- `Constrained GraspNet Animation`：播放一个符合当前约束的抓取演示动画。
- `RGB`：虚拟相机拍到的彩色图。
- `Depth`：虚拟相机深度图。

动画中的物体模型优先使用完整 mesh：

- 如果结果中保存了 `object_orientation`，GUI 会用 PyBullet 中真实的物体姿态渲染 mesh。
- 如果旧结果没有 `object_orientation`，GUI 会从物体点云估计姿态，尽量保持 mesh 与场景一致。
- lift 阶段物体和夹爪同步上升。

## GUI 输入格式

`results.json` 的核心字段示例：

```json
{
  "total": 3,
  "success": 3,
  "obj_path": "/path/to/duck.obj",
  "object_position": [0.31, 0.008, -0.003],
  "object_orientation": [0.66, -0.24, -0.24, 0.66],
  "gripper": {
    "model": "jaka_zu3_robotiq85",
    "execution": "jaka_ik_attached_robotiq",
    "moveit_enabled": false
  },
  "grasps": [
    {
      "grasp_index": 0,
      "success": true,
      "grasped_by_gripper": true,
      "score": 0.127,
      "translation": [0.28, -0.05, 0.045],
      "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "width": 0.066,
      "depth": 0.02,
      "obj_z_before": -0.003,
      "obj_z_after": -0.003,
      "obj_lift_delta": 0.0,
      "frame_log": []
    }
  ]
}
```

`results_viz_data.pkl` 通常包含：

```python
{
    "rgb": np.ndarray,
    "depth": np.ndarray,
    "point_cloud": np.ndarray,
    "obj_path": str,
    "object_orientation": list,
}
```

## 常见问题

### 浏览器显示 connection refused

先确认 Dash 进程是否还在跑：

```bash
ps -ef | grep 'gui/app.py'
```

再确认启动时用了：

```text
--host 0.0.0.0 --port 8050
```

如果是在远程服务器上跑，还需要确认 `8050` 端口已经转发到本地。

### `ModuleNotFoundError: No module named 'dash'`

通常是没有进入 `smartgrasp` 环境：

```bash
conda activate smartgrasp
python -c "import dash"
```

### GUI 看到的不是最新结果

GUI 读的是磁盘上的结果文件。重新生成结果后，需要启动 GUI 时指定新的路径：

```bash
python gui/app.py \
  --results results/grasp_simulation.json \
  --viz-data results/grasp_simulation_viz_data.pkl
```

浏览器中可以用 `Ctrl + Shift + R` 强制刷新。

### 动画里物体不是完整模型

优先使用 `scripts/demo_closed_loop.py` 重新生成结果。新的结果会保存 `object_orientation`，GUI 才能更稳定地按真实姿态渲染完整 mesh。

### 抓取成功了，但物体没真正被提起来

当前判定与参考实现一致，只看夹爪闭合后主关节角（`is_gripper_closed()`）——指间有阻挡即判成功。真实物理夹取下，物体可能被夹到但在抬升时从指间滑脱，此时 `obj_lift_delta` 接近 0。需要“物体确实被提起”才算成功时，用结果里的 `obj_lift_delta` 字段自行加判据即可。

### 抓不到 / 候选都被护栏拦下

常见原因是物体尺度不对（图形学单位 mesh 没缩放，用 `--scale` 调到约 5~8 cm），或物体随机朝向下平躺导致 GraspNet 抓取质量差。可多跑几次，或换更立体、规则的物体。

## RealSense 多步骤场景采集

使用独立脚本连续采集同一个真实场景中的多个步骤：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace
python capture_realsense_scenes.py --camera-serial 72659
```

脚本会在 `realworld_data/` 下创建下一个未使用的场景目录，编号从 `scene_1` 开始。预览窗口左侧为 RGB，右侧为对齐到彩色图的伪彩深度：

- 按 `c`：把当前原始 RGB-D 帧保存到当前场景的 `step_N/`，步骤编号从 `step_0` 递增。
- 按 `q`：停止采集、关闭相机并退出。

输出结构如下：

```text
realworld_data/
└── scene_1/
    ├── step_0/
    │   ├── rgb.png
    │   ├── depth.png
    │   └── camera_meta.json
    └── step_1/
        ├── rgb.png
        ├── depth.png
        └── camera_meta.json
```

`depth.png` 是无损 `uint16` 原始深度，不是窗口中显示的伪彩图；真实米制深度等于像素值乘以 `camera_meta.json` 中的 `depth_scale_m`。如需修改数据根目录，可传入 `--output-root /path/to/realworld_data`。

## Eye-in-hand 棋盘格手眼标定

当前真实机械臂场景中，RealSense 固定在夹爪/末端上。此时需要标定相机坐标系和 JAKA TCP/夹爪坐标系之间的固定变换 `T_tcp_camera`。根据当前实测候选坐标，真实抓取运行时使用该矩阵的逆矩阵：

```text
T_base_grasp = T_base_tcp_capture @ inv(T_tcp_camera) @ T_camera_grasp
```

你当前棋盘参数按“12 x 9 个方格、单格 1 cm”处理，因此 OpenCV 检测参数是 **11 x 8 个内角点**，方格边长 `10.0 mm`。如果 12 x 9 实际指内角点数量，求解命令中把 `--pattern-cols 11 --pattern-rows 8` 改成 `--pattern-cols 12 --pattern-rows 9`。

物理采集要求：

1. 棋盘格固定在桌面或硬质板上，采集过程中不能移动。
2. RealSense 固定在夹爪/末端上，采集过程中不能松动。
3. 手动移动机械臂，让相机从不同方向看到完整棋盘；推荐采集 15-25 组。
4. 姿态要同时包含平移和明显的 roll/pitch/yaw 旋转变化，不能只平移。

采集前可以用手操脚本调整机械臂姿态：

```bash
/home/admin128/anaconda3/envs/smartgrasp310/bin/python plush5.py
```

如果只想在终端打印当前夹爪/TCP 在 JAKA base 坐标系下的位姿，可以运行：

```bash
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace

python scripts/print_jaka_tcp_pose.py \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc
```

输出中的 `x/y/z` 单位是 mm，`rx/ry/rz` 单位是 rad，坐标系是 `jaka_base`。

采集脚本会打开 RealSense 预览，并实时检测棋盘角点：检测成功时会在画面上画出角点，状态显示 `chessboard=FOUND`；检测失败时显示 `NOT FOUND`。每移动到一个合适姿态后按 `c` 保存一组 `RGB + depth + 当前 TCP pose + 相机内参 + 棋盘角点`；按 `q` 退出。若当前帧没有识别到棋盘角点，按 `c` 不会保存样本：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace

python scripts/collect_handeye_chessboard.py \
  --output-dir calibration/handeye_chessboard_raw \
  --camera-serial 72659 \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc
```

采集结果写入：

```text
graspnet-workspace/calibration/handeye_chessboard_raw/samples.jsonl
graspnet-workspace/calibration/handeye_chessboard_raw/sample_0/rgb.png
graspnet-workspace/calibration/handeye_chessboard_raw/sample_0/depth.png
graspnet-workspace/calibration/handeye_chessboard_raw/sample_0/corners.json
graspnet-workspace/calibration/handeye_chessboard_raw/sample_0/corners_visualization.png
graspnet-workspace/calibration/handeye_chessboard_raw/sample_0/metadata.json
...
```

其中 `corners_visualization.png` 是带角点标注的可视化结果，`corners.json` 保存该帧识别到的 11 x 8 个棋盘角点像素坐标。`samples.jsonl` 会记录每组 `sample_x` 的路径、TCP 位姿、相机内参和角点数量。

求解手眼标定：

```bash
python scripts/solve_handeye_chessboard.py \
  --input-dir calibration/handeye_chessboard_raw \
  --pattern-cols 11 \
  --pattern-rows 8 \
  --square-mm 10.0 \
  --output calibration/hand_eye_tcp_camera.json
```

输出文件中最重要的是：

```text
T_tcp_camera
```

同时脚本会输出棋盘在机器人 base 下的一致性验证指标。`base_board_translation_error_mean_mm` 建议尽量小于 5-10 mm；如果超过 20 mm，优先检查棋盘内角点参数、方格边长、TCP 定义、采集时相机是否松动，以及是否采集了足够多旋转姿态。

## 实机运行

真实抓取入口是 `graspnet-workspace/scripts/realworld_grasp.py`。运行前确认机器人工作区无人、急停可用、JAKA 已上电解锁、Robotiq 串口可访问，并检查代理状态。所有 Python 命令都在 `smartgrasp` 环境中运行；JAKA 控制默认通过独立的 `smartgrasp310` Python 子进程完成。

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace
proxy_status
```

三个常用模式已有独立脚本，脚本内部会调用仓库根目录的 `run_realworld_grasp.sh`，自动处理 conda、代理和 Python 路径：

| 模式 | 脚本 | 默认行为 |
|------|------|----------|
| 拍照模式 | `scripts/run_realworld_capture.sh` | 回拍照位、打开夹爪、采集并生成候选，不抓取 |
| 单次抓取 | `scripts/run_realworld_single_grasp.sh` | 新采集并完成一次抓取、放置 |
| 连续抓取 | `scripts/run_realworld_continuous_grasp.sh` | 持续执行拍照、抓取、放置，直到中断 |

直接运行：

```bash
bash scripts/run_realworld_capture.sh
bash scripts/run_realworld_single_grasp.sh
bash scripts/run_realworld_continuous_grasp.sh
```

可通过环境变量覆盖常用参数，例如切换严格 mask 模式、候选编号和运动速度：

```bash
GRASP_INPUT_MODE=mask bash scripts/run_realworld_capture.sh
CANDIDATE_INDEX=1 VELOCITY=10 ACCELERATION=10 bash scripts/run_realworld_single_grasp.sh
GRASP_INPUT_MODE=mask VELOCITY=30 ACCELERATION=15 bash scripts/run_realworld_continuous_grasp.sh
```

三个脚本还接受额外命令行参数，并将其追加传给 `realworld_grasp.py`。连续脚本默认 `NUM_CYCLES=0`，即持续运行；例如只连续运行 3 次：

```bash
NUM_CYCLES=3 bash scripts/run_realworld_continuous_grasp.sh
```

### 1. 新采集并生成候选，不执行抓取

这是第一次试抓的推荐入口。该命令会先将机械臂移动到拍照关节位、打开夹爪，再采集 RGB-D、交互选择 SAM 目标并生成候选，但不会执行候选抓取。

```bash
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --top-k 100 \
  --grasp-input-mode bbox \
  --trial-log-subdir single_object \
  --trial-name capture_only \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 10 \
  --acceleration 10
```

若只需要严格 SAM 掩码内的目标点云，将 `--grasp-input-mode bbox` 改成：

```bash
--grasp-input-mode mask
```

`bbox` 会使用 SAM 掩码外接矩形内的有效深度点，`mask` 只使用掩码内部的点。`--no-use-sam-mask` 会跳过 SAM，并让 GraspNet 使用完整有效深度点云。

### 2. 复用现有 RGB-D，不移动机器人

`--reuse-capture` 读取 `result/rgb.png`、`depth.raw`、`camera_meta.json` 和 `capture_tcp_pose.json`，不重新采集，也不会在未指定 `--execute` 时连接或移动 JAKA：

```bash
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --reuse-capture \
  --grasp-input-mode bbox \
  --top-k 100 \
  --trial-log-subdir single_object \
  --trial-name reuse_capture
```

`capture_tcp_pose.json` 必须与复用的 RGB-D 来自同一拍照时刻。也可以用 `--capture-tcp-pose X Y Z RX RY RZ` 显式提供拍照时 TCP 位姿，其中位置单位为 mm、姿态单位为 rad。

### 3. 低速执行一次抓取

确认候选、坐标变换和工作空间安全后，使用单次抓取脚本。它会重新回到拍照位、采集当前 RGB-D、运行推理和过滤，然后执行本轮结果中的第 0 个候选：

```bash
bash scripts/run_realworld_single_grasp.sh
```

对应的完整命令为：

```bash
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --grasp-input-mode bbox \
  --top-k 100 \
  --trial-log-subdir single_object \
  --trial-name grasp_execute_once \
  --execute \
  --candidate-index 0 \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 10 \
  --acceleration 10 \
  --approach-offset-mm 100 \
  --lift-mm 80
```

`--candidate-index` 指向当前这次运行过滤、重排后的候选。若通过第 2 节的 `--reuse-capture` 调试后直接执行，机械臂必须仍处于该帧对应的拍照位姿；复用输入仍会重新采样点云和运行 GraspNet，不保证同编号候选与上一次完全相同。

实际执行顺序为：预抓取位姿 -> 抓取位姿 -> 闭合夹爪 -> 回拍照关节位 -> 到放置关节位 -> 沿机器人 base Z 向下 100 mm -> 打开夹爪 -> 回拍照关节位。

### 4. 连续拍照、抓取和放置

`--loop` 等价于 `--num-cycles 0`，会一直循环直到 `Ctrl+C` 或急停。先完成单次低速验证后再使用。

使用 bbox 点云：

```bash
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --top-k 100 \
  --trial-log-subdir single_object \
  --trial-name grasp_execute_bbox \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 40 \
  --acceleration 20 \
  --loop \
  --execute \
  --grasp-input-mode bbox \
  --grasp-crop-margin-px 50 \
  --grasp-crop-margin-ratio 0 \
  --target-mask-center-tolerance-px 0
```

使用严格 mask 点云时，只需将末尾的输入参数替换为：

```bash
--grasp-input-mode mask
```

需要固定循环次数时，用 `--num-cycles N` 代替 `--loop`。

### 5. 常用参数

以下默认值来自 `config/realworld_config.yaml` 和 `scripts/realworld_grasp.py`；命令行参数会覆盖配置值。

| 参数 | 当前默认值 | 作用 |
|------|------------|------|
| `--output-dir` | `../result` | 当前一轮的 RGB-D、点云和候选输出目录 |
| `--camera-serial` | 后缀 `72659` | RealSense 序列号或唯一后缀 |
| `--reuse-capture` | 关闭 | 复用输出目录中的现有 RGB-D |
| `--warmup-frames` | `30` | 新采集前的相机预热帧数 |
| `--device` | `cuda:0` | GraspNet 推理设备 |
| `--num-points` | `20000` | 输入 GraspNet 的采样点数 |
| `--top-k` | `50` | 保存和过滤的高分候选数量 |
| `--if-pca` | 关闭 | GraspNet 原始候选为 0 时，用目标点云生成 PCA fallback |
| `--grasp-input-mode` | `bbox` | SAM 启用时选择 `bbox` 或 `mask` 点云 |
| `--grasp-crop-margin-px` | `50` | bbox 每侧固定扩张像素数 |
| `--grasp-crop-margin-ratio` | `0.2` | bbox 按目标尺寸扩张的比例 |
| `--target-mask-center-tolerance-px` | `25` | 候选中心允许落在 mask 外的像素距离 |
| `--min-target-tcp-z-mm` | `165` | 最终物理 TCP 的最低允许 base Z |
| `--filter-grasp-collisions` | 开启 | 启用 model-free 碰撞过滤 |
| `--prefer-topdown-candidate` | 开启 | 在前 10 个候选中优先选择接近俯抓的姿态 |

执行和运动参数：

| 参数 | 当前默认值 | 作用 |
|------|------------|------|
| `--execute` | 关闭 | 执行当前轮选定候选；不加时只生成结果 |
| `--candidate-index` | `0` | 执行过滤和重排后的候选编号 |
| `--velocity` | `60` | JAKA 笛卡尔直线运动速度 |
| `--acceleration` | `60` | JAKA 笛卡尔直线运动加速度 |
| `--joint-velocity-rad-s` | `0.5` | 关节运动速度，单位 rad/s |
| `--approach-offset-mm` | `80` | 抓取前沿 TCP 局部 Z 的退让距离 |
| `--lift-mm` | `170` | 闭合后沿机器人 base Z 的抬升距离 |
| `--capture-joint-pose-deg` | `[0, 90, 45, 135, 270, 72]` | 每轮拍照前的 JAKA 关节角 |
| `--place-target-joint-pose-deg` | `[-75, 90, 45, 135, 270, 72]` | 抓取后的放置关节角 |
| `--place-release-lower-mm` | `100` | 到达放置关节位后沿 base Z 向下距离 |
| `--gripper-open-force` | `30` | Robotiq 打开力参数 |
| `--gripper-close-force` | `200` | Robotiq 闭合力参数 |
| `--num-cycles` | `1` | 完整流程循环次数，`0` 表示无限循环 |
| `--loop` | 关闭 | 无限循环，等价于 `--num-cycles 0` |

JAKA 和标定参数：

| 参数 | 当前默认值 | 作用 |
|------|------------|------|
| `--calibration-mode` | `legacy_plate` | 坐标变换模式；当前实机命令显式使用 `hand_eye` |
| `--hand-eye-calibration` | `calibration/hand_eye_tcp_camera.json` | 包含 `T_tcp_camera` 的标定文件 |
| `--jaka-ip` | `192.168.1.199` | JAKA 控制器地址 |
| `--robotiq-port` | `/dev/ttyUSB0` | Robotiq 串口 |
| `--jaka-executor` | `subprocess` | 在独立 Python 进程中执行 JAKA 动作 |
| `--persistent-jaka-worker` | 开启 | 循环时复用同一个 JAKA 子进程连接 |
| `--jaka-python` | `smartgrasp310/bin/python` | 兼容 `jkrc` 的 Python 解释器 |
| `--jkrc-dir` | `graspnet-workspace/jkrc` | `jkrc.so` 和 `libjakaAPI.so` 所在目录 |

### 6. 输出、日志与候选检查

当前输出默认写到 `/home/admin128/qiuguanhe/SmartGrasp/result/`。重点检查：

```text
rgb.png
depth.raw
camera_meta.json
capture_tcp_pose.json
mask.png
grasp_crop_overlay.png
point_cloud_object_camera.npy
point_cloud_grasp_input_camera.npy
grasp_candidates.json
grasp_candidates.png
grasp_candidates_3d.html
grasp_candidates.ply
```

`grasp_candidates.json` 同时保存相机坐标系候选和转换后的 `target_jaka_tcp_pose`。手眼模式的坐标链为：

```text
T_base_grasp = T_base_tcp_capture @ inv(T_tcp_camera) @ T_camera_grasp
```

查看 PLY 点云和夹爪网格：

```bash
python visualize_ply.py ../result/grasp_candidates.ply
```

成功生成 SAM 掩码后，每轮试验默认记录到：

```text
graspnet-workspace/log/single_object_grasp/YYYYMMDD_HHMMSS[_trial_name]/
```

使用 `--trial-log-subdir single_object` 时写入 `log/single_object/`。`run_info.json` 记录命令、输出和执行状态，`manual_result.json` 用于人工标记成功或失败；`--no-trial-log` 可关闭试验日志。

### 7. 实机辅助指令

手动控制机械臂：

```bash
/home/admin128/anaconda3/envs/smartgrasp310/bin/python plush5.py
```

读取当前 JAKA TCP 位姿：

```bash
python scripts/print_jaka_tcp_pose.py \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc
```
