# SmartGrasp Grasp Execution Module

这个分支保存了当前的 milestone 版本：基于 **GraspNet + PyBullet** 的单物体抓取仿真，以及用于查看结果和播放抓取动画的 **Dash GUI**。

当前抓取执行采用 **JAKA Zu3 机械臂 + Robotiq-85 二指夹爪**，用 PyBullet IK 驱动机械臂、Robotiq-85 欠驱动夹爪做**真实摩擦夹持**（不再用固定约束“吸附”物体），整体流程对齐参考实现 `environment_sim.py` 的 `grasp()` 原语：张开 → 移到目标上方 → 直线下插 → 闭合 → 直线抬回 → 按夹爪关节角判定是否夹到实体。当前只抓**单个物体**，不含 VLM / LangSAM。

当前重点代码在 `graspnet-workspace/` 下面。`perception/` 是 SmartGrasp 原有感知模块，本 README 主要说明 grasp execution 这部分如何运行。

## 目录结构

```text
SG_graspmodule/
├── perception/                 # SmartGrasp 原有感知 pipeline
├── graspnet-workspace/         # GraspNet + PyBullet 仿真和 GUI
├── smartgrasp.full.yml         # conda 环境导出文件
└── smartgrasp.full.no_pip.yml
```

`graspnet-workspace/` 中最重要的文件是：

```text
graspnet-workspace/
├── scripts/demo_closed_loop.py # 主入口：建场景 -> 拍 RGB-D -> 裁剪点云 -> GraspNet -> JAKA 仿真评估
├── simulation/
│   ├── scene.py                # PyBullet 场景：桌面、加载 .obj（支持缩放）
│   ├── camera.py               # 虚拟 RGB-D 相机（1280x720）+ 点云反投影
│   ├── robot_gripper.py        # JAKA Zu3 + Robotiq-85 适配器（IK、欠驱动夹持、is_gripper_closed）
│   ├── evaluator.py            # 抓取执行与物理评估（approach/close/lift，逐帧轨迹）
│   └── planning/moveit_bridge.py # 可选的 ROS2/MoveIt 规划桥接（默认不启用）
├── gui/app.py                  # Dash GUI，读取结果文件并展示动画
├── gui/README.md               # GUI 快速启动说明
├── models/                     # GraspNet 网络
├── utils/                      # 点云、碰撞检测、数据处理工具
├── pointnet2/, knn/            # CUDA extension 源码
└── graspnet_api/               # GraspGroup 等接口
```

## 环境准备

所有命令建议在 `smartgrasp` conda 环境中运行：

```bash
conda activate smartgrasp
cd /home/qiuguanhe/SmartGrasp/graspnet-workspace
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
/home/qiuguanhe/SmartGrasp/graspnet-workspace/checkpoints/checkpoint-rs.tar
```

2. 一个待抓取物体 mesh，例如：

```text
/home/qiuguanhe/SmartGrasp/assert/workspace/data/banana.obj
```

运行时也可以通过 `--obj` 指定其他 `.obj` 文件。

3. JAKA Zu3 与 Robotiq-85 的 URDF（已随仓库 `assert/` 提供，无需另外准备）：

```text
assert/jaka_zu3/jaka_zu3_pybullet.urdf
assert/ur5e/gripper/robotiq_2f_85.urdf
```

> 注意 mesh 的单位：仿真按米制处理。图形学单位的 mesh（如 `duck.obj` 等）需要用
> `--scale` 缩到约 5~8 cm 的桌面小物体尺寸，否则会因为太大/太小而抓不到。`banana.obj`
> 本身就是米制（约 22 cm），用默认 `--scale 1.0` 即可。

## 运行闭环仿真

推荐从仓库根目录通过 SLURM 脚本运行。脚本会进入 `graspnet-workspace`、激活环境、设置依赖路径，并把输出统一写到 `graspnet-workspace/results/`：

```bash
conda activate smartgrasp
cd /home/qiuguanhe/SmartGrasp

GRASP_OBJ_PATH=/home/qiuguanhe/SmartGrasp/assert/unseen_objects/gelatin_box/textured.obj \
GRASP_TOP_K=5 \
sbatch run_grasp_simulation.sh
```

如果要抓一个图形学单位的立体物体（例：duck 缩到约 6 cm），用 `--scale` 透传给主入口：

```bash
sbatch run_grasp_simulation.sh \
  --obj /home/qiuguanhe/SmartGrasp/assert/workspace/data/duck.obj \
  --top_k 5 \
  --scale 0.04
```

如果只是调试流程、没有可用 GPU，可以通过 `GRASP_DEVICE=cpu` 跑小规模测试（较慢）。默认不录制 MP4；需要视频时显式加 `GRASP_RECORD_VIDEO=1`。

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
cd /home/qiuguanhe/SmartGrasp/graspnet-workspace

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
  --camera-serial 72508 \
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

### 启动真实抓取

第一次试抓不要直接执行机械臂，先只拍照、生成候选抓取，并确认候选落在目标物体上：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace

MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --top-k 20 \
  --grasp-input-mode bbox \
  --trial-name ring_horizontal_01 \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 10 \
  --acceleration 10
```

`--grasp-input-mode` 控制送入 GraspNet 的点云区域：

- `bbox`（默认）：使用 SAM 掩码外接矩形内的全部有效深度点。`--grasp-crop-margin-px` 和 `--grasp-crop-margin-ratio` 只影响该模式。
- `mask`：仅使用 SAM 掩码内部的有效深度点，适合外接矩形中混入大量桌面或相邻物体的场景。

严格按 SAM 掩码生成 GraspNet 输入时，使用以下完整命令：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace

MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --top-k 20 \
  --grasp-input-mode mask \
  --trial-name ring_horizontal_mask_01 \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 10 \
  --acceleration 10
```

如果指定 `--no-use-sam-mask`，程序会输入完整深度点云，忽略 `--grasp-input-mode`。

生成后先检查：

```text
/home/admin128/qiuguanhe/SmartGrasp/result/grasp_candidates.png
/home/admin128/qiuguanhe/SmartGrasp/result/grasp_candidates_3d.html
/home/admin128/qiuguanhe/SmartGrasp/result/grasp_candidates.json
```

点云调试文件的含义如下：

- `point_cloud_object_camera.npy`：始终为严格 SAM 掩码内的目标点云。
- `point_cloud_grasp_input_camera.npy`：本轮实际送入 GraspNet 的点云；内容由 `--grasp-input-mode` 决定。
- `grasp_crop_overlay.png`：同时显示 SAM 掩码和矩形裁剪范围，两种输入模式都会生成。

`grasp_candidates.json` 会记录 `grasp_input_mode`、`num_object_points`、`num_grasp_crop_points` 和 `num_grasp_input_points`，用于确认本轮实际输入模式及各类点数。

拍照并成功生成目标掩码后，脚本会默认保存一份轻量试验日志到：

```text
graspnet-workspace/log/single_object_grasp/YYYYMMDD_HHMMSS[_trial_name]/
```

其中包含 `rgb.png`、`depth.raw`、`camera_meta.json`、`grasp_candidates.json`、`grasp_candidates.png`、`scene_grasps.ply`、`mask_overlay.png`、`run_info.json` 和可手动填写成功/失败的 `manual_result.json`。

如果上一把急停后机器人未解锁，导致本次在回默认拍照位姿或打开夹爪阶段失败，本次不会创建试验日志；只有完成拍照并成功生成掩码后，才视为一次需要记录的试验。

所有试验日志都固定放在 `graspnet-workspace/log/` 下。如果希望当前单物体试验日志放到 `graspnet-workspace/log/single_object/` 下，运行时指定：

```bash
--trial-log-subdir single_object
```

例如：

```bash
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --top-k 20 \
  --trial-log-subdir single_object \
  --trial-name ring_horizontal_01 \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 10 \
  --acceleration 10
```

如果本次不想保存试验日志，添加：

```bash
--no-trial-log
```

确认 `grasp_candidates.json` 中：

```text
camera_to_robot_chain = T_base_tcp_capture @ inv(T_tcp_camera) @ T_camera_grasp
```

如果候选姿态合理，再复用同一帧 RGB-D 和同一份 `capture_tcp_pose.json` 低速执行：

```bash
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --reuse-capture \
  --camera-serial 243122072659 \
  --trial-log-subdir single_object \
  --trial-name ring_horizontal_01_exec \
  --execute \
  --candidate-index 0 \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 10 \
  --acceleration 10 \
  --approach-offset-mm 100 \
  --lift-mm 80
```

如果第 0 个候选不合理，先在 `grasp_candidates.json` 里选更合适的编号，再修改：

```text
  --candidate-index 0
```

hand-eye 模式会在拍照后记录当前 TCP 到 `result/capture_tcp_pose.json`，并用 `T_base_tcp_capture @ inv(T_tcp_camera) @ T_camera_grasp` 生成每个候选的 `target_jaka_tcp_pose`。如果使用 `--reuse-capture`，必须保证 `result/capture_tcp_pose.json` 与这张 RGB-D 的拍照时刻一致，或显式传入 `--capture-tcp-pose X Y Z RX RY RZ`。

放置物体时，机械臂到达 `--place-target-joint-pose-deg` 指定的关节位姿后，会默认沿机器人 base 坐标系的 Z 轴向下移动 50 mm，再张开夹爪。可通过以下参数调整下移距离：

```bash
--place-release-lower-mm 50
```

设置为 `0` 可取消放置前下移。该动作是笛卡尔直线运动，使用 `--velocity` 和 `--acceleration` 的速度参数。

## Git 备注

这台共享服务器上 GitHub SSH 可能会被 `LD_LIBRARY_PATH` 里的 conda OpenSSL 影响。如果出现 OpenSSL mismatch，可以临时清掉这个环境变量：

```bash
env -u LD_LIBRARY_PATH git status
```

如果需要从服务器 push 到 GitHub，可以使用当前验证过的 SSH 形式：

```bash
env -u LD_LIBRARY_PATH \
  GIT_SSH_COMMAND='ssh -i /home/admin128/.ssh/beilei_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o KexAlgorithms=ecdh-sha2-nistp256' \
  git push origin feat/GraspExecutionModule
```
##手操机械臂
```bash
/home/admin128/anaconda3/envs/smartgrasp310/bin/python plush5.py
```

MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --execute \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 20 \
  --acceleration 20 \
  --candidate-index 0


cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace

python scripts/print_jaka_tcp_pose.py \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc

## 运动到默认位置,拍照但是不抓取:
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --top-k 20 \
  --trial-log-subdir single_object \
  --trial-name capture_only \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 10 \
  --acceleration 10

## 拍照,抓取(循环,bbox 模式):
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --top-k 100 \
  --trial-log-subdir single_object \
  --trial-name grasp_execute \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 40 \
  --acceleration 20 \
  --loop \
  --execute \
  --grasp-input-mode bbox \
  --grasp-crop-margin-px 0 \
  --grasp-crop-margin-ratio 0 \
  --target-mask-center-tolerance-px 0

## 拍照,抓取(循环,严格 mask 模式):
MPLCONFIGDIR=/tmp/smartgrasp_mpl python scripts/realworld_grasp.py \
  --calibration-mode hand_eye \
  --hand-eye-calibration calibration/hand_eye_tcp_camera.json \
  --camera-serial 243122072659 \
  --top-k 100 \
  --trial-log-subdir single_object \
  --trial-name grasp_execute_mask \
  --jaka-python /home/admin128/anaconda3/envs/smartgrasp310/bin/python \
  --jkrc-dir /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/jkrc \
  --velocity 40 \
  --acceleration 20 \
  --loop \
  --execute \
  --grasp-input-mode mask \
  --target-mask-center-tolerance-px 0 
