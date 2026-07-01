# SmartGrasp Grasp Execution Module

这个分支保存了当前的 milestone 版本：基于 **GraspNet + PyBullet** 的单物体抓取仿真，以及用于查看结果和播放抓取动画的 **Dash GUI**。

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
├── scripts/demo_closed_loop.py # 主入口：建场景 -> 拍 RGB-D -> GraspNet -> 仿真评估
├── simulation/                 # PyBullet 场景、相机、夹爪、抓取评估器
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
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
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
/home/admin128/beilei/graspnet-baseline/checkpoints/checkpoint-rs.tar
```

也可以放到：

```text
graspnet-workspace/checkpoints/checkpoint-rs.tar
```

2. 一个待抓取物体 mesh，例如：

```text
/home/admin128/beilei/obj_phase3/002/textured.obj
```

运行时也可以通过 `--obj` 指定其他 `.obj` 文件。

## 运行闭环仿真

推荐从仓库根目录下面的 `graspnet-workspace` 运行：

```bash
conda activate smartgrasp
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

python scripts/demo_closed_loop.py \
  --obj /home/admin128/beilei/obj_phase3/002/textured.obj \
  --ckpt /home/admin128/beilei/graspnet-baseline/checkpoints/checkpoint-rs.tar \
  --top_k 5 \
  --device cuda:0 \
  --output results_phase3_002/results.json
```

如果只是调试流程、没有可用 GPU，可以用 CPU 跑小规模测试：

```bash
python scripts/demo_closed_loop.py \
  --obj /home/admin128/beilei/obj_phase3/002/textured.obj \
  --ckpt /home/admin128/beilei/graspnet-baseline/checkpoints/checkpoint-rs.tar \
  --top_k 5 \
  --device cpu \
  --output results_phase3_002/results.json
```

这个脚本会依次完成：

1. 在 PyBullet 中搭建桌面和单物体场景。
2. 加载物体 mesh，并记录物体位姿。
3. 用虚拟相机拍 RGB-D。
4. 将 depth 转成点云。
5. 调用 GraspNet 生成候选抓取。
6. 用 PyBullet evaluator 检查 top-k 抓取。
7. 保存 JSON 结果和 GUI 可视化数据。

输出文件通常是：

```text
results_phase3_002/results.json
results_phase3_002/results_viz_data.pkl
```

其中：

- `results.json`：抓取分数、位姿、宽度、深度、成功/失败、失败原因、动画轨迹日志。
- `results_viz_data.pkl`：RGB、depth、点云、物体路径、物体姿态等 GUI 需要的数据。

## 运行 Reveal Push 仿真（JAKA ZU3 可视化）

`scripts/demo_reveal_push.py` 在 `graspnet-workspace/` 下面，所以必须先进入这个目录。不要在 `SmartGrasp/` 根目录直接运行 `python scripts/demo_reveal_push.py`。

推荐命令：

```bash
conda activate smartgrasp
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

bash scripts/run_reveal_push_jaka.sh
```

当前默认 `--robot-model jaka`，会加载仓库内复制好的 JAKA ZU3 + Robotiq URDF 做网页回放可视化：

```text
/home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace/assets/robots/jaka_zu3/gazebo_jaka_zu3_robotiq.urdf
```

说明：

- JAKA ZU3 模型用于可视化和 IK 跟随。
- JAKA/Robotiq 的 URDF 和 mesh 都在 `graspnet-workspace/assets/robots/jaka_zu3/`，不再依赖外部机器人模型目录。
- push 接触仍由原来的简化夹爪碰撞体完成，避免复杂 URDF 碰撞导致仿真不稳定。
- 如果要临时切回原来的简化模型，添加 `--robot-model simple`。

切回简化模型：

```bash
bash scripts/run_reveal_push_simple.sh
```

真实 RGB-D + mask 数据的 push 仿真：

```bash
RGB=/path/to/aligned_rgb.jpg \
DEPTH=/path/to/aligned_depth.npy \
MASK=/path/to/object_mask.png \
INTRINSICS=/path/to/camera_intrinsics.json \
OUTPUT=results/reveal_push_real_rgbd.json \
bash scripts/run_reveal_push_real_rgbd.sh
```

查看 JAKA push 回放：

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

## 当前仿真约束

当前 milestone 不是只展示网络输出，而是额外加了物理合理性限制：

- 夹爪不能从桌面下方接近物体。
- 抓取中心必须落在物体点云附近，不能隔空抓取。
- GraspNet 的 local X 轴作为 approach/depth 方向。
- GraspNet 的 local Y 轴作为夹爪开合方向。
- lift 阶段动画中，夹爪和物体使用相同的 z 方向位移，避免两者上升速度不一致。

相关常量在：

```text
graspnet-workspace/simulation/evaluator.py
graspnet-workspace/gui/app.py
```

当前默认值：

```text
TABLE_Z = 0.0
TABLE_CLEARANCE = 0.005
MAX_GRASP_CENTER_DIST = 0.04
```

## 启动 Dash GUI

GUI 不会重新运行 GraspNet，也不会重新跑 PyBullet。它只读取已经保存好的：

```text
results.json
results_viz_data.pkl
```

启动命令：

```bash
conda activate smartgrasp
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

python gui/app.py \
  --host 0.0.0.0 \
  --port 8050 \
  --results results_phase3_002/results.json \
  --viz-data results_phase3_002/results_viz_data.pkl
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
  "total": 5,
  "success": 1,
  "obj_path": "/path/to/textured.obj",
  "object_position": [0.3, 0.0, 0.05],
  "object_orientation": [0.0, 0.0, 0.0, 1.0],
  "grasps": [
    {
      "grasp_index": 0,
      "success": false,
      "score": 1.0,
      "lift_z": 0.04,
      "width": 0.06,
      "depth": 0.03,
      "translation": [0.3, 0.0, 0.05],
      "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "failure_reason": "approach_below_table"
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
  --results path/to/results.json \
  --viz-data path/to/results_viz_data.pkl
```

浏览器中可以用 `Ctrl + Shift + R` 强制刷新。

### 动画里物体不是完整模型

优先使用 `scripts/demo_closed_loop.py` 重新生成结果。新的结果会保存 `object_orientation`，GUI 才能更稳定地按真实姿态渲染完整 mesh。

### 抓取看起来过于理想

当前 evaluator 在夹爪闭合后使用固定约束来模拟抓住物体，因此它适合 milestone 展示和方案验证，不等价于完整的接触物理 benchmark。

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

---

# 常用仿真命令（直接复制）

所有命令都从 `graspnet-workspace` 目录运行：

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
```

## 1. 生成 JAKA ZU3 + Robotiq 可视化的 push 仿真结果

```bash
conda run -n smartgrasp bash scripts/run_reveal_push_jaka.sh
```

## 2. 打开网页查看 JAKA push 回放

```bash
conda run --no-capture-output -n smartgrasp bash scripts/run_reveal_push_gui.sh
```

默认端口：

```text
http://127.0.0.1:8051
```

## 3. 使用原简化夹爪做 push 仿真

```bash
conda run -n smartgrasp bash scripts/run_reveal_push_simple.sh
```

## 4. 使用真实 RGB-D + mask 做 push 仿真

```bash
RGB=/path/to/aligned_rgb.jpg \
DEPTH=/path/to/aligned_depth.npy \
MASK=/path/to/object_mask.png \
INTRINSICS=/path/to/camera_intrinsics.json \
conda run -n smartgrasp bash scripts/run_reveal_push_real_rgbd.sh
```

## 5. 生成 JAKA ZU3 + Robotiq 可视化的 GraspNet 抓取仿真

```bash
conda activate smartgrasp
bash scripts/run_grasp_closed_loop.sh
```

默认启用展示模式：如果 GraspNet 候选中心偏到桌面，会吸附到物体点云中心附近再执行完整 JAKA 回放。这样用于检查 JAKA/夹爪展示效果，不等价于严格抓取 benchmark。

如需关闭展示模式，使用严格候选抓取判定：

```bash
DEMO_SNAP_TO_OBJECT=0 bash scripts/run_grasp_closed_loop.sh
```

如需回退到原简化夹爪抓取仿真：

```bash
ROBOT_MODEL=simple \
PYTORCH_NVML_BASED_CUDA_CHECK=0 bash scripts/run_grasp_closed_loop.sh
```

## 6. 打开网页查看抓取仿真回放

```bash
RESULTS=results/grasp_closed_loop.json \
VIZ_DATA=results/grasp_closed_loop_viz_data.pkl \
conda run --no-capture-output -n smartgrasp bash scripts/run_reveal_push_gui.sh
```

## 7. 旧版 GraspNet + PyBullet 命令行抓取验证

```bash
conda run -n smartgrasp bash scripts/run_grasp_simulation.sh
```

## 8. 纯 GraspNet 推理 Demo

```bash
conda run -n smartgrasp bash scripts/run_inference.sh
```
