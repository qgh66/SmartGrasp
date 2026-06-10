# GraspNet Dash GUI

这是 `graspnet-workspace` 的浏览器可视化页面，用来读取已经保存的 GraspNet + PyBullet 仿真结果，并展示点云、RGB-D、抓取动画和 Reveal push 轨迹。

GUI 只负责展示，不会重新运行 GraspNet 推理或 PyBullet 仿真。

## 启动方式

```bash
conda activate smartgrasp
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
python gui/app.py \
  --host 0.0.0.0 \
  --port 8050 \
  --results results_phase3_002/results.json \
  --viz-data results_phase3_002/results_viz_data.pkl
```

本机打开：

```text
http://127.0.0.1:8050
```

远程服务器运行时，需要通过 VS Code/Cursor 端口转发或 SSH 转发 `8050`，然后在本地浏览器打开同样的地址。

## 输入文件

- `results.json`：抓取分数、位姿、宽度、深度、lift 结果、成功/失败标签、失败原因、轨迹日志。
- `results_viz_data.pkl`：RGB 图、depth 图、点云、物体 mesh 路径、物体姿态。

GUI 也会扫描 workspace 中兼容的结果 JSON，并允许在侧边栏切换。

## Reveal Push 仿真

先运行无头 PyBullet 仿真，生成真实逐帧 push 轨迹：

```bash
conda activate smartgrasp
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
python scripts/demo_reveal_push.py \
  --distance 0.05 \
  --output results/reveal_push.json
```

使用自己的遮挡物模型时增加 `--obj /path/to/object.obj`。需要同时观察
PyBullet 窗口时增加 `--gui`。

然后启动 Dash 回放：

```bash
python gui/app.py \
  --results results/reveal_push.json \
  --viz-data results/reveal_push_viz_data.pkl
```

`reveal_api.py` 负责生成 +X 扰动计划，`simulation/evaluator.py` 负责执行物理
push 并写入 `frame_log`，Dash 只读取和回放这些帧。

## 页面内容

- 第一张 3D 图显示点云和候选动作位姿，不渲染完整夹爪。
- 动画区域优先回放 `frame_log` 中的真实 PyBullet 轨迹；旧结果没有轨迹时才使用合成抓取动画。
- RGB 和 Depth 区域显示虚拟相机输入。
