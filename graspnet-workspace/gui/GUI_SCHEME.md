# GraspNet Dash Web GUI 方案说明

## 1. 目标

当前 GUI 的目标不是替代 PyBullet 做物理仿真，而是替代 PyBullet GUI 做结果查看和诊断。

PyBullet 继续作为仿真后端，推荐使用 `DIRECT` 模式运行；Dash Web GUI 负责展示已经保存的仿真结果，包括点云、抓取姿态、成功/失败标签、RGB/Depth 图像和抓取动画。

## 2. 当前实现位置

```text
graspnet-workspace/
├── gui/
│   ├── app.py          # Dash Web GUI 主程序
│   ├── README.md       # 简短运行说明
│   └── GUI_SCHEME.md   # 当前方案说明
```

启动入口：

```bash
cd /home/admin128/beilei/graspnet-workspace
conda run -n smartgrasp python gui/app.py \
  --host 0.0.0.0 \
  --port 8050 \
  --results results_phase3_002/results.json \
  --viz-data results_phase3_002/results_viz_data.pkl
```

本机浏览器访问：

```text
http://localhost:8050
```

如果在远程服务器上运行，需要通过 VS Code/Cursor 端口转发或 SSH port forwarding 将服务器 `8050` 转发到本地。

## 3. 输入数据

GUI 不重新运行模型，也不重新运行 PyBullet。它只读取已有结果文件。

### 3.1 results.json

支持两种格式。

格式一：字典格式：

```json
{
  "total": 5,
  "success": 3,
  "obj_path": ".../textured.obj",
  "object_position": [0.3, 0.0, 0.05],
  "grasps": [
    {
      "grasp_index": 0,
      "success": true,
      "score": 1.12,
      "lift_z": 0.18,
      "width": 0.08,
      "depth": 0.03,
      "translation": [0.3, 0.0, 0.05],
      "rotation": [[...], [...], [...]]
    }
  ]
}
```

格式二：列表格式：

```json
[
  {
    "grasp_index": 0,
    "success": true,
    "score": 1.12,
    "lift_z": 0.18,
    "width": 0.08,
    "depth": 0.03,
    "translation": [0.3, 0.0, 0.05],
    "rotation": [[...], [...], [...]]
  }
]
```

### 3.2 viz_data.pkl

GUI 需要搭配一个可视化数据文件，常见命名：

```text
results_viz_data.pkl
results_closed_loop_viz_data.pkl
viz_data.pkl
```

内部字段：

```python
{
    "rgb": np.ndarray,          # (H, W, 4), RGBA
    "depth": np.ndarray,        # (H, W), depth in meters
    "point_cloud": np.ndarray,  # (1, N, 3) or (N, 3)
    "grasp_trajectories": list  # optional
}
```

## 4. 页面功能

### 4.1 结果文件选择

GUI 会扫描 `graspnet-workspace/` 下兼容的 `*.json` 结果文件，并在左侧下拉框中列出。

切换结果文件后，GUI 会自动尝试匹配同目录下的 `results_viz_data.pkl` 或 `viz_data.pkl`。

### 4.2 3D 点云 + 抓取姿态

主视图使用 Plotly `Scatter3d` 和 `Mesh3d` 展示：

- 蓝色点：物体点云
- 灰色点：桌面/背景点云
- 绿色 grasp：成功
- 红色 grasp：失败
- 高亮实体夹爪：当前选中的 grasp

左侧控制项：

- `Top K`：显示得分最高的前 K 个抓取
- `Point samples`：控制点云采样数量
- `Score min`：按 score 过滤
- `Outcome`：显示 success、failed 或两者
- `Selected grasp`：选择当前高亮和动画回放的抓取

### 4.3 RGB 和 Depth 面板

页面下方显示虚拟相机保存的：

- RGB 图
- Depth 图

这用于对照 3D 点云中的物体位置和视角。

### 4.4 抓取动画

GUI 中包含一个合成动画视图，按当前选中的 grasp 展示：

1. 夹爪从 pre-grasp 位置接近
2. 夹爪闭合
3. 夹爪向上抬升
4. 如果该 grasp 是 success，物体点云跟随上升

注意：当前动画是基于 `results.json + point_cloud` 合成的解释性动画，不是 PyBullet 每一步真实记录的轨迹回放。

如果需要真实轨迹回放，需要在 `simulation/evaluator.py` 中保存每个仿真 step 的：

- gripper base pose
- left/right finger pose
- object pose
- constraint 状态

## 5. 夹爪渲染约定

当前 Dash GUI 的夹爪几何已经对齐 GraspNet API 的官方可视化函数：

```python
graspnetAPI.utils.utils.plot_gripper_pro_max(center, R, width, depth)
```

局部坐标约定：

```text
local x = depth / approach direction
local y = width / jaw opening direction
local z = gripper height
```

GUI 使用 `R @ local_vertices + center` 将夹爪从局部坐标变换到世界坐标。

夹爪由四个 box 组成：

- left finger
- right finger
- bottom
- tail

这比早期手写的 U 形线框更接近 GraspNet 原始几何，也避免了错误地把 `R[:, 2]` 当作接近方向。

## 6. 当前仿真评估和 GUI 的关系

当前 GUI 展示的是仿真输出结果，但不会改变仿真成功/失败判定。

当前 `simulation/evaluator.py` 的核心逻辑是：

1. 抓取中心离物体点云太远则直接失败
2. 夹爪移动到抓取位姿
3. 手指闭合
4. 创建 fixed constraint
5. 抬升夹爪
6. 如果物体高度超过阈值，则判定成功

这套评估还不是严格的真实物理抓取。

尤其需要注意：

- 当前成功依赖 fixed constraint，物体会被约束到夹爪上
- 当前没有在 evaluator 中执行完整的夹爪实体碰撞检查
- 因此 success 结果可能偏乐观

## 7. 碰撞检测现状

项目里已经有 GraspNet 的 model-free collision detector：

```text
utils/collision_detector.py
```

它的检测逻辑同样使用 GraspNet 坐标约定：

```text
x = depth / approach
y = width / opening
z = height
```

检测区域包括：

- left finger
- right finger
- bottom
- shifting / approach path

但是当前 closed-loop evaluator 还没有把这个 collision detector 接入抓取执行流程。

推荐后续改进：

1. 在 GraspNet 输出后先运行 `ModelFreeCollisionDetector`
2. 把 collision mask 写入 `results.json`
3. 在 GUI 里增加 collision 状态显示
4. 对 collision grasp 使用单独颜色或标签
5. 在 fixed constraint 之前执行碰撞过滤

## 8. 为什么不用 PyBullet GUI

PyBullet GUI 在这个项目中的问题：

- 点云显示能力弱
- GraspNet Top-K 候选难以清晰叠加
- 远程服务器显示不稳定
- 不适合做筛选、对比、统计和报告截图
- 对夹爪坐标轴、score、success/fail 信息表达不清楚

因此推荐架构是：

```text
PyBullet DIRECT backend
    -> results.json + viz_data.pkl
    -> Dash Web GUI
```

PyBullet 负责计算，Dash 负责查看。

## 9. 使用建议

日常查看结果：

```bash
cd /home/admin128/beilei/graspnet-workspace
conda run -n smartgrasp python gui/app.py \
  --host 0.0.0.0 \
  --port 8050 \
  --results results_phase3_002/results.json \
  --viz-data results_phase3_002/results_viz_data.pkl
```

本机浏览器：

```text
http://localhost:8050
```

如果页面不更新，强制刷新：

```text
Ctrl + Shift + R
```

如果浏览器显示 connection refused，先确认：

1. 服务器端 Dash 进程是否在运行
2. 本地是否转发了 `8050` 端口
3. 浏览器打开的是 `localhost:8050`，不是服务器公网 IP

## 10. 后续优先级

建议后续按以下顺序增强：

1. 将 `ModelFreeCollisionDetector` 接入 evaluator
2. 在 `results.json` 中记录 `collision`, `empty_grasp`, `min_dist`, `failure_reason`
3. GUI 中增加 failure reason 面板
4. 保存真实逐帧仿真轨迹，实现真实 PyBullet replay
5. 支持加载多个物体结果并做批量统计
6. 支持导出当前视角截图到报告目录
