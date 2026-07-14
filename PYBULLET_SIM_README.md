# SmartGrasp 执行层 JSON 接入说明

本文档只说明感知和推理模块如何调用执行层。项目路径：

```text
/home/admin128/qiuguanhe/Simulation/SmartGrasp
```

执行层统一入口：

```text
execution/run_execution.py
```

接入方式只有一种：

```text
上游生成 request JSON
-> 调用 execution/run_execution.py
-> 读取 response JSON
```

## 1. 调用命令

在服务器上运行：

```bash
cd /home/admin128/qiuguanhe/Simulation/SmartGrasp
conda activate smartgrasp

python execution/run_execution.py \
  --input <request_json路径> \
  --output <response_json路径>
```

示例：

```bash
python execution/run_execution.py \
  --input execution/examples/fully_visible_grasp_request.json \
  --output graspnet-workspace/results/test_fully_visible_response.json
```

```bash
python execution/run_execution.py \
  --input execution/examples/reveal_request.json \
  --output graspnet-workspace/results/test_reveal_response.json
```

## 2. Request JSON 格式

### Fully Visible：直接抓目标物体

当推理模块判断目标完整可见时，发送：

```json
{
  "request_id": "scene_0001_step_00",
  "branch": "fully_visible",
  "task_type": "grasp",
  "object": {
    "id": 3,
    "name": "medium_clamp",
    "category": "clamp",
    "role": "target"
  },
  "scene": {
    "scene_config": "graspnet-workspace/config/industrial_scene.json",
    "rgb_path": null,
    "depth_path": null,
    "mask_path": null,
    "point_cloud_path": null,
    "camera_intrinsics_path": null,
    "occlusion_graph_path": null
  },
  "execution": {
    "top_k": 5,
    "device": "cuda:0",
    "output": "graspnet-workspace/results/scene_0001_step_00_grasp.json",
    "gui": false
  }
}
```

现成示例：

```text
execution/examples/fully_visible_grasp_request.json
```

### Occluded：先移动遮挡物

当推理模块判断目标被遮挡，需要先移开遮挡物时，发送：

```json
{
  "request_id": "scene_0001_step_01",
  "branch": "partially_occluded",
  "task_type": "reveal",
  "object": {
    "id": 6,
    "name": "adjustable_wrench",
    "category": "wrench",
    "role": "occluder"
  },
  "reveal": {
    "action_type": "push",
    "move_distance": 0.05,
    "direction": [1.0, 0.0, 0.0],
    "center_point": null
  },
  "scene": {
    "scene_config": "graspnet-workspace/config/industrial_scene.json",
    "rgb_path": null,
    "depth_path": null,
    "mask_path": "perception/output/scene_0001/object_6_mask.png",
    "point_cloud_path": "perception/output/scene_0001/object_6_points.npy",
    "camera_intrinsics_path": null,
    "occlusion_graph_path": "data/integrated_runs/scene_0001_query_tool/occlusion_graph.json"
  },
  "execution": {
    "output": "graspnet-workspace/results/scene_0001_step_01_reveal.json",
    "gui": false
  }
}
```

现成示例：

```text
execution/examples/reveal_request.json
```

`branch` 可以是：

```text
fully_visible
partially_occluded
fully_occluded
```

`task_type` 可以是：

```text
grasp
reveal
```

当前 reveal 物理执行只支持：

```text
reveal.action_type = push
```

## 3. 字段说明

必须关注的字段：

```text
request_id          本轮请求 ID，用于日志追踪。
branch              推理模块输出的分支。
task_type           grasp 或 reveal。
object.id           感知模块给的物体 ID。
object.name         执行层当前主要用它在仿真场景中找物体。
object.category     物体类别。
object.role         target 或 occluder。
scene.scene_config  PyBullet 场景配置。
execution.output    执行层详细结果 JSON。
```

可选/预留字段：

```text
scene.rgb_path
scene.depth_path
scene.mask_path
scene.point_cloud_path
scene.camera_intrinsics_path
scene.occlusion_graph_path
```

当前本地仿真阶段，执行层主要使用：

```text
object.name
scene.scene_config
branch
task_type
reveal.action_type
reveal.direction
reveal.move_distance
execution.output
```

`reveal.center_point` 可以填 `null`。如果填了中心点但明显不在目标物体附近，执行层会自动回退到 PyBullet 中该物体的 AABB 中心。

## 4. Response JSON 格式

执行结束后，读取 `--output` 指定的 response JSON。

response 主要字段：

```json
{
  "request_id": "scene_0001_step_01",
  "status": "finished",
  "success": false,
  "branch": "partially_occluded",
  "action_type": "reveal",
  "object": {
    "id": 6,
    "name": "adjustable_wrench",
    "category": "wrench",
    "role": "occluder"
  },
  "result": {
    "failure_reason": "push_displacement_below_threshold"
  },
  "artifacts": {
    "result_json": "/home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/scene_0001_step_01_reveal.json",
    "viz_data_pkl": "/home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/scene_0001_step_01_reveal_viz_data.pkl"
  },
  "request_reloop": true,
  "diagnostics": {}
}
```

上游只需要优先读：

```text
status
success
result.failure_reason
artifacts.result_json
artifacts.viz_data_pkl
request_reloop
diagnostics
```

判断规则：

```text
status == "finished"       执行层脚本跑完。
status == "failed"         执行层脚本报错或请求格式有问题。
success == true            本轮动作达到执行层成功判定。
success == false           本轮动作执行了，但没有达到成功判定。
request_reloop == true     上游需要重新获取 RGB-D，并重新跑感知/推理。
artifacts.result_json      详细仿真结果。
artifacts.viz_data_pkl     可视化/回放数据。
```

## 5. 上游最小接入代码

Python 调用示例：

```python
import json
import subprocess
from pathlib import Path

repo = Path("/home/admin128/qiuguanhe/Simulation/SmartGrasp")
request_path = repo / "execution/examples/reveal_request.json"
response_path = repo / "graspnet-workspace/results/test_reveal_response.json"

subprocess.run(
    [
        "python",
        "execution/run_execution.py",
        "--input",
        str(request_path),
        "--output",
        str(response_path),
    ],
    cwd=repo,
    check=True,
)

with response_path.open("r", encoding="utf-8") as f:
    response = json.load(f)

print(response["status"])
print(response["success"])
print(response["request_reloop"])
print(response["artifacts"]["result_json"])
```

## 6. 当前可用物体名

`object.name` 当前可填：

```text
phillips_screwdriver
flat_screwdriver
adjustable_wrench
power_drill
ycb_hammer
medium_clamp
large_clamp
two_color_hammer
small_clamp
battery
```

这些名字来自：

```text
graspnet-workspace/config/industrial_scene.json
```

## 7. 注意事项

- 队友只需要生成 request JSON，不需要直接调用 PyBullet、GraspNet 或 `demo_closed_loop.py`。
- `object.name` 必须和 `industrial_scene.json` 中的物体名一致。
- `execution.output` 是执行层详细结果文件；`--output` 是给上游读取的标准 response 文件。
- 当前真实 `mask_path / point_cloud_path` 还在接入中，本地仿真主要根据 `object.name` 操作 PyBullet 场景内物体。
- 当前 `push` 已接入 reveal 物理执行；`pick_and_place` 还未实现。
