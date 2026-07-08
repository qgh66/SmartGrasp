# SmartGrasp 执行层 JSON 接入说明

本文档只说明感知和推理模块如何调用执行层。项目路径：

```text
/home/admin128/qiuguanhe/SmartGrasp
```

## 0. 队友必须提供/确认的信息

为了让执行层使用真实感知结果，而不是只用 PyBullet segmentation，请感知/推理模块先确认下面这些字段：

```text
1. point_cloud_path 是否能提供：
   - 推荐格式：.npy 或 .npz
   - shape：N x 3 或 N x 6，前三列必须是 xyz
   - 坐标系：优先 PyBullet/world frame
   - 单位：优先 meter

2. 如果不能直接提供 world-frame 点云，需要提供：
   - mask_path
   - depth_path
   - camera_intrinsics_path
   - depth_scale
   - camera_to_world_path

3. mask/depth 格式需要确认：
   - mask：PNG/JPG 或 .npy/.npz，非 0 像素表示目标物体
   - depth：.npy/.npz 或图像，乘 depth_scale 后单位必须是 meter
   - intrinsics：JSON，包含 fx/fy/cx/cy，或 K/camera_matrix

4. object.id 和 object.name 需要稳定：
   - object.id 是感知/推理链路里的物体 ID
   - object.name 必须能对应 PyBullet scene_config 中的物体 name
```

当前执行层的输入优先级：

```text
优先级 1：scene.point_cloud_path
  -> 直接读取目标物体点云
  -> 如果 point_cloud_frame=world，则用于 GraspNet 抓取

优先级 2：scene.mask_path + scene.depth_path + scene.camera_intrinsics_path
  -> 执行层反投影生成目标物体点云
  -> 如果提供 camera_to_world_path，则转换到 world frame 后用于抓取

优先级 3：以上都没有
  -> fallback 到 PyBullet segmentation
  -> 用于本地仿真自测
```

注意：抓取执行最终需要 **PyBullet/world frame、meter 单位** 的点云。camera frame 点云目前只会记录诊断，不会直接用于真实抓取执行。

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
cd /home/admin128/qiuguanhe/SmartGrasp
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
  --input execution/examples/upstream_point_cloud_grasp_request.json \
  --output graspnet-workspace/results/test_upstream_point_cloud_response.json
```

```bash
python execution/run_execution.py \
  --input execution/examples/reveal_request.json \
  --output graspnet-workspace/results/test_reveal_response.json
```

## 2. Request JSON 格式

### Fully Visible：直接抓目标物体

当推理模块判断目标完整可见时，推荐发送 world-frame 点云：

```json
{
  "request_id": "scene_0001_step_00_upstream_pc",
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
    "point_cloud_path": "perception/output/scene_0001/object_3_points_world.npy",
    "point_cloud_frame": "world",
    "point_cloud_unit": "meter",
    "camera_intrinsics_path": null,
    "camera_to_world_path": null,
    "occlusion_graph_path": "data/integrated_runs/scene_0001_query_tool/occlusion_graph.json"
  },
  "execution": {
    "top_k": 5,
    "device": "cuda:0",
    "output": "graspnet-workspace/results/scene_0001_step_00_upstream_pc_grasp.json",
    "gui": false
  }
}
```

现成示例：

```text
execution/examples/upstream_point_cloud_grasp_request.json
```

如果上游还没有点云，可以先发送 fallback 版本：

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
    "point_cloud_frame": "world",
    "point_cloud_unit": "meter",
    "camera_intrinsics_path": null,
    "camera_to_world_path": null,
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
    "point_cloud_frame": "world",
    "point_cloud_unit": "meter",
    "camera_intrinsics_path": null,
    "camera_to_world_path": null,
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
scene.point_cloud_path        上游目标点云，推荐 world frame + meter。
scene.point_cloud_frame       world 或 camera；当前抓取执行要求 world。
scene.point_cloud_unit        meter 或 millimeter；millimeter 会自动转 meter。
scene.mask_path               目标物体 mask，用于 mask+depth 反投影。
scene.depth_path              深度图，用于 mask+depth 反投影。
scene.camera_intrinsics_path  相机内参 JSON。
scene.camera_to_world_path    camera frame 到 PyBullet/world frame 的 4x4 外参 JSON。
scene.depth_scale             depth 乘这个比例后得到 meter；默认 1.0。
execution.output    执行层详细结果 JSON。
```

当前支持的点云/图像输入：

```text
.npy: 直接保存 numpy array
.npz: 支持 points / point_cloud / xyz / arr_0 / depth / mask 等 key
mask 图像: PNG/JPG，非 0 像素表示目标物体
intrinsics JSON: {"fx":..., "fy":..., "cx":..., "cy":...}
外参 JSON: {"T_world_camera": [[... 4x4 ...]]}
```

当前本地仿真阶段，执行层主要使用：

```text
object.name
scene.scene_config
scene.point_cloud_path 或 mask/depth/intrinsics
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
    "result_json": "/home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/results/scene_0001_step_01_reveal.json",
    "viz_data_pkl": "/home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace/results/scene_0001_step_01_reveal_viz_data.pkl"
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

`diagnostics.target_point_source` 会说明本次抓取点云来源：

```text
point_cloud_path              使用了上游点云
mask_depth_intrinsics         使用 mask + depth + intrinsics 生成点云
pybullet_segmentation_fallback 没有上游点云，回退到 PyBullet segmentation
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

repo = Path("/home/admin128/qiuguanhe/SmartGrasp")
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
