# SmartGrasp 执行层 PyBullet 仿真完成计划

本文档记录执行层从当前状态到“完整仿真抓取闭环完成”的工作计划。后续每完成一项，都更新本文档中的状态、证据文件和下一步。

项目路径：

```text
/home/admin128/qiuguanhe/SmartGrasp
```

当前分支：

```text
feat/grasp_realworld
```

最后更新：

```text
2026-07-08
```

## 最终目标

执行层最终要做到：

```text
感知/推理模块输出一个 execution request JSON
  -> 执行层读取 object id / name / branch / mask / point cloud
  -> 如果 fully_visible：执行 GraspNet 抓取
  -> 如果 partially_occluded 或 fully_occluded：执行 reveal 动作
  -> PyBullet 中使用 JAKA Zu3 + Robotiq-85 完成动作
  -> 保存动作结果、失败原因、可视化数据、更新后的 RGB-D/seg/scene state
  -> 返回 response JSON，告诉上游是否 request_reloop
```

执行层不负责：

```text
自然语言解析
目标检测和 ID 标注
SAM 分割
遮挡关系图构建
InfoGain / Cost 推理
论文级评估指标设计
```

执行层负责：

```text
PyBullet 场景
工业工具物体模型加载
JAKA Zu3 + Robotiq-85 模型加载
mask/depth/point cloud 到抓取输入的转换
GraspNet 抓取候选调用和过滤
机械臂/夹爪动作执行
reveal push 或 pick-and-place
执行结果记录和闭环重新观测
```

## 总体进度

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 0. 基础仿真资产接入 | 已完成 | 工业工具物体、JAKA Zu3 + Robotiq 合体 URDF 已进入当前工作区 |
| 1. fully_visible 抓取原型 | 已完成 | 当前能跑 PyBullet 场景、GraspNet、JAKA+Robotiq 抓取评估 |
| 2. 统一 JSON 接口雏形 | 已完成 | `execution/run_execution.py` 已支持 fully_visible/grasp 转调当前脚本 |
| 3. reveal 物理执行 | 进行中 | push 物理执行入口已接入，待实际运行验证；pick-and-place 未实现 |
| 4. 上游真实 mask/点云接入 | 未完成 | 当前仍主要使用 PyBullet segmentation 模拟感知结果 |
| 5. ID/name/body_id 对齐 | 未完成 | 需要把感知 ID、推理 object、PyBullet body id 统一映射 |
| 6. 抓取成功率调优 | 未完成 | 当前流程能跑完，但抓取成功率还需要调 |
| 7. 闭环重新观测输出 | 未完成 | reveal/grasp 后需要输出更新后的 RGB-D/seg/scene state |
| 8. 多场景回归测试 | 未完成 | 需要固定几组工业场景验证抓取和 reveal |
| 9. 队友联调交付 | 未完成 | 需要用感知/推理真实输出跑通接口 |

## 阶段 0：基础仿真资产接入

状态：已完成

已完成内容：

- 工业工具 `.obj` 模型已放入 `graspnet-workspace/assets/objects/industrial_tools/`。
- 当前包括螺丝刀、扳手、电钻、锤子、夹子、电池等工具类模型。
- JAKA Zu3 + Robotiq-85 合体 URDF 已放入 `graspnet-workspace/assets/robots/jaka_zu3/`。
- 场景配置已写入 `graspnet-workspace/config/industrial_scene.json`。

关键文件：

```text
graspnet-workspace/config/industrial_scene.json
graspnet-workspace/assets/objects/industrial_tools/
graspnet-workspace/assets/robots/jaka_zu3/gazebo_jaka_zu3_robotiq.urdf
```

完成标准：

- PyBullet 能加载工业工具物体。
- PyBullet 能加载 JAKA Zu3 + Robotiq-85 合体 URDF。
- `industrial_scene.json` 中的物体路径都能解析。

后续维护：

- 新增工具物体时，统一放到 `graspnet-workspace/assets/objects/industrial_tools/<object_name>/`。
- 新物体必须在 `industrial_scene.json` 里写清楚 `name`、`category`、`position`、`scale`、`mass`、`friction`。

## 阶段 1：fully_visible 抓取原型

状态：已完成，但需要继续调优成功率

已完成内容：

- `demo_closed_loop.py` 支持多物体场景。
- 支持 `--scene-config` 和 `--target-object`。
- 使用虚拟 RGB-D 相机生成点云。
- 使用 PyBullet segmentation 获取目标物体点云。
- 使用 GraspNet 生成候选抓取。
- 按目标物体点云过滤候选。
- 使用 JAKA Zu3 + Robotiq-85 执行 top-k 抓取。
- 保存 JSON 和 PKL 结果。

关键文件：

```text
run_grasp_simulation.sh
graspnet-workspace/scripts/demo_closed_loop.py
graspnet-workspace/simulation/scene.py
graspnet-workspace/simulation/camera.py
graspnet-workspace/simulation/robot_gripper.py
graspnet-workspace/simulation/evaluator.py
```

当前可运行命令：

```bash
cd /home/admin128/qiuguanhe/SmartGrasp

bash run_grasp_simulation.sh \
  --scene-config config/industrial_scene.json \
  --target-object medium_clamp \
  --top_k 5
```

当前问题：

- 流程能跑完，但抓取成功率低。
- 之前运行出现过 `0/5 成功`。
- 2026-07-08 通过 `execution/run_execution.py` 运行 fully_visible 示例，请求和结果文件均生成成功，但抓取结果仍为 `0/5 成功`。
- 这说明入口、场景、GraspNet、JAKA+Robotiq 执行链路基本打通，但动作质量还需要调。

下一步调优点：

- 检查 GraspNet 输出坐标系和 PyBullet world 坐标是否完全一致。
- 检查抓取中心高度是否过低或过高。
- 检查 JAKA TCP 和 Robotiq fingertip 的偏置。
- 检查夹爪闭合宽度是否和候选 `width` 对应。
- 检查 approach 方向是否容易撞桌面。
- 检查物体质量、摩擦、碰撞模型是否合理。
- 检查成功判定是不是只看夹爪闭合而忽略物体抬升。

完成标准：

- 至少对 3 类工具物体能稳定跑完 fully_visible 抓取流程。
- 每次输出明确的 `success`、`failure_reason`、`obj_lift_delta`、`grasped_by_gripper`。
- 失败时能从 JSON 里看出是候选问题、IK 问题、碰撞问题、夹爪问题还是判定问题。

## 阶段 2：统一 JSON 接口雏形

状态：已完成，但还需要扩展真实输入

已完成内容：

- 新增统一入口 `execution/run_execution.py`。
- 新增 fully_visible 示例请求。
- 新增 reveal 示例请求。
- `fully_visible + grasp` 会转调 `run_grasp_simulation.sh`。
- `reveal` 当前会返回 reveal plan，并标记物理执行未完成。

关键文件：

```text
execution/run_execution.py
execution/examples/fully_visible_grasp_request.json
execution/examples/reveal_request.json
PYBULLET_SIM_README.md
```

当前可运行命令：

```bash
cd /home/admin128/qiuguanhe/SmartGrasp

python execution/run_execution.py \
  --input execution/examples/fully_visible_grasp_request.json \
  --output graspnet-workspace/results/scene_0001_step_00_response.json
```

当前接口请求格式：

```text
request_id
branch
task_type
object.id
object.name
object.category
object.role
scene.scene_config
scene.rgb_path
scene.depth_path
scene.mask_path
scene.point_cloud_path
scene.camera_intrinsics_path
scene.occlusion_graph_path
execution.top_k
execution.device
execution.output
execution.gui
```

需要改进：

- `run_execution.py` 目前对 fully_visible 主要依赖 `object.name`。
- 还没有真正读取 `mask_path` 或 `point_cloud_path`。
- reveal 还没有物理执行，只返回计划。

完成标准：

- 感知/推理队友只需要生成 request JSON，不需要知道执行层内部脚本。
- 执行层固定输出 response JSON。
- response JSON 中包含：
  - `status`
  - `success`
  - `branch`
  - `action_type`
  - `object`
  - `result`
  - `artifacts`
  - `request_reloop`
  - `diagnostics`

## 阶段 3：reveal 物理执行

状态：进行中

目标：

让 `partially_occluded` 和 `fully_occluded` 分支不只是返回动作计划，而是真正在 PyBullet 中用 JAKA+Robotiq 对遮挡物执行 push 或小幅 pick-and-place。

当前已完成内容：

- 新增 `graspnet-workspace/simulation/reveal_push.py`。
- `execution/run_execution.py` 的 reveal push 分支已经改为调用 PyBullet JAKA push。
- reveal push 会加载工业场景，按 `object.name` 找到遮挡物，执行侧向 push。
- reveal push 会保存动作前后物体位姿、`frame_log`、before/after RGB-D/seg/point cloud。
- `pick_and_place` 仍未实现，当前只支持 `push` 的物理执行。

当前待验证：

- 已运行一次 reveal push，入口可生成 response/result/viz 文件，但第一次失败。
- 第一次失败原因：示例 request 写死的 `center_point` 和实际 `adjustable_wrench` AABB 中心偏差过大，push 打空，物体位移约为 0。
- 已修正：当上游 `center_point` 缺失或明显不在目标物体 AABB 附近时，执行层自动回退到 PyBullet AABB 中心。
- 已修正：`frame_log` 中的 `gripper_pos` 改为记录 JAKA TCP 位姿，而不是固定机器人底座。
- 第二次运行时中心点已正确回退到 AABB 中心，JAKA TCP 也在动，但物体位移仍约为 0。
- 第二次失败判断：TCP 轨迹接触深度不足，Robotiq 碰撞体没有有效推到目标物体。
- 已修正：push 轨迹现在会从物体近侧外部进入，并额外穿过 AABB 一段距离；接触高度提高到物体 AABB 中部和桌面安全高度以上。
- 2026-07-08 通过 `execution/run_execution.py` 再次运行 reveal 示例，请求和结果文件均生成成功，但 `signed_displacement` 仍约为 0，结果为 `push_displacement_below_threshold`。该结果需要用最新 push 接触深度代码重新运行确认。
- 需要再次运行验证，确认 JAKA+Robotiq 的 push 接触是否能让遮挡物发生合理位移。
- 需要确认生成的 JSON/PKL 能被现有 GUI 正常读取。

需要实现的能力：

```text
输入 occluder object id/name/category
-> 找到 PyBullet 中对应物体
-> 获取遮挡物中心或点云
-> 规划一个安全 push 或 pick-and-place 动作
-> JAKA 运动到预接触位姿
-> 执行 3-5cm 微动
-> 保存动作前后物体位姿
-> 重新拍 RGB-D/seg
-> 返回 request_reloop=true
```

建议优先实现 push，再实现 pick-and-place。

推荐实现路径：

1. 在 `graspnet-workspace/simulation/evaluator.py` 或新模块中增加 reveal 执行函数。
2. 输入为：

```text
object_id
center_point
direction
move_distance
action_type
```

3. 使用 `JakaZu3Robotiq85Gripper` 规划一个侧向推动作。
4. 保存 `frame_log`，方便 GUI 回放。
5. 已将 `execution/run_execution.py` 的 reveal push 分支从 `not_implemented` 改成真实调用；下一步需要运行验证并根据结果调动作参数。

关键文件：

```text
execution/reveal_api.py
execution/run_execution.py
graspnet-workspace/simulation/reveal_push.py
graspnet-workspace/simulation/robot_gripper.py
graspnet-workspace/simulation/evaluator.py
graspnet-workspace/scripts/demo_closed_loop.py
```

当前推荐验证命令：

```bash
cd /home/admin128/qiuguanhe/SmartGrasp

python execution/run_execution.py \
  --input execution/examples/reveal_request.json \
  --output graspnet-workspace/results/scene_0001_step_01_reveal_response.json
```

完成标准：

- 运行 `execution/examples/reveal_request.json` 时，PyBullet 中遮挡物发生可观测位移。
- response JSON 不再是 `status=not_implemented`。
- 输出动作前后物体位姿。
- 输出新的 RGB-D/seg 或对应的 viz data。
- 返回 `request_reloop=true`。

## 阶段 4：上游真实 mask / 点云接入

状态：未完成

目标：

让执行层真正使用感知模块输出的 mask 或点云，而不是只用 PyBullet segmentation。

当前情况：

```text
当前 fully_visible 抓取使用 PyBullet segmentation 获取 target_points。
这适合仿真自测，但不是最终和感知模块联调的形式。
```

需要支持的输入：

```text
rgb_path
depth_path
mask_path
camera_intrinsics_path
point_cloud_path
```

两条接入路线：

路线 A：mask + depth 生成点云

```text
读取 RGB
读取 depth
读取 mask
读取 intrinsics
反投影得到目标局部点云
送入 GraspNet
```

路线 B：直接读取 point_cloud

```text
读取上游保存的 .npy / .ply / .pcd
检查坐标系
送入 GraspNet
```

建议优先做路线 A，因为它和感知模块的 SAM mask 输出最贴近。

需要注意：

- 深度单位必须明确，毫米还是米。
- mask 分辨率必须和 RGB-D 对齐。
- 点云坐标系必须明确，是相机坐标还是 PyBullet world 坐标。
- 如果是相机坐标，需要提供 camera pose 或外参，转换到仿真 world。

关键文件：

```text
execution/pointcloud_utils.py
execution/run_execution.py
graspnet-workspace/scripts/demo_closed_loop.py
graspnet-workspace/simulation/camera.py
```

完成标准：

- `execution/run_execution.py` 能读取 `mask_path + depth_path + intrinsics_path`。
- 能生成目标物体点云。
- 能把点云送入 GraspNet 或转成当前 `demo_closed_loop.py` 可用的数据结构。
- response JSON 记录使用的是 `pybullet_segmentation` 还是 `upstream_mask`。

## 阶段 5：ID / name / body_id 对齐

状态：未完成

目标：

把感知模块的 object ID、推理模块的 object、PyBullet body id、场景配置中的 object name 对齐，避免“推理说抓 3 号物体，但 PyBullet 抓了另一个”的问题。

当前情况：

```text
当前执行层主要靠 object.name / --target-object。
PyBullet 内部还有 body_id。
感知模块会输出自己的 object.id。
```

需要建立映射表：

```json
{
  "perception_id": 3,
  "object_name": "medium_clamp",
  "category": "clamp",
  "pybullet_body_id": 8,
  "mask_path": "...",
  "point_cloud_path": "..."
}
```

推荐实现：

- 场景加载后导出 `scene_registry.json`。
- 每个物体包含：
  - `name`
  - `category`
  - `pybullet_body_id`
  - `initial_pose`
  - `current_pose`
  - `metadata`
- 如果上游只给 `object.id`，执行层能通过 registry 找到 `object.name`。

关键文件：

```text
graspnet-workspace/simulation/scene.py
execution/run_execution.py
graspnet-workspace/config/industrial_scene.json
```

完成标准：

- request JSON 可以只给 `object.id`，也可以给 `object.name`。
- 执行层能稳定找到目标物体。
- response JSON 中同时返回 upstream object id 和 PyBullet body id。

## 阶段 6：抓取执行成功率调优

状态：未完成

目标：

让 fully_visible 抓取不只是能跑，而是能较稳定地抓起常见工业工具物体。

优先调试顺序：

1. 坐标系
2. TCP 和夹爪偏置
3. 抓取高度
4. 夹爪宽度映射
5. approach 和 retreat 路径
6. 碰撞模型
7. 质量和摩擦
8. 成功判定

建议先用 scripted grasp 测执行层，不要一上来调 GraspNet：

```bash
cd /home/admin128/qiuguanhe/SmartGrasp/graspnet-workspace

/home/admin128/anaconda3/envs/smartgrasp/bin/python scripts/test_scripted_grasp.py \
  --scene-config config/industrial_scene.json \
  --target-object medium_clamp \
  --top_k 4
```

如果 scripted grasp 都抓不起来，问题主要在执行层。

如果 scripted grasp 可以抓起来，但 GraspNet 抓不起来，问题主要在候选位姿、点云输入或过滤。

完成标准：

- scripted grasp 至少能让一类物体出现成功样例。
- GraspNet fully_visible 至少能对几类工具产生可解释的成功或失败结果。
- 每个失败结果有明确 `failure_reason`。

## 阶段 7：闭环重新观测输出

状态：未完成

目标：

执行动作后，输出更新后的场景观测，供输入/感知模块进入下一轮。

需要输出：

```text
updated_rgb_path
updated_depth_path
updated_seg_path
updated_point_cloud_path
updated_scene_registry_path
result_json
viz_data_pkl
request_reloop
```

当前缺口：

- 当前结果主要保存执行结果和 viz data。
- 没有形成标准的“动作后重新观测包”。

推荐目录：

```text
graspnet-workspace/results/<request_id>/
  request.json
  response.json
  before_rgb.npy 或 .png
  before_depth.npy
  before_seg.npy
  after_rgb.npy 或 .png
  after_depth.npy
  after_seg.npy
  scene_registry_before.json
  scene_registry_after.json
  viz_data.pkl
```

完成标准：

- grasp 和 reveal 都能输出动作后的 RGB-D/seg。
- 上游可以直接把 `updated_*` 路径作为下一轮输入。

## 阶段 8：多场景回归测试

状态：未完成

目标：

固定几组工业工具场景，防止后续改代码时把已有功能改坏。

建议场景：

```text
scene_001_fully_visible_screwdriver
scene_002_fully_visible_clamp
scene_003_partially_occluded_screwdriver
scene_004_fully_occluded_tool_under_clamp
scene_005_mixed_tools_dense_table
```

每个场景记录：

```text
scene_config
target object
branch
expected action type
expected response fields
known limitations
```

执行层验收指标：

```text
脚本不崩溃
能输出 response JSON
能输出 result artifacts
失败原因可解释
reveal 后物体位姿发生合理变化
request_reloop 正确
```

注意：这里不是做感知/推理的论文评估，只是执行层回归测试。

完成标准：

- 至少 5 个固定场景可重复运行。
- 每个场景都有结果 JSON。
- 每个失败都有可读原因。

## 阶段 9：队友联调交付

状态：未完成

目标：

让感知和推理队友真正能调用执行层，而不是只看 README。

联调输入：

```text
自然语言解析后的 task_type
推理后的 branch
目标或遮挡物 object id/category/name
mask_path
depth_path
rgb_path
camera_intrinsics_path
occlusion_graph_path
```

联调输出：

```text
response JSON
动作结果
失败原因
updated RGB-D/seg
request_reloop
```

推荐联调流程：

1. 推理模块先手写一个 fully_visible request JSON。
2. 执行层跑 `run_execution.py`。
3. 感知模块提供真实 mask/depth。
4. 执行层切换到真实 mask 点云输入。
5. 推理模块提供 partially_occluded request JSON。
6. 执行层跑 reveal。
7. 执行层返回 updated observation。
8. 输入/感知模块开始下一轮。

完成标准：

- 队友不需要手动调用 `demo_closed_loop.py`。
- 队友只需要生成 request JSON。
- 执行层输出 response JSON 和 artifacts。
- 三个分支都能走通：
  - `fully_visible -> grasp`
  - `partially_occluded -> reveal`
  - `fully_occluded -> reveal`

## 当前最推荐的下一步

优先级从高到低：

1. 实现 reveal 的 PyBullet 物理执行。
2. 运行 `execution/examples/reveal_request.json` 验证 reveal push 是否真的推动物体。
3. 根据验证结果调整 push 接触点、方向、距离和 JAKA TCP 姿态。
4. 接入上游真实 `mask_path + depth_path + intrinsics_path`。
5. 做 ID/name/body_id 映射。
6. 调 scripted grasp，让至少一个工具物体能被 JAKA+Robotiq 成功抓起。
7. 再回头调 GraspNet fully_visible 抓取成功率。
8. 建立 5 个固定场景回归测试。

## 每次更新本文档的规则

每完成一个任务，必须更新：

```text
总体进度表的状态
对应阶段的“已完成内容”
对应阶段的“当前问题”
对应阶段的“完成标准”是否满足
最后更新时间
新增的关键文件或结果文件
```

状态只使用：

```text
未完成
进行中
已完成
已完成，但需要调优
阻塞
```

如果任务阻塞，必须写清楚：

```text
阻塞原因
需要谁提供什么
临时替代方案
```

## 当前交付物清单

文档：

```text
PYBULLET_SIM_README.md
PYBULLET_SIM_COMPLETION_PLAN.md
```

统一入口：

```text
execution/run_execution.py
```

请求示例：

```text
execution/examples/fully_visible_grasp_request.json
execution/examples/reveal_request.json
```

仿真入口：

```text
run_grasp_simulation.sh
graspnet-workspace/scripts/demo_closed_loop.py
graspnet-workspace/scripts/test_scripted_grasp.py
```

核心仿真模块：

```text
graspnet-workspace/simulation/scene.py
graspnet-workspace/simulation/camera.py
graspnet-workspace/simulation/robot_gripper.py
graspnet-workspace/simulation/evaluator.py
```

资产和配置：

```text
graspnet-workspace/config/industrial_scene.json
graspnet-workspace/assets/objects/industrial_tools/
graspnet-workspace/assets/robots/jaka_zu3/gazebo_jaka_zu3_robotiq.urdf
```
