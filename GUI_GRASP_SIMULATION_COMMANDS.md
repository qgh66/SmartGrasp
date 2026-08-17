# SmartGrasp Perception + Reason GUI 批量抓取命令

本文包含两类测试流程：

1. 新增的 Perception + Reason 核验清场流程：不执行机械臂抓取，核验正确后直接从仿真场景删除物体。
2. 原有的 Perception + Reason + GraspNet + 机械臂物理抓取流程。

当前保留以下五个 GUI 场景命令：

1. 平铺场景依次核验并删除全部 10 个物体。
2. 堆叠场景依次核验并删除全部 12 个物体。
3. 平铺场景依次抓取除电钻外的全部物体。
4. 平铺场景依次抓取两个螺丝刀、两个锤子和电池。
5. 堆叠场景依次抓取除电钻外的全部物体。

两种场景在每个物体、每次重试前都会重新运行：

```text
虚拟 RGB-D
→ Perception
→ Intent
→ Reason
→ Reason mask 与 PyBullet body 匹配
├─ 核验模式：与 PyBullet 真值比较 → 正确后删除物体
└─ 物理模式：GraspNet → JAKA Zu3 + Robotiq-85 抓取、搬运和投放
```

单 OBJ、夹爪对比和独立 JSON Execution 接口不属于当前测试范围。

## 1. 运行前准备

从本地电脑连接服务器：

```bash
ssh -Y -C admin128@100.115.245.13
```

进入项目并设置 GUI 与 Pipeline 环境：

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/Simulation/SmartGrasp

export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

export PYBULLET_GUI=1
export GRASP_RECORD_VIDEO=1
export GRASP_GUI_SPEED=1.0

echo "$DISPLAY"
```

`DISPLAY` 必须输出非空值，否则 PyBullet GUI 无法显示。

当前服务器如果提示 `sbatch: command not found`，说明没有安装 SLURM 客户端。
此时直接运行项目封装好的 Shell 入口：

```bash
bash run_grasp_simulation.sh [参数...]
```

不要直接裸跑 Python 脚本。只有在已安装并配置 SLURM 的计算节点上，才将下面
命令开头的 `bash` 替换为 `sbatch`。

## 2. 新增：只核验 Perception + Reason，正确后删除物体

新增开关为：

```text
--perception-reason-test
```

该模式仍会为每个物体重新拍摄 RGB-D，并完整运行 Perception、Intent 和
Reason，但不会加载 GraspNet，也不会执行抓取、投放或 Push。机械臂保持初始
拍摄位姿，测试过程中不会发送运动指令。

每个目标使用 PyBullet segmentation 真值做两层核验：

1. `perception_correct`：Perception 至少有一个整物体 mask 能以不低于
   `--target-mask-min-iou` 的 IoU 映射到当前指定物体。
2. `reason_correct`：Reason 的语义目标 `target_object` 和准备抓取的
   `grasp_object` 都映射到当前指定物体。

只有两项都为 `true` 时，程序才将该物体从 PyBullet 场景和活动物体注册表中
删除。任一项错误时物体保持原样，结果中会保留错误详情，并按
`--max-task-rounds` 和 `--max-stalled-passes` 继续重试。

### 2.1 平铺场景：依次核验并删除全部十个物体

```bash
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取{target}" \
  --run-pipeline-after-capture \
  --perception-reason-test \
  --continuous-grasp \
  --max-task-rounds 3 \
  --max-stalled-passes 3 \
  --target-mask-min-iou 0.05 \
  --initial-pose-hold-seconds 0 \
  --gui-speed 1.0 \
  --output results/validation_flat_all_objects.json
```

不传 `--target-objects` 时，平铺场景按配置中的物体顺序处理全部 10 个物体，
包括电钻。

### 2.2 堆叠场景：依次核验并删除全部十二个物体

```bash
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取{target}" \
  --run-pipeline-after-capture \
  --perception-reason-test \
  --continuous-grasp \
  --max-task-rounds 3 \
  --max-stalled-passes 3 \
  --target-mask-min-iou 0.05 \
  --reobserve-settle-steps 120 \
  --initial-pose-hold-seconds 0 \
  --gui-speed 1.0 \
  --output results/validation_stacked_all_objects.json
```

堆叠场景使用 `industrial_scene_stacked.json` 中的
`continuous_grasp.target_order`，先删除上层或遮挡物，再重新拍摄并核验对应的
下层物体。电钻及其下方的电池都包含在 12 个目标中。该模式不会用 Push
主动改变堆叠关系，场景变化只来自“核验通过后删除当前目标”。

### 2.3 快速无窗口版本（推荐先运行）

如果只想尽快查看 Perception + Reason 的成功率，不需要观察 PyBullet 窗口，
可以在命令前强制设置：

```bash
PYBULLET_GUI=0 GRASP_RECORD_VIDEO=0
```

这会使用 PyBullet DIRECT 后端，并关闭 MP4 录制。核验模式没有机械臂动作，
因此可同时使用 `--reobserve-settle-steps 0`。VLM API 调用仍然需要时间，但不会
再产生 GUI 渲染和录屏开销。

平铺场景快速无窗口命令：

```bash
PYBULLET_GUI=0 GRASP_RECORD_VIDEO=0 bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取{target}" \
  --run-pipeline-after-capture \
  --perception-reason-test \
  --continuous-grasp \
  --max-task-rounds 1 \
  --max-stalled-passes 1 \
  --target-mask-min-iou 0.05 \
  --reobserve-settle-steps 0 \
  --initial-pose-hold-seconds 0 \
  --output results/validation_flat_all_objects_fast.json
```

堆叠场景快速无窗口命令：

```bash
PYBULLET_GUI=0 GRASP_RECORD_VIDEO=0 bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取{target}" \
  --run-pipeline-after-capture \
  --perception-reason-test \
  --continuous-grasp \
  --max-task-rounds 1 \
  --max-stalled-passes 1 \
  --target-mask-min-iou 0.05 \
  --reobserve-settle-steps 0 \
  --initial-pose-hold-seconds 0 \
  --output results/validation_stacked_all_objects_fast.json
```

快速版每个目标单次进入 `_attempt_pipeline_target` 时只推理一轮，并在整轮没有
任何成功物体后立即停止；适合先看结果。需要允许模型重试时，再使用前面的
`--max-task-rounds 3 --max-stalled-passes 3` 完整命令。

核验模式输出：

```text
graspnet-workspace/results/validation_flat_all_objects.json
graspnet-workspace/results/validation_flat_all_objects_viz_data.pkl
graspnet-workspace/results/validation_stacked_all_objects.json
graspnet-workspace/results/validation_stacked_all_objects_viz_data.pkl
graspnet-workspace/results/validation_flat_all_objects_fast.json
graspnet-workspace/results/validation_stacked_all_objects_fast.json
```

最终 JSON 的 `mode` 为 `perception_reason_validation_and_delete`。重点查看
`experiment_summary`、`objects`、`attempts`，以及每次尝试中的
`perception_correct`、`reason_correct`、`removed_from_scene` 和
`pipeline_rounds[].action_result`。完整通过时，平铺应为 `10/10`，堆叠应为
`12/12`，并且 `final_scene_objects` 为空。

每次实验结束时，日志会直接打印：

```text
实验结果汇总:
  成功率: 80.00% (8/10)
  成功: 8 个
  失败: 2 个
  成功物体: ...
  失败物体: ...
```

JSON 中同时保存 `object_total`、`object_success`、`object_failed`、
`success_rate`，以及：

```json
{
  "experiment_summary": {
    "total": 10,
    "success": 8,
    "failed": 2,
    "success_rate": 0.8,
    "success_rate_percent": 80.0,
    "successful_objects": ["..."],
    "failed_objects": ["..."]
  }
}
```

## 3. 平铺场景：依次抓取除电钻外的九个物体

完整命令：

```bash
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取{target}" \
  --run-pipeline-after-capture \
  --use-reason-part-mask \
  --continuous-grasp \
  --target-objects \
    phillips_screwdriver \
    flat_screwdriver \
    adjustable_wrench \
    ycb_hammer \
    medium_clamp \
    large_clamp \
    two_color_hammer \
    small_clamp \
    battery \
  --drop-after-grasp \
  --max-task-rounds 3 \
  --max-stalled-passes 3 \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --gui-speed 1.0 \
  --max-candidates-per-object 30 \
  --output results/pipeline_flat_all_except_drill.json
```

目标总数为 `9`。`power_drill` 没有出现在 `--target-objects` 中，因此不会被
抓取。

每个目标都会把 `{target}` 替换为场景配置中的自然语言名称。例如：

```text
抓取十字螺丝刀
抓取一字螺丝刀
抓取活动扳手
...
抓取电池
```

## 4. 平铺场景：依次抓取两个螺丝刀、两个锤子和电池

完整命令：

```bash
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取{target}" \
  --run-pipeline-after-capture \
  --use-reason-part-mask \
  --continuous-grasp \
  --target-objects \
    phillips_screwdriver \
    flat_screwdriver \
    ycb_hammer \
    two_color_hammer \
    battery \
  --drop-after-grasp \
  --max-task-rounds 3 \
  --max-stalled-passes 3 \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --gui-speed 1.0 \
  --max-candidates-per-object 30 \
  --output results/pipeline_flat_screwdrivers_hammers_battery.json
```

目标数为 `5`，顺序是：

```text
十字螺丝刀
→ 一字螺丝刀
→ 银色羊角锤
→ 双色锤
→ 电池
```

每个目标以及每次失败重试都会重新拍摄，并重新运行 Perception、Intent 和
Reason。`--use-reason-part-mask` 会优先把 Reason 选择的把手等可抓部件传给
GraspNet。

## 5. 堆叠场景：依次抓取除电钻外的十一个物体

完整命令：

```bash
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取{target}" \
  --run-pipeline-after-capture \
  --use-reason-part-mask \
  --continuous-grasp \
  --target-objects \
    adjustable_wrench_cover_b \
    flat_screwdriver_cover_c \
    two_color_hammer_cover_d \
    battery_cover_e \
    medium_clamp_cover_f \
    flat_screwdriver_partially_occluded_b \
    large_clamp_base_c \
    phillips_screwdriver_base_d \
    medium_clamp_base_e \
    small_clamp_base_f \
    battery_fully_occluded_a \
  --drop-after-grasp \
  --occlusion-action push \
  --push-distance 0.05 \
  --max-task-rounds 6 \
  --reobserve-settle-steps 120 \
  --max-stalled-passes 3 \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --gui-speed 1.0 \
  --max-candidates-per-object 30 \
  --output results/pipeline_stacked_all_except_drill.json
```

目标总数为 `11`。顺序先处理五个上层物体，再处理对应下层物体，最后处理
电钻下面的电池。

`power_drill_cover_a` 没有出现在 `--target-objects` 中，因此不会作为抓取目标。
处理最后一个电池时，如果 Perception/Reason 仍判断电池不可抓，程序只允许对
电钻执行安全 Push，然后重新拍摄并重新运行 Perception、Intent 和 Reason；不会
抓走电钻。

如果某个上层遮挡物本身仍是尚未完成的批量目标，程序会先等待它按目标顺序被
抓取，不会把它提前当作 Push 对象。

## 6. 如何确认确实运行了 Perception 和 Reason

每个目标都应出现类似日志：

```text
Pipeline 指令: 抓取...
💾 Round 输入: scene_id=...
⏳ 开始运行 Perception + Intent + Reason
✅ Round 推理: branch=..., target=..., grasp_object=...
🔗 Reason Object ... -> body_id=...
🧭 批量 Pipeline 动作: ...
```

每轮数据分别保存在：

```text
input/scene_<id>/
data/scene_<id>/perception/
data/scene_<id>/intent/
data/scene_<id>/reason/
```

## 7. 输出文件

平铺场景：

```text
graspnet-workspace/results/pipeline_flat_all_except_drill.json
graspnet-workspace/results/pipeline_flat_all_except_drill_viz_data.pkl
graspnet-workspace/results/pipeline_flat_all_except_drill_pybullet.mp4
```

平铺五物体场景：

```text
graspnet-workspace/results/pipeline_flat_screwdrivers_hammers_battery.json
graspnet-workspace/results/pipeline_flat_screwdrivers_hammers_battery_viz_data.pkl
graspnet-workspace/results/pipeline_flat_screwdrivers_hammers_battery_pybullet.mp4
```

堆叠场景：

```text
graspnet-workspace/results/pipeline_stacked_all_except_drill.json
graspnet-workspace/results/pipeline_stacked_all_except_drill_viz_data.pkl
graspnet-workspace/results/pipeline_stacked_all_except_drill_pybullet.mp4
```

同名命令再次运行会覆盖对应 JSON、PKL 和 MP4，Perception/Reason 的
`scene_<id>` 数据不会覆盖，而是继续使用新的场景编号。

## 8. 当前验证状态

以上命令目前只完成代码和静态检查。按照项目测试约定，尚未运行耗时 GUI/VLM
仿真，因此不能提前声明核验清场或物理抓取全部成功。实际测试结果应以最终
JSON 中的 `object_success`、`object_total`、`objects` 和 `attempts` 字段为准。
