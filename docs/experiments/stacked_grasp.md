# 堆叠遮挡与连续抓取实验命令

本文档中的命令均在仓库根目录
`/home/admin128/qiuguanhe/Simulation/SmartGrasp` 执行。所有模式共用同一个入口
`run_grasp_simulation.sh`，没有直接运行 Python 脚本。

新增场景配置为
`graspnet-workspace/config/industrial_scene_stacked.json`。它包含 12 个物体、
6 组双层堆叠，并同时提供初始完全遮挡和部分遮挡目标。原来的
`graspnet-workspace/config/industrial_scene.json` 未修改。

## 运行前准备

```bash
ssh -Y -C admin128@100.115.245.13
```

```bash
conda activate smartgrasp
cd ~/qiuguanhe/Simulation/SmartGrasp
```

每个实验启动前设置相同的模型环境变量：

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability
```

## 遮挡动作与最终目标闭环

需要“移开遮挡物后继续观察，直到最终目标被抓到”时，必须添加：

```bash
--task-closed-loop
```

闭环每轮都会重新拍摄，并重新运行 Perception、Intent 和 Reason。中间遮挡物
操作成功只记录为 `rounds[*].action_success=true`；只有最终指令目标物体完成
物理抓取和搬运后，顶层 `task_success` 才为 `true`。

动作对象来源由 `--task-selection-policy` 控制：

- `configured`（默认）：保留场景 `occludes` 关系对 Reason 结果的稳定演示兜底。
- `reason`：动作对象严格来自 Reason 的 `grasp_object`，有效遮挡分支严格来自
  Reason 的 `branch`。场景配置仅用于判断最终真实目标是否抓取成功，不能覆盖
  Reason 动作选择。Reason 没有返回有效对象或 mask 无法映射时，本轮失败，不会
  偷偷回退到配置遮挡物。

纯 Reason 多轮实验需要显式添加：

```bash
--task-selection-policy reason
```

`--task-selection-policy` 只作用于 `--task-closed-loop` 物理闭环，不影响
`--perception-reason-test` 的 no-grasp/no-push 核验删除流程。

Case 28（重度遮挡中号夹具）默认使用 no-grasp/no-push 多轮核验删除模式：

```bash
bash graspnet-workspace/scripts/run_case28_multiround.sh
```

脚本按实际堆叠顺序核验十字螺丝刀、大号夹具、灰色锤子和最终中号夹具。
每个当前目标只有在 Perception 和 Reason 都映射正确后才直接从 PyBullet
场景消失，然后重新拍摄下一层；全程不加载 GraspNet，也不执行机械臂或 Push。

也可以把第一个参数指定为结果 JSON 路径，并通过环境变量调整每个目标的
最大核验轮数：

```bash
MAX_TASK_ROUNDS=5 MAX_STALLED_PASSES=3 \
  bash graspnet-workspace/scripts/run_case28_multiround.sh \
  results/case28_no_grasp_delete.json
```

仿真场景还会使用对象 metadata 中的 `instruction_aliases` 锁定稳定的最终
PyBullet body，并使用遮挡物的 `occludes` 关系防止 Perception 将上下两个工具
合成一个 mask 后误把遮挡物当成最终目标。新增自定义目标指令时，应给对应对象
补充能在指令中匹配到的 `instruction_aliases`。

遮挡物处理由下面的参数选择：

- `--occlusion-action auto`：推荐。部分遮挡先向工作区外推动 5 cm；完全遮挡
  先抓走遮挡物。部分遮挡推动失败时，下一轮自动改为抓走。
- `--occlusion-action push`：部分遮挡和完全遮挡都只推动遮挡物。完全遮挡时
  可能扰动看不见的目标，因此主要用于对比实验。
- `--occlusion-action grasp-away`：部分遮挡和完全遮挡都抓走遮挡物，搬运到
  投放位姿后松爪，再重新观察。

可以用 `--push-distance 0.03` 调整推动距离。默认推动方向根据遮挡物相对工作区
中心的位置自动向外选择；也可以通过下面的形式指定世界坐标方向：

```bash
--push-direction 1 0 0
```

`--max-task-rounds 6` 控制最多运行多少轮感知—动作循环。

每一轮运行 Perception、Intent 和 Reason 时，机械臂会停在固定拍摄位姿。终端
会在 Pipeline 开始时提示这一状态，并每 30 秒输出一次累计等待时间；这段静止
不是机械臂卡死。Pipeline 完成后会显示本轮总推理耗时，随后才生成抓取候选并
执行物理动作。

GUI 模式下，推动动作和抓取动作都会遵守 `GRASP_GUI_SPEED`。例如
`GRASP_GUI_SPEED=0.5` 表示以基准速度的一半显示动作，便于观察活动扳手的推动
轨迹。

闭环每完成一轮都会立即把检查点写入 `--output` 指定的 JSON。任务尚未结束时
文件包含 `"in_progress": true`，并通过 `last_completed_round` 和 `rounds`
保留最近完成轮次的动作及失败原因。正常结束后同一文件会被最终结果覆盖，
`"in_progress"` 变为 `false`。因此后续 Pipeline 被中断时，已经完成的轮次
不会丢失。

## 1. 原平铺场景单次抓取（兼容性基线）

下面是原命令，参数和行为保持不变。它用于确认新增模式没有改变原平铺场景的
单次 Pipeline 抓取。

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取红黑手柄的一字螺丝刀" \
  --run-pipeline-after-capture \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --output results/integration_single_grasp_gui.json
```

## 2. 堆叠场景：部分遮挡目标闭环 + Reason part mask

该命令让 Perception、Intent、Reason 处理堆叠场景，并用
`--use-reason-part-mask` 将 GraspNet 裁剪、候选过滤和物理评估区域限制到
Reason 选中的可抓部件。目标是一字螺丝刀，其上方有偏置的活动扳手。

`auto` 策略会先尝试把活动扳手向工作区外推动 5 cm，然后重新拍照；如果螺丝刀
已经完全露出，下一轮用它的 Reason part mask 抓取螺丝刀。推动失败时，下一轮
会改为抓走活动扳手。抓到活动扳手不算最终任务成功。

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取被活动扳手部分遮挡的一字螺丝刀" \
  --run-pipeline-after-capture \
  --task-closed-loop \
  --occlusion-action auto \
  --max-task-rounds 6 \
  --use-reason-part-mask \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --output results/stacked_partial_occlusion_part_mask.json
```

输出 JSON 中重点检查：

- 顶层 `task_success` 只有抓到一字螺丝刀后才为 `true`。
- `rounds` 记录每轮的分支、遮挡动作、实际操作物体和任务是否完成。
- 抓取轮的 `rounds[*].action_result.grasp_region.source` 应为
  `reason_part_mask`。
- `rounds[*].action_result.grasp_region.part_mask.intersection_pixels` 应大于 0。
- `target_selection.source` 为配置覆盖时，原始 mask 映射位于
  `rounds[*].target_selection.reason_selection`；其中的 `selected_iou` 应不低于
  `0.05`。
- `rounds[*].action_result.grasps[*].translation` 是最终在 PyBullet 世界坐标系
  执行的抓取中心。

### 2.1 部分遮挡：强制只推动遮挡物

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取被活动扳手部分遮挡的一字螺丝刀" \
  --run-pipeline-after-capture \
  --task-closed-loop \
  --occlusion-action push \
  --push-distance 0.05 \
  --max-task-rounds 6 \
  --use-reason-part-mask \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --output results/stacked_partial_occlusion_push_closed_loop.json
```

### 2.2 部分遮挡：强制抓走遮挡物

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取被活动扳手部分遮挡的一字螺丝刀" \
  --run-pipeline-after-capture \
  --task-closed-loop \
  --occlusion-action grasp-away \
  --max-task-rounds 6 \
  --use-reason-part-mask \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --output results/stacked_partial_occlusion_grasp_away_closed_loop.json
```

## 3. 堆叠场景：完全遮挡目标闭环

场景 A 的电池初始位于电钻正下方。完全不可见的电池没有可用 part mask，因此
该命令不加 `--use-reason-part-mask`。`auto` 在完全遮挡分支会先抓走电钻并松爪，
重新拍摄后再定位和抓取露出的电池。只有电池抓取成功才算任务完成。

仿真相机的深度单位为米，Pipeline 会在这个入口中使用 `0.01 m` 的遮挡前后景
阈值；FreeGrasp 数据处理的默认 `0.5` 不变。如需调参，可在命令前额外设置
`export SIMULATION_DEPTH_GAP_THRESHOLD=0.01`。

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取电钻下面完全被遮挡的电池" \
  --run-pipeline-after-capture \
  --task-closed-loop \
  --occlusion-action auto \
  --max-task-rounds 6 \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --output results/stacked_full_occlusion_closed_loop.json
```

### 3.1 完全遮挡：强制推动遮挡物（对比实验）

该策略会推动电钻而不是抓走。由于电池完全不可见，推动可能连带扰动电池，建议
只用于和 `auto`、`grasp-away` 做对比。

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取电钻旁边的电池" \
  --run-pipeline-after-capture \
  --task-closed-loop \
  --occlusion-action push \
  --push-distance 0.05 \
  --max-task-rounds 6 \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --output results/stacked_full_occlusion_push_closed_loop.json
```

### 3.2 完全遮挡：强制抓走遮挡物

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取电钻下面完全被遮挡的电池" \
  --run-pipeline-after-capture \
  --task-closed-loop \
  --occlusion-action grasp-away \
  --max-task-rounds 6 \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --output results/stacked_full_occlusion_grasp_away_closed_loop.json
```

## 4. 堆叠场景：不调用 VLM 的物理抓取核对

该模式按场景对象名直接选择部分遮挡的一字螺丝刀，适合单独核对堆叠、碰撞过滤和
PyBullet 抓取执行，不受 Perception/Reason 选择误差影响。

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --target-object flat_screwdriver_partially_occluded_b \
  --stop-on-success \
  --assisted-grasp \
  --output results/stacked_partial_occlusion_physics_only.json
```

## 5. 堆叠场景：连续抓取、投放、松爪

`--continuous-grasp` 是独立于单次 Pipeline 和旧 `--all-objects` 的清场模式。
每次成功后会：

1. 将物体搬运到 `place_target_joint_pose_deg`；
2. 解除辅助抓取约束并张开夹爪；
3. 等待物体自然落下；
4. 重新拍摄当前场景；
5. 继续处理剩余物体，并重试之前完全不可见或没有候选的物体。

新场景在 `continuous_grasp.target_order` 中按每组堆叠从上到下给出推荐顺序。
初始拍照时物体暂时锁定在配置位姿；某物体真正进入抓取评估前会恢复其质量和
动力学，因此抓取、抬升、运输和投放仍由 PyBullet 物理执行。
连续两轮没有任何成功后停止，未清除的物体会保留在结果 JSON 中，避免死循环。

### 5.1 所有物品依次抓取：SLURM 提交命令

下面的命令按照 `continuous_grasp.target_order` 依次处理所有物品；成功抓取并投放
一个物品后重新拍摄，再继续处理剩余物品。按项目运行约定，通过 SLURM 提交：

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=0 \
GRASP_RECORD_VIDEO=0 \
GRASP_GUI_SPEED=1.0 \
sbatch run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --continuous-grasp \
  --stop-on-success \
  --assisted-grasp \
  --max-candidates-per-object 30 \
  --max-stalled-passes 3 \
  --output results/stacked_all_objects_continuous_grasp.json
```

这条命令只能在已经安装并配置好 SLURM 的提交节点执行。当前 `labserver0` 没有
`sbatch` 或 `srun`，不能在该机器上使用这条命令。不要只根据终端提示安装
`slurm-client`；如果没有集群控制器地址和 SLURM 配置，安装客户端后仍然无法提交。

注意：当前版本还不能通过命令行实现以下策略：

- 同一个物品连续 3 次物理抓取失败后，自动执行 reveal push；
- 一直找不到合适抓取位姿（`no_filtered_candidates`）时，自动执行 reveal push。

`--max-stalled-passes 3` 只表示所有剩余物品连续 3 个完整 pass 都没有任何抓取
成功后终止任务，不会触发推动。当前 `demo_all_objects.py` 的连续清场流程没有调用
`RevealPushExecutor`，命令行也没有“单物体失败次数达到阈值后 push”的参数。因此，
如果实验必须包含上述两条 reveal 规则，需要先为连续清场模式实现该状态机和对应的
命令行参数；在实现之前不存在能够完整满足该实验定义的有效命令。

### 5.2 labserver0 本地运行（仅在允许绕过 SLURM 时）

如果确认本次允许不通过 SLURM、直接占用 `labserver0` 的 GPU，可以将 `sbatch`
替换为 `bash`。这仍然通过项目 Shell 入口运行，不是裸跑 Python：

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=0 \
GRASP_RECORD_VIDEO=0 \
GRASP_GUI_SPEED=1.0 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --continuous-grasp \
  --stop-on-success \
  --assisted-grasp \
  --max-candidates-per-object 30 \
  --max-stalled-passes 3 \
  --output results/stacked_all_objects_continuous_grasp.json
```

### 5.3 紧密堆叠场景：GUI 连续抓取动画

紧密布局配置 `industrial_scene_compact_stacked_layout.json` 继承原 stacked 场景的
物体、抓取顺序、拍摄/投放关节位姿和连续抓取参数，只覆盖相机与 12 个物体的紧密
多层位姿。原 `industrial_scene_stacked.json` 不变。

先确认当前 SSH 会话启用了 X11 转发；`echo "$DISPLAY"` 必须输出非空值。然后运行：

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=1 \
GRASP_RECORD_VIDEO=1 \
GRASP_GUI_SPEED=0.5 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_compact_stacked_layout.json \
  --continuous-grasp \
  --stop-on-success \
  --assisted-grasp \
  --max-candidates-per-object 30 \
  --max-stalled-passes 3 \
  --output results/compact_stacked_continuous_grasp_gui.json
```

运行时会弹出 PyBullet 窗口；同时录制
`graspnet-workspace/results/compact_stacked_continuous_grasp_gui_pybullet.mp4`。
逐次 RGB-D、实例分割和 mask 位于
`graspnet-workspace/results/compact_stacked_continuous_grasp_gui_captures/`。

该紧密场景为了复现高遮挡，初始拍摄时会锁定配置位姿；目标进入抓取评估前恢复
动态质量。由于布局比普通 stacked 场景更密集，物体恢复动力学时可能发生明显接触
或弹开，这正是该压力测试需要观察的现象。

连续模式输出 JSON 中：

- `mode` 为 `continuous_grasp_and_drop`。
- `objects` 是每个物体的最终清除状态。
- `attempts` 保留每轮尝试、失败原因和候选执行记录。
- `capture_root` 是逐次相机拍摄结果目录；每一次抓取尝试都有独立的
  `capture_NNNN_<object>` 子目录。
- 成功候选的 `placement.released_after_place` 为 `true`。
- `placement.post_release_object_position` 是松爪并等待下落后的物体位置。

每个 `capture_NNNN_<object>` 目录包含同一时刻的完整相机输出：

- `rgb.png`：虚拟相机 RGB 图像；
- `depth_m.npy`：原始 `float32` 米制深度；
- `depth_mm.png`：可被图像工具读取的 16-bit 毫米深度图；
- `depth_color.png`：方便肉眼查看的伪彩色深度图；
- `segmentation.npy`：PyBullet 原始分割标签；
- `body_ids_plus_one.png` 和 `segmentation_color.png`：实例标签及其彩色预览；
- `target_mask.png`：本次待抓目标的二值 mask；
- `masks/*.png`：场景中每个配置物体的独立二值 mask；
- `capture.json`：深度范围、mask 像素数和上述文件路径。

这些 mask 是与 RGB-D 严格同帧的 PyBullet 实例真值 mask。`--continuous-grasp`
当前不运行 Perception，因此它们不是 SAM2/Perception 的预测 mask。

旧结果如果只有 `*_viz_data.pkl`，可使用下面的离线工具补导出同帧完整相机文件：

```bash
bash export_simulation_capture.sh \
  --viz-data graspnet-workspace/results/integration_single_grasp_gui_viz_data.pkl \
  --results graspnet-workspace/results/integration_single_grasp_gui.json \
  --output-dir graspnet-workspace/results/camera_preview
```

需要覆盖场景默认等待参数时，可添加：

```bash
  --drop-settle-steps 240 \
  --max-stalled-passes 3
```

## 6. SLURM 无界面运行

集群批量实验关闭 GUI 和原生 GUI 录屏，其他参数与第 5 节一致：

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=0 \
GRASP_RECORD_VIDEO=0 \
GRASP_GUI_SPEED=1.0 \
sbatch run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --continuous-grasp \
  --stop-on-success \
  --assisted-grasp \
  --max-candidates-per-object 30 \
  --output results/stacked_continuous_grasp_drop_slurm.json
```

任务完成后查看：

```bash
tail -100 logs/grasp-sim.err
tail -100 logs/grasp-sim.out
```

### 6.1 SLURM 最终目标闭环

下面以“完全遮挡电池、自动选择遮挡动作”为例。它与第 3 节逻辑相同，只是关闭
GUI 并通过 SLURM 提交：

```bash
export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

PYBULLET_GUI=0 \
GRASP_RECORD_VIDEO=0 \
GRASP_GUI_SPEED=1.0 \
sbatch run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
  --instruction "抓取电钻下面完全被遮挡的电池" \
  --run-pipeline-after-capture \
  --task-closed-loop \
  --occlusion-action auto \
  --max-task-rounds 6 \
  --target-mask-min-iou 0.05 \
  --stop-on-success \
  --assisted-grasp \
  --output results/stacked_full_occlusion_closed_loop_slurm.json
```
