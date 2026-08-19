# SmartGrasp 2×3 多轮测试命令

本文档按 FreeGrasp 的分组方式，给出 SmartGrasp 仿真实验的 6 组命令：

| 提示词 / 场景 | 低复杂度 | 中复杂度 | 高复杂度 |
|---|---|---|---|
| w/o ambiguity（无歧义） | 平铺、目标类别唯一 | 堆叠场景中可见的上层唯一类别 | 唯一类别的上下层揭露序列 |
| with ambiguity（有歧义） | 平铺场景中的多个同类实例 | 堆叠场景中的同类上下层实例 | 完全遮挡且存在同类干扰物的揭露序列 |

这里的“有歧义”与 FreeGraspData 一致：场景中有多个同类物体，
指令需要借助大小、外观、上下层或遮挡关系消歧；并不是故意给出
无法确定唯一目标的指令。

## 1. 统一实验策略

六组命令全部使用 `--perception-reason-test`，流程固定为：

```text
虚拟 RGB-D 拍摄
→ Perception
→ Intent
→ Reason
→ PyBullet 真值核验
→ Perception 和 Reason 都正确时直接删除物体
```

所有命令都不传 `--ckpt`、`--use-reason-part-mask`、`--assisted-grasp`
或任何 GraspNet 参数。运行时 `network=None`，不会加载 GraspNet，
不会让机械臂执行抓取、搬运或 Push。

每个独立场景使用 3 种等价提示词各跑 1 次，对应 FreeGraspData 中
同一场景的 3 条标注。为了尽量快且避免重试抬高成功率，每个目标
只运行 1 轮 Perception–Reason。全部六组共包含 42 个独立 episode。

## 2. 运行前准备

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/Simulation/SmartGrasp

proxy_status
# 如果上一条不是代理模式，先执行：
proxy_on
proxy_status

export OPENAI_BASE_URL=https://yunwu.ai/v1
export REVIEW_MODEL_ID=gpt-5.5
export REASON_MODEL=gpt-5.5
export REASON_PRIOR_PROMPT=graspability
export REASON_RANKING_SCORE=ig_graspability

export PYBULLET_GUI=0
export GRASP_RECORD_VIDEO=0
export GRASP_GUI_SPEED=1.0

mkdir -p logs graspnet-workspace/results/freegrasp_style

PROMPTS=(
  "抓取{target}"
  "请找到{target}并将它取出"
  "目标物体是{target}，请将它抓取出来"
)
```

六组命令统一使用 PyBullet DIRECT 后端，不创建 GUI 窗口，也不录制视频。
运行过程只在 SLURM 日志中输出进度，并将成功率和逐物体结果写入 JSON。

### 2.1 推荐：一键串行六类实验

下面的命令只申请一次 SLURM 资源，然后在同一作业中依次跑完六类
实验。任一 episode 失败不会阻止后续 episode，最后会自动生成总报告。

```bash
mkdir -p logs
sbatch run_freegrasp_style_benchmark.sh
```

查看作业：

```bash
squeue -u "$USER"
```

作业完成后的统一报告为：

```text
FREEGRASP_STYLE_RESULTS.md
```

每次运行的原始 JSON 都保存在独立目录：

```text
graspnet-workspace/results/freegrasp_style/job_<SLURM_JOB_ID>/
```

默认速度配置为：无 GUI、无录像、无 GraspNet、不写额外可视化 PKL、
`--reobserve-settle-steps 0`、每目标只推理 1 轮。默认 IoU 阈值为 `0.5`，
与 FreeGrasp Table I 中 SSR 的定义对齐。如确实需要允许每个目标重试，
可在提交前设置：

```bash
export FREEGRASP_MAX_TASK_ROUNDS=3
sbatch run_freegrasp_style_benchmark.sh
```

### 2.2 三种提示词模板

每个 episode 对同一目标使用下面三种等价表达：

1. `抓取{target}`
2. `请找到{target}并将它取出`
3. `目标物体是{target}，请将它抓取出来`

`{target}` 不是原样发给 VLM 的字符，程序会用场景 JSON 中的中文目标
描述替换它。例如第一种模板会展开为 `抓取活动扳手`、
`抓取上层的电池` 或 `抓取画面左下方的电池`。

### 2.3 六类场景和实际目标词

| 条件 | 场景和目标 | `{target}` 实际取值 | 消歧依据 |
|---|---|---|---|
| 低 / w/o | 10 物体平铺场景；活动扳手、电钻、电池各自独立重建场景 | `活动扳手`、`电钻`、`电池` | 三个目标的类别均唯一，无需空间关系 |
| 低 / with | 同一平铺场景同时有三个夹具；每个夹具独立重建场景 | `小号夹具`、`中号夹具`、`大号夹具` | 同类实例通过尺寸消歧 |
| 中 / w/o | 12 物体、6 组两层堆叠；只测可见上层的唯一类别 | `电钻`、`上层的活动扳手`、`上层的双色锤` | 目标类别唯一，但背景有堆叠、接触和更多杂物 |
| 中 / with | 同一堆叠场景；目标类别在其他堆叠中还有同类实例 | `上层的一字螺丝刀`、`上层的电池`、`上层的中号夹具` | 通过“上层”和物体属性消歧 |
| 高 / w/o | 同一 episode 先删除上层双色锤，再重新拍摄下层十字螺丝刀 | 第一步 `上层的双色锤`；揭露后第二步 `画面左上方的十字螺丝刀` | 需要两轮场景更新，但十字螺丝刀细分类别唯一 |
| 高 / with | 同一 episode 先删除电钻，再重新拍摄完全遮挡的电池；场景中仍有另一块电池 | 第一步 `电钻`；揭露后第二步 `画面左下方的电池` | 通过揭露顺序和当前画面位置区分两块电池 |

低、中难度的每个目标都在全新仿真场景中运行，以保证同类干扰物始终
存在。高难度则必须在同一 episode 中保留“删除上层 → 重新拍摄 →
定位下层”的状态变化。

## 3. 低复杂度 + w/o ambiguity

平铺场景、无遮挡。每次作业只处理一个类别唯一的物体：
活动扳手、电钻、电池。

```bash
for TARGET in adjustable_wrench power_drill battery; do
  for PROMPT_INDEX in "${!PROMPTS[@]}"; do
    REPEAT=$((PROMPT_INDEX + 1))
    sbatch --wait \
      --job-name=sg-low-wo \
      --output="logs/sg-low-wo-${TARGET}-r${REPEAT}-%j.out" \
      --error="logs/sg-low-wo-${TARGET}-r${REPEAT}-%j.err" \
      run_grasp_simulation.sh \
      --scene-config graspnet-workspace/config/industrial_scene.json \
      --instruction "${PROMPTS[$PROMPT_INDEX]}" \
      --run-pipeline-after-capture \
      --perception-reason-test \
      --continuous-grasp \
      --skip-viz-data \
      --target-objects "$TARGET" \
      --max-task-rounds 1 \
      --max-stalled-passes 1 \
      --target-mask-min-iou 0.5 \
      --reobserve-settle-steps 0 \
      --initial-pose-hold-seconds 0 \
      --seed "$REPEAT" \
      --output "results/freegrasp_style/low_without_ambiguity_${TARGET}_r${REPEAT}.json"
  done
done
```

## 4. 低复杂度 + with ambiguity

仍为平铺无遮挡场景，但同时存在小号、中号和大号三个夹具。
每个目标在全新场景中独立测试，不会因前一个夹具已删除而降低歧义。

```bash
for TARGET in small_clamp medium_clamp large_clamp; do
  for PROMPT_INDEX in "${!PROMPTS[@]}"; do
    REPEAT=$((PROMPT_INDEX + 1))
    sbatch --wait \
      --job-name=sg-low-w \
      --output="logs/sg-low-w-${TARGET}-r${REPEAT}-%j.out" \
      --error="logs/sg-low-w-${TARGET}-r${REPEAT}-%j.err" \
      run_grasp_simulation.sh \
      --scene-config graspnet-workspace/config/industrial_scene.json \
      --instruction "${PROMPTS[$PROMPT_INDEX]}" \
      --run-pipeline-after-capture \
      --perception-reason-test \
      --continuous-grasp \
      --skip-viz-data \
      --target-objects "$TARGET" \
      --max-task-rounds 1 \
      --max-stalled-passes 1 \
      --target-mask-min-iou 0.5 \
      --reobserve-settle-steps 0 \
      --initial-pose-hold-seconds 0 \
      --seed "$REPEAT" \
      --output "results/freegrasp_style/low_with_ambiguity_${TARGET}_r${REPEAT}.json"
  done
done
```

## 5. 中复杂度 + w/o ambiguity

使用 12 物体堆叠场景，但只测试当前可见的上层物体，并选择
在场景中类别唯一的电钻、活动扳手和双色锤。

```bash
for TARGET in power_drill_cover_a adjustable_wrench_cover_b two_color_hammer_cover_d; do
  for PROMPT_INDEX in "${!PROMPTS[@]}"; do
    REPEAT=$((PROMPT_INDEX + 1))
    sbatch --wait \
      --job-name=sg-medium-wo \
      --output="logs/sg-medium-wo-${TARGET}-r${REPEAT}-%j.out" \
      --error="logs/sg-medium-wo-${TARGET}-r${REPEAT}-%j.err" \
      run_grasp_simulation.sh \
      --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
      --instruction "${PROMPTS[$PROMPT_INDEX]}" \
      --run-pipeline-after-capture \
      --perception-reason-test \
      --continuous-grasp \
      --skip-viz-data \
      --target-objects "$TARGET" \
      --max-task-rounds 1 \
      --max-stalled-passes 1 \
      --target-mask-min-iou 0.5 \
      --reobserve-settle-steps 0 \
      --initial-pose-hold-seconds 0 \
      --seed "$REPEAT" \
      --output "results/freegrasp_style/medium_without_ambiguity_${TARGET}_r${REPEAT}.json"
  done
done
```

## 6. 中复杂度 + with ambiguity

同一堆叠场景中，选择有同类上下层干扰物的上层一字螺丝刀、
电池和中号夹具。指令中的“上层”用于区分同类实例。

```bash
for TARGET in flat_screwdriver_cover_c battery_cover_e medium_clamp_cover_f; do
  for PROMPT_INDEX in "${!PROMPTS[@]}"; do
    REPEAT=$((PROMPT_INDEX + 1))
    sbatch --wait \
      --job-name=sg-medium-w \
      --output="logs/sg-medium-w-${TARGET}-r${REPEAT}-%j.out" \
      --error="logs/sg-medium-w-${TARGET}-r${REPEAT}-%j.err" \
      run_grasp_simulation.sh \
      --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
      --instruction "${PROMPTS[$PROMPT_INDEX]}" \
      --run-pipeline-after-capture \
      --perception-reason-test \
      --continuous-grasp \
      --skip-viz-data \
      --target-objects "$TARGET" \
      --max-task-rounds 1 \
      --max-stalled-passes 1 \
      --target-mask-min-iou 0.5 \
      --reobserve-settle-steps 0 \
      --initial-pose-hold-seconds 0 \
      --seed "$REPEAT" \
      --output "results/freegrasp_style/medium_with_ambiguity_${TARGET}_r${REPEAT}.json"
  done
done
```

## 7. 高复杂度 + w/o ambiguity

每个独立 episode 先核验并删除上层双色锤，然后重新拍摄并核验
下层十字螺丝刀。十字螺丝刀在场景中是唯一细分类别。

```bash
for PROMPT_INDEX in "${!PROMPTS[@]}"; do
  REPEAT=$((PROMPT_INDEX + 1))
  sbatch --wait \
    --job-name=sg-high-wo \
    --output="logs/sg-high-wo-r${REPEAT}-%j.out" \
    --error="logs/sg-high-wo-r${REPEAT}-%j.err" \
    run_grasp_simulation.sh \
    --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
    --instruction "${PROMPTS[$PROMPT_INDEX]}" \
    --run-pipeline-after-capture \
    --perception-reason-test \
    --continuous-grasp \
    --skip-viz-data \
    --target-objects \
      two_color_hammer_cover_d \
      phillips_screwdriver_base_d \
    --max-task-rounds 1 \
    --max-stalled-passes 1 \
    --target-mask-min-iou 0.5 \
    --reobserve-settle-steps 0 \
    --initial-pose-hold-seconds 0 \
    --seed "$REPEAT" \
    --output "results/freegrasp_style/high_without_ambiguity_r${REPEAT}.json"
done
```

## 8. 高复杂度 + with ambiguity

每个独立 episode 先删除电钻，再重新观察完全遮挡的电池。场景的
另一组堆叠中还有第二块电池，因此揭露后使用“画面左下方的电池”
继续消歧。

```bash
for PROMPT_INDEX in "${!PROMPTS[@]}"; do
  REPEAT=$((PROMPT_INDEX + 1))
  sbatch --wait \
    --job-name=sg-high-w \
    --output="logs/sg-high-w-r${REPEAT}-%j.out" \
    --error="logs/sg-high-w-r${REPEAT}-%j.err" \
    run_grasp_simulation.sh \
    --scene-config graspnet-workspace/config/industrial_scene_stacked.json \
    --instruction "${PROMPTS[$PROMPT_INDEX]}" \
    --run-pipeline-after-capture \
    --perception-reason-test \
    --continuous-grasp \
    --skip-viz-data \
    --target-objects \
      power_drill_cover_a \
      battery_fully_occluded_a \
    --max-task-rounds 1 \
    --max-stalled-passes 1 \
    --target-mask-min-iou 0.5 \
    --reobserve-settle-steps 0 \
    --initial-pose-hold-seconds 0 \
    --seed "$REPEAT" \
    --output "results/freegrasp_style/high_with_ambiguity_r${REPEAT}.json"
done
```

## 9. 结果检查

一键脚本的全部进度位于同一组 SLURM 日志：

```bash
tail -50 logs/freegrasp-style-<jobid>.out
tail -50 logs/freegrasp-style-<jobid>.err
```

六种情况的汇总表、SSR、RSR、删除成功率、逐 episode 提示词和失败信息
会自动写入：

```text
FREEGRASP_STYLE_RESULTS.md
```

核心字段为：

- `mode` 必须是 `perception_reason_validation_and_delete`。
- `objects[*].perception_correct` 用于统计 SSR 风格的分割成功率。
- `objects[*].reason_correct` 用于统计 RSR 风格的推理成功率。
- `objects[*].removed_from_scene` 只在两项都正确时为 `true`。
- 高复杂度 episode 要求 `2/2` 成功：遮挡物和下层目标都必须核验通过。

这套测试没有机械臂路径或物理抓取，因此可以对齐 Table I 中的
SSR/RSR 思路，但不能用来计算 Table II 的 SR、PE 或 SPL。

## 10. 注意事项

- 推荐只提交一次 `run_freegrasp_style_benchmark.sh`，脚本内部会串行六类实验。
- 手动命令只用于单独重跑某类实验，不要和一键作业并行。
- 同一输出名会被 `run_grasp_simulation.sh` 清理后重写，因此不要删除
  文档中的目标名和 `r1/r2/r3` 后缀。
- 本文档定义的“中/高难度”是 SmartGrasp 仿真协议：中难度侧重
  堆叠杂乱中的可见目标，高难度侧重删除遮挡物后重新观察。
  它与 FreeGraspData 原数据集的 easy/medium/hard 划分类似，但不是完全相同的场景标注。
