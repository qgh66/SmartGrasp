# SmartGrasp 单轮 Perception → Reason → Execution 使用说明

本文档对应仓库：

```text
/home/admin128/hanhuang/temp/SmartGrasp
```

当前实现的是单轮完整链路：

```text
PyBullet 相机拍摄一帧
→ 保存同帧 RGB、Depth、Segmentation 和用户指令
→ Perception
→ Intent
→ Reason
→ 读取 Reason 的 grasp_object.id
→ 找到该 Object ID 的整物体 mask
→ 与本帧 PyBullet segmentation 做最大 IoU 映射
→ 得到 PyBullet body ID
→ GraspNet 生成该物体的抓取候选
→ PyBullet 执行一次夹取
→ 保存结果并结束
```

## 1. 运行前准备

进入隔离仓库：

```bash
cd /home/admin128/hanhuang/temp/SmartGrasp
```

确认使用 `smartgrasp` 环境：

```bash
conda activate smartgrasp
```

设置 API：

```bash
export OPENAI_API_KEY="你的 API Key"
```

如果需要覆盖仓库中的默认 API 地址：

```bash
export OPENAI_BASE_URL="https://yunwu.ai/v1"
```

`run_grasp_simulation.sh` 会检查代理状态，并在服务器提供
`proxy_on` 时尝试开启代理。运行前仍建议手动确认：

```bash
proxy_status
```

同时需要确保：

- `graspnet-workspace/checkpoints/checkpoint-rs.tar` 存在，或者通过
  `--ckpt` 指定 checkpoint。
- 场景 JSON 中的物体模型路径有效。
- GPU、PyBullet、SAM2 和 GraspNet 依赖已经安装。

## 2. 单轮完整链路

```bash
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取红色螺丝刀" \
  --run-pipeline-after-capture \
  --stop-on-success
```

默认使用整物体点云定位 GraspNet 候选。若要使用 Reason 输出的最佳部件
mask 聚焦抓取区域，增加：

```bash
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取红色螺丝刀" \
  --run-pipeline-after-capture \
  --use-reason-part-mask
```

`--use-reason-part-mask` 仅改变 GraspNet 的抓取区域：

1. 整物体 mask 仍用于将 Reason Object ID 映射到 PyBullet body ID。
2. `part_id` 从 1 开始连续编号，与 SAM2Auto candidate id 分离。
3. part → object 直接沿用 VLM 已经建立的关系；未映射、重复映射或 owner
   未进入最终 graph 的 SAM2 候选不会进入正式 part 列表。
4. 不限制进入 SAM2 label、全尺寸 mask、part sheet 和 VLM 编号的候选数量；
   `visible_parts[].sam2_ids` 必须是所属 object 顶层 `sam2_ids` 的子集。
5. 一个 SAM2 id 对应一个几何 part；同一 id 的多个文字描述不会被人为
   拆成多个 part。
6. Reason 只能为当前 object 的 `object_id_to_part_ids` 列表返回和选择 part。
7. 正式 part mask 保存前与所属 object mask 求交，并保持原图坐标下的
   全尺寸二值格式。
8. part mask 与已选 body 的 segmentation 求交后反投影为部件点云。
9. 部件点云用于裁剪 GraspNet 输入、过滤候选和约束执行高度。
10. part mask 缺失、归属不匹配、为空或与已选 body 无重叠时直接报错，
   不静默回退。

具体映射、mask 路径、源 SAM2 id、覆盖率与拒绝原因保存在 Perception
`summary.json` 的 `part_records`、`object_id_to_part_ids`、
`part_id_to_object_id` 和 `rejected_part_candidates` 中。如果候选 mask 本身
接近整个物体，程序仍会在其覆盖至少 95% 的可见 body 时输出警告。

正常使用时不需要传 `--scene-id`。程序会自动分配：

```text
第一次拍摄：scene_id=1
第二次拍摄：scene_id=2
第三次拍摄：scene_id=3
...
```

当前编号保存在：

```text
input/.capture_scene_id
```

如果对应的 `input/scene_<id>` 已经存在，程序会跳过该编号，避免覆盖。

## 3. 一次运行的输入

相机只调用一次：

```python
rgb, depth, segmentation = camera.capture()
```

三个数组来自同一台 PyBullet 相机、同一场景状态和同一次拍摄：

- RGB：彩色图像，保存时去掉 alpha 通道。
- Depth：`float32` 米制深度。
- Segmentation：`int32` PyBullet body/link 分割值。
- Instruction：`--instruction` 传入的用户自然语言指令。

以 `scene_id=1` 为例，输入写入：

```text
input/scene_1/
├── scene_image.png
├── depth.npy
├── segmentation.npy
├── input.txt
└── summary.json
```

`summary.json` 会记录文件路径、数据形状、数据类型、深度单位，以及
RGB、Depth、Segmentation 来自同一帧的标记。

## 4. Perception、Intent 和 Reason 输出

完整模式内部调用：

```text
run_pipeline_for_scene()
→ run_pipeline.sh <scene_id> --instruction=input
→ perception/run_perception.sh
→ reason.run_reason
```

输出位置：

```text
data/scene_1/
├── perception/
│   ├── summary.json
│   ├── occlusion_graph.json
│   ├── occlusion_graph.png
│   ├── label_2_vlm.png
│   ├── final_objects_sheet.png
│   └── mask/
├── intent/
│   ├── intent_result.json
│   └── id.txt
└── reason/
    ├── summary.json
    ├── results.csv
    └── reason.txt
```

Execution 读取：

```text
data/scene_1/reason/summary.json
```

其中用于执行的主要字段是：

```json
{
  "branch": "fully_visible",
  "grasp_object": {
    "id": 3,
    "label": "red screwdriver"
  }
}
```

程序从 `occlusion_graph.json` 找到 Object 3 的整物体 mask，然后与本次
`segmentation.npy` 中每个可见 PyBullet body 的 mask 计算 IoU，选择
IoU 最大且高于阈值的 body ID。

默认最小 IoU：

```text
target_mask_min_iou = 0.01
```

可以覆盖：

```bash
--target-mask-min-iou 0.05
```

## 5. GraspNet 和执行输出

完整模式会先运行 Perception/SAM2，再加载 GraspNet，避免两个模型同时
占用大量 GPU 显存。

GraspNet 只对 Reason 映射得到的 PyBullet 目标物体生成和过滤抓取候选。
`--stop-on-success` 表示找到第一个物理抓取成功的候选后停止。

默认输出：

```text
graspnet-workspace/results/grasp_simulation.json
graspnet-workspace/results/grasp_simulation_viz_data.pkl
```

`grasp_simulation.json` 会记录：

- Reason 选择的 Object ID、label 和 branch。
- 使用的整物体 mask 和 Reason summary 路径。
- 映射得到的 PyBullet body ID 和物体名称。
- 所有 body 的 IoU 候选及最终 `selected_iou`。
- GraspNet 候选、执行位姿、成功数量和失败原因。
- Perception、Reason 和输入文件目录。

关键结果字段示例：

```json
{
  "reason_target": {
    "scene_id": 1,
    "branch": "fully_visible",
    "object_id": 3,
    "object_label": "red screwdriver",
    "object_mask_path": ".../mask/mask_003_....png"
  },
  "target_body_id": 7,
  "target_selection": {
    "reason_object_id": 3,
    "selected_body_id": 7,
    "selected_object_name": "screwdriver",
    "selected_iou": 0.82
  },
  "success": 1
}
```

## 6. 其他运行模式

只导出同帧输入并运行 Perception，不运行 Intent、Reason 和 Reason 目标映射：

```bash
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取红色螺丝刀" \
  --run-perception-after-capture
```

完整模式和只跑 Perception 模式不能同时使用：

```text
--run-pipeline-after-capture
--run-perception-after-capture
```

完整模式会自动确定目标，因此不能同时传：

```text
--target-object
--target-mask
--all-objects
```

## 7. 修改 Perception 和 Reason 配置

完整链路直接调用 `perception/run_perception.sh`，没有复制第二套配置。
运行 `run_grasp_simulation.sh` 前设置的环境变量会继续传递给
`run_pipeline.sh`、Perception 和 Reason。

### 7.1 Perception 模型与基础配置

```bash
REVIEW_MODEL_ID=gpt-4o \
REVIEW_TIMEOUT=300 \
MODE=vlm \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取红色螺丝刀" \
  --run-pipeline-after-capture \
  --stop-on-success
```

主要变量：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `REVIEW_MODEL_ID` | `gpt-5.5` | Perception VLM review 模型 |
| `REVIEW_TIMEOUT` | `300` | Perception VLM 请求超时秒数 |
| `MODE` | `vlm` | Perception 模式 |
| `OPENAI_API_KEY` | 无 | API Key |
| `OPENAI_BASE_URL` | 仓库脚本默认值 | Perception API 地址 |

### 7.2 Perception SAM2 与后处理参数

默认值：

```text
RGB SAM2
points_per_side = 24
pred_iou_thresh = 0.68
stability_score_thresh = 0.83
crop_n_layers = 0

Depth SAM2
crop_n_layers = 1
pred_iou_thresh = 0.58
stability_score_thresh = 0.73

Post-process
kernel_size = 11
min_contact_pixels = 50
min_contact_ratio = 0.002
mask_clean_kernel = 3
proposal_min_area_ratio = 0.006
proposal_max_area_ratio = 0.11
proposal_border_fraction_threshold = 0.18
```

对应环境变量：

| 环境变量 | 默认值 |
| --- | --- |
| `SAM2_POINTS_PER_SIDE` | `24` |
| `SAM2_PRED_IOU_THRESH` | `0.68` |
| `SAM2_STABILITY_SCORE_THRESH` | `0.83` |
| `SAM2_CROP_N_LAYERS` | `0` |
| `DEPTH_SAM2_POINTS_PER_SIDE` | 未单独指定 |
| `DEPTH_SAM2_PRED_IOU_THRESH` | `0.58` |
| `DEPTH_SAM2_STABILITY_SCORE_THRESH` | `0.73` |
| `DEPTH_SAM2_CROP_N_LAYERS` | `1` |
| `KERNEL_SIZE` | `11` |
| `MIN_CONTACT_PIXELS` | `50` |
| `MIN_CONTACT_RATIO` | `0.002` |
| `MASK_CLEAN_KERNEL` | `3` |
| `PROPOSAL_MIN_AREA_RATIO` | `0.006` |
| `PROPOSAL_MAX_AREA_RATIO` | `0.11` |
| `PROPOSAL_BORDER_FRACTION_THRESHOLD` | `0.18` |

修改示例：

```bash
REVIEW_MODEL_ID=gpt-4o \
SAM2_POINTS_PER_SIDE=32 \
SAM2_PRED_IOU_THRESH=0.72 \
SAM2_STABILITY_SCORE_THRESH=0.86 \
DEPTH_SAM2_CROP_N_LAYERS=1 \
DEPTH_SAM2_PRED_IOU_THRESH=0.60 \
DEPTH_SAM2_STABILITY_SCORE_THRESH=0.75 \
KERNEL_SIZE=11 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取红色螺丝刀" \
  --run-pipeline-after-capture \
  --stop-on-success
```

不设置这些环境变量时，完整链路始终使用
`perception/run_perception.sh` 中的默认值。

### 7.3 Reason 模型、prompt 和 ranking 方法

Reason 的主要变量：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `REASON_MODEL` | `gpt-5.5` | Reason VLM 模型 |
| `REASON_PRIOR_PROMPT` | `graspability` | VLM prior prompt 类型 |
| `REASON_RANKING_SCORE` | `ig_graspability` | 候选物体排序公式 |
| `RUN_INTENT` | `1` | 是否先根据指令运行 Intent |
| `TARGET_ID` | 未设置 | 调试时直接指定 Perception Object ID |
| `PIPELINE_VERBOSE` | `0` | 是否显示更多 Pipeline 日志 |

`REASON_PRIOR_PROMPT` 可选：

```text
original
graspability
```

- `original`：使用原始语义先验 prompt。
- `graspability`：要求 VLM 同时返回 Object-level graspability、
  Part-level graspability 和相应的最佳部件。

注意：当前 `fully_visible` handler 无论 prior prompt 是 `original`
还是 `graspability`，都会为唯一目标额外计算一次 graspability。
因为该分支只有一个可选目标，这个分数会被记录，但不会改变 Object ID。

`REASON_RANKING_SCORE` 可选：

```text
legacy
ig
ig_graspability
theory
```

在 partially occluded 分支中：

```text
legacy          = P_prior × IG
ig              = IG
ig_graspability = IG × Graspability
theory          = Graspability × P_prior × normalized_IG
```

在 fully occluded 分支中：

```text
legacy          = IG
ig              = IG
ig_graspability = IG × Graspability
theory          = normalized_IG × Graspability
```

当前默认组合：

```bash
REASON_PRIOR_PROMPT=graspability
REASON_RANKING_SCORE=ig_graspability
```

即使用 Object-level graspability：

```text
score = IG × Graspability
```

四种常用实验组合：

```text
Information Gain，不带 graspability:
  REASON_PRIOR_PROMPT=original
  REASON_RANKING_SCORE=ig

Information Gain，带 graspability:
  REASON_PRIOR_PROMPT=graspability
  REASON_RANKING_SCORE=ig_graspability

Theory，不带 graspability:
  REASON_PRIOR_PROMPT=original
  REASON_RANKING_SCORE=theory

Theory，带 graspability:
  REASON_PRIOR_PROMPT=graspability
  REASON_RANKING_SCORE=theory
```

在 `original + theory` 中，未请求到的候选物体 graspability 按 `1.0`
处理，因此排序主要由 prior probability 和 normalized IG 决定。

例如 Perception 和 Reason 都使用 GPT-4o，并使用带 graspability 的
Information Gain：

```bash
OPENAI_API_KEY="你的 API Key" \
REVIEW_MODEL_ID=gpt-4o \
REASON_MODEL=gpt-4o \
REASON_PRIOR_PROMPT=graspability \
REASON_RANKING_SCORE=ig_graspability \
PIPELINE_VERBOSE=1 \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取红色螺丝刀" \
  --run-pipeline-after-capture \
  --stop-on-success
```

使用 Theory + graspability：

```bash
REASON_MODEL=gpt-5.5 \
REASON_PRIOR_PROMPT=graspability \
REASON_RANKING_SCORE=theory \
bash run_grasp_simulation.sh \
  --scene-config graspnet-workspace/config/industrial_scene.json \
  --instruction "抓取红色螺丝刀" \
  --run-pipeline-after-capture \
  --stop-on-success
```

### 7.4 Intent 与 API 地址的当前边界

正式的自然语言指令流程应保留：

```bash
RUN_INTENT=1
```

调试时如果已经知道目标 Object ID，可以绕过 Intent：

```bash
TARGET_ID=3 \
bash run_grasp_simulation.sh ...
```

不要在正式单目标执行中随意使用 `RUN_INTENT=0`。该设置会让 Reason
遍历所有物体，不再根据用户指令确定唯一目标。

当前 Intent 模型仍由 `intent/run_intent.py` 中的常量控制：

```text
RUN_INTENT_MODEL = gpt-5.5
```

Reason 的 API 地址由 `reason/vlm/config.py` 中的 `VLM_BASE_URL`
控制，Intent 地址由 `intent/run_intent.py` 中的
`RUN_INTENT_BASE_URL` 控制。因此当前 `OPENAI_BASE_URL` 主要覆盖
Perception，不会自动统一覆盖 Intent 和 Reason。

总结：

- Perception 模型和参数：可以通过环境变量修改。
- Reason 模型、prompt 和 ranking 方法：可以通过环境变量修改。
- Intent 模型：目前需要修改 `intent/run_intent.py`。
- Reason/Intent API 地址：目前需要修改各自配置文件。

## 8. 失败处理

以下情况会停止，不会退回到默认物体继续误抓：

- Perception 或 Reason 命令返回非零状态。
- 缺少 `perception/summary.json`、`occlusion_graph.json` 或
  `reason/summary.json`。
- Reason 没有输出 `grasp_object.id`。
- Reason Object ID 不在遮挡图节点中。
- 对应整物体 mask 不存在或为空。
- mask 与所有 PyBullet body 的最大 IoU 低于阈值。
- 使用 `--use-reason-part-mask` 时，Reason 没有输出有效 part mask、part
  不属于选中的 object，或 part mask 与选中的 PyBullet body 没有重叠。
- GraspNet 没有生成有效候选。

## 9. 当前限制

当前只实现单轮：

```text
拍摄一次 → 推理一次 → 夹取一次 → 结束
```

- 如果 `branch=fully_visible`，本轮通常夹取用户目标。
- 如果 `branch=partially_occluded` 或 `fully_occluded`，本轮夹取 Reason
  选择的遮挡物，然后结束；不会自动重新拍照继续处理原目标。
- 默认仍按整物体点云生成抓取；传入 `--use-reason-part-mask` 后会使用
  Reason 的最佳部件 mask 聚焦 GraspNet，但精度受 Perception 部件 mask
  粒度限制。
- 当前代码已经完成静态检查，但仍需要一次真实的 GPU、API 和 PyBullet
  端到端运行来验证服务器环境。
