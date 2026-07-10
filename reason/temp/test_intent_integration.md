# test.py Intent Integration Notes

这个说明对应当前 `test.py` 的整合逻辑：把 `run_intent` 风格的目标物体
解析，接到后面的 `classify_branch` 和三种 handler 分析流程里。

## 目标

现在 `test.py` 有两种 target 来源：

- 直接给定目标物体 id。
- 通过 intent VLM 根据自然语言指令和 `perception/summary.json` 解析目标物体 id。

拿到目标 id 之后，后面的流程是同一套：

```text
target id
  -> classify_branch
  -> fully_visible / partially_visible / fully_occluded handler
  -> 输出 results.csv、branch_results.json、scene_details/scene_<id>.csv
```

也就是说，`run_intent` 只负责选目标物体 id，不负责判断
`fully_visible`、`partially_visible` 或 `fully_occluded`。

## 方式 1：直接给定目标 id

适合做固定目标的对比实验，不需要 intent VLM 先判断目标。

```bash
source /home/qiuguanhe/miniconda3/etc/profile.d/conda.sh
conda activate smartgrasp

python test.py \
  --root data \
  --scene-id 1094 \
  --target-source id \
  --target-id 11 \
  --prior-prompt original \
  --ranking-score ig \
  --out-root runs_detail
```

这里 `--target-source id` 表示目标来自命令行的 `--target-id`。

## 方式 2：先用 intent VLM 解析目标 id

适合测试完整流程：自然语言指令先进入 intent 解析，再进入后面的分支判断和
抓取候选分析。

```bash
source /home/qiuguanhe/miniconda3/etc/profile.d/conda.sh
conda activate smartgrasp

python test.py \
  --root data \
  --scene-id 1094 \
  --target-source intent \
  --prior-prompt original \
  --ranking-score ig \
  --out-root runs_detail
```

这里 `--target-source intent` 会调用 `reason.intent_handle.resolve_intent(...)`。
它使用的配置默认来自 `run_intent.py`：

```text
RUN_INTENT_API_KEY_ENV
RUN_INTENT_BASE_URL
RUN_INTENT_MODEL
```

API key 仍然从 `.env` 对应环境变量读取；base url 和 model 名称写在
`run_intent.py` 里。

intent API timeout 默认是 `300` 秒，可以用下面参数覆盖：

```bash
--intent-timeout 300
```

`reason/vlm/config.py` 里的 VLM prior API timeout 也默认是 `300` 秒，但它和
`run_intent.py` 的 intent timeout 是独立配置。

如果不传 `--instruction`，代码会直接使用 `summary.json` 里的 `annotation`
作为 intent 指令。例如当前数据里可能是：

```json
{
  "annotation": "the topmost box",
  "point_source": "sam2-langsam"
}
```

这时 intent VLM 接收到的 instruction 就是 `"the topmost box"`。

## none 的处理

如果 intent VLM 判断当前场景没有合适目标，或者返回了无效 id，
`resolve_intent(...)` 会得到：

```text
target_object = None
```

`test.py` 现在不会给它编造一个新 id，也不会继续跑
`classify_branch` 或 handler。它会在输出里写一行：

```text
target_id = None
target_label = none
status = intent_no_target
reason = run_intent returned no target object
```

这样后续统计时能区分：

- intent 没找到目标。
- intent 找到目标，但后面的 branch/handler 失败。

## 和 graspability 对比实验一起使用

原始 information gain 实验：

```bash
python test.py \
  --root data \
  --scene-id 1094 \
  --target-source id \
  --target-id 11 \
  --prior-prompt original \
  --ranking-score ig \
  --out-root runs_detail
```

新的 `information_gain * graspability` 实验：

```bash
python test.py \
  --root data \
  --scene-id 1094 \
  --target-source id \
  --target-id 11 \
  --prior-prompt graspability \
  --ranking-score ig_graspability \
  --out-root runs_detail
```

如果要用 intent 解析出来的目标做同样对比，只需要把：

```text
--target-source id --target-id 11
```

换成：

```text
--target-source intent
```

## 输出字段

同级输出目录会额外生成：

```text
reason.txt
```

如果 `target_source=intent`，每个目标会记录两类理由：

- `intent_reason`：run_intent/intent VLM 为什么选择这个目标物体。
- `downstream_reason`：后续 branch/handler/VLM prior 为什么选择这个抓取或移除目标。

如果直接指定 `--target-source id --target-id ...`，则只记录后续分析的
`downstream_reason`。

`results.csv` 里新增了 intent 相关字段：

- `target_source`：`all`、`id` 或 `intent`。
- `intent_instruction`：传给 intent VLM 的自然语言指令。
- `intent_reason`：intent VLM 的选择理由。
- `intent_candidate_ids`：intent VLM 返回的候选目标 id。
- `intent_vlm_decision`：intent VLM 原始 JSON 结果。

`scene_details/scene_<id>.csv` 里会保留候选物级别的打分字段：

- `IG`
- `graspability`
- `score_ig`
- `score_ig_graspability`
- `score`
- `vlm_reason`
- `selected`

## Prompt 结构

intent prompt 参考 in-context affordance reasoning 的结构，但输出仍然只保留
object 选择，不输出 object part 或 affordance：

```text
Step 1. Task analysis
Step 2. Relevant object identification
Step 3. Spatial reasoning
```

要求 VLM 返回：

```json
{
  "target_present": true,
  "inferred_task": "short phrase or null",
  "target_object_id": 2,
  "target_category": "box",
  "candidate_object_ids": [1, 2],
  "reason": "short explanation"
}
```

`reason/vlm` 的 partial/invisible prior 也独立要求 VLM 返回自己的 `reason`，
用于解释候选分数或 graspability 分数；它和 intent 的 `reason` 是两套独立字段。

## 我跑过的检查

已经在 `smartgrasp` 环境里跑过：

```bash
python -m py_compile test.py
python test.py --help
python test.py \
  --root data \
  --scene-id 1094 \
  --target-source id \
  --target-id 11 \
  --prior-prompt original \
  --ranking-score ig \
  --out-root /tmp/test_integrated
```

第三个命令跑通，输出目录是：

```text
/tmp/test_integrated/gpt-5.5/original/ig/
```

当前沙箱环境不能解析 VLM API 域名，所以真实 intent VLM 路径没有在这里实网验证。
你在有网络和 `.env` API key 的环境里跑 `--target-source intent` 即可。
