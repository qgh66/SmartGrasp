# Partial / Invisible Graspability Scoring Notes

这个说明对应当前 `reason` 里的新实验：

- 原始对比方式：只用 information gain 排序。
- 新方式：用 `information_gain * graspability` 排序。

这里没有新增独立的 VLM affordance 接口。`graspability` 是在原来的
`partially_visible` 和 `invisible` prior VLM 调用里，由同一个 prompt 一起返回的。

## 改动位置

- `reason/vlm/helper.py`
  - 保留原 prompt。
  - 新增 `graspability` prompt 模式，要求 VLM 返回：

```json
{
  "scores": {"<mid>": 0.0},
  "graspability": {"<mid>": 1.0}
}
```

- `reason/vlm/client.py`
  - 仍然使用原来的 `score_occluders_partial(...)` 和
    `score_occluders_invisible(...)`。
  - 没有新增新的 VLM 方法。
  - 当 `--prior-prompt graspability` 时，会把 `sam2_rgb_parts_sheet` 也传给 VLM。

- `reason/data_loader.py`
  - 从 `summary.json` 读取：
    - `sam2_rgb_parts_sheet_png`
    - `object_id_to_sam2_part_ids`
    - `object_id_to_sam2_part_files`
  - 放进 `PerceptionOutput`，供 VLM prompt 描述候选物的 SAM2 parts。

- `reason/partially_visible/handler.py`
  - 每个候选物同时记录：
    - `score_ig = IG`
    - `score_ig_graspability = IG * graspability`
    - `score`：本次实验实际用于排序的分数

- `reason/invisible/handler.py`
  - 同样记录 `score_ig`、`score_ig_graspability` 和 `score`。

## 运行方式

先进入环境：

```bash
source /home/qiuguanhe/miniconda3/etc/profile.d/conda.sh
conda activate smartgrasp
```

### 原始 information gain 实验

```bash
python test.py \
  --root data \
  --scene-id 1094 \
  --target-id 11 \
  --prior-prompt original \
  --ranking-score ig \
  --out-root runs_detail
```

这个模式使用原 prompt。`graspability` 会默认是 `1.0`，所以排序等价于只看：

```text
score = IG
```

### 新的 graspability 加权实验

```bash
python test.py \
  --root data \
  --scene-id 1094 \
  --target-id 11 \
  --prior-prompt graspability \
  --ranking-score ig_graspability \
  --out-root runs_detail
```

这个模式会要求 VLM 同时返回：

```text
scores: 原来的语义/遮挡 prior
graspability: 综合最佳可行 part/region、碰撞风险、整体移除稳定性的抓取难易程度
```

最终排序分数是：

```text
score = IG * graspability
```

## 输出位置

默认会按实验配置分目录，避免覆盖：

```text
runs_detail/<model>/original/ig/
runs_detail/<model>/graspability/ig_graspability/
```

每个目录里有：

```text
results.csv
branch_results.json
scene_details/scene_<id>.csv
```

重点看：

```text
scene_details/scene_<id>.csv
```

里面关键列：

- `candidate_id`：候选要移除/抓取的 top-layer object id。
- `IG`：information gain。
- `graspability`：VLM 判断的综合抓取难易程度。
- `score_ig`：只用 information gain 的分数。
- `score_ig_graspability`：`IG * graspability`。
- `score`：当前 `--ranking-score` 实际采用的排序分数。
- `selected`：最终被选中的候选物。

## 注意

如果 VLM 网络/API 调用失败，代码会 fallback：

```text
scores = 0.5 或 uniform
graspability = 1.0
```

这种情况下 `ig_graspability` 会退化得接近 `ig`，不能作为真实对比结果。
