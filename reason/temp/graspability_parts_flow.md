# Graspability Parts Flow

思路很简单：先在 perception 里把“物体 id”和“部件”建立映射，再把这个映射交给 VLM 估计每个候选物体的抓取难度。

## perception 侧怎么来的

`summary.json` 里会带：

- `object_id_to_sam2_part_ids`
- `object_id_to_sam2_part_files`
- `sam2_rgb_parts_sheet_png`

`reason/data_loader.py` 会把它们读进 `PerceptionOutput`：

- `object_id_to_sam2_part_ids`：某个物体 id 对应哪些 SAM2 part id
- `object_id_to_sam2_part_files`：这个物体对应的部件图文件
- `sam2_rgb_parts_sheet`：把所有部件拼在一起的可视化图

所以这里已经完成了：

```text
object id -> part ids / part files -> parts sheet image
```

## VLM 怎么用

在 `partial_visible` / `invisible` 的 prior 里，候选物体会连同这些信息一起传给 VLM：

- 物体 id
- 物体 label
- 对应的 SAM2 part ids
- labeled scene image
- `sam2_rgb_parts_sheet`

如果是 `prompt_mode=graspability`，prompt 会要求 VLM 返回一个综合
graspability：

- `scores`：原来的语义/遮挡分数
- `graspability`：综合考虑最佳可行 part/region 和整体物体移除难度后的抓取分数

例如：

```json
{
  "graspability": {
    "5": 0.65
  }
}
```

这里 VLM 会先观察可见 part/region，比如 handle、rim、flat surface 等，再综合判断：

- 最佳可抓部件是否暴露、可达、稳定
- 夹爪是否有足够接触面积和厚度
- 抓这个部件能否带动整个物体
- 移除时是否可能碰撞、滑移或不稳定

最后只返回一个综合分数：

```text
graspability = 0.65
```

## 最后怎么落到输出

VLM 返回后，代码使用 VLM 显式返回的综合 `graspability`：

```text
graspability(object) = graspability[object]
```

如果旧 prompt 或异常返回里没有 `graspability`，代码仍兼容旧格式，会退回到旧逻辑：

```text
graspability(object) = max(graspability_parts[object].values())
```

然后再合成最终排序分数：

```text
score_ig_graspability = IG * graspability
```

其中：

- `IG` 是原来的信息增益
- `graspability` 是 VLM 判断的综合抓取分数，缺失时退回到最大 part 分数

最后在 `scene_details/scene_<id>.csv` 里输出：

- `graspability`
- `score_ig`
- `score_ig_graspability`
- `selected`

也就是说，部件信息没有单独变成一个新模块，而是作为 perception 到 VLM 的辅助上下文。
VLM 会利用可见部件来判断最佳抓取区域，但最终只输出一个综合 graspability。
