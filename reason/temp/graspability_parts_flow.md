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

如果是 `prompt_mode=graspability`，prompt 会要求 VLM 对每个候选物体的
SAM2 part 分别打分：

- `scores`：原来的语义/遮挡分数
- `graspability_parts`：每个 object id 下面，每个 part id 的抓取难度

例如：

```json
{
  "graspability_parts": {
    "5": {
      "12": 0.3,
      "18": 0.8
    }
  }
}
```

这里 object `5` 的 part `18` 分数最高，所以：

```text
graspability = 0.8
graspability_part_id = 18
```

## 最后怎么落到输出

VLM 返回后，代码先把每个物体的 part 分数压成一个物体级分数：

```text
graspability(object) = max(graspability_parts[object].values())
```

然后再合成最终排序分数：

```text
score_ig_graspability = IG * graspability
```

其中：

- `IG` 是原来的信息增益
- `graspability` 是该物体所有 SAM2 parts 里的最大抓取分数
- `graspability_part_id` 是这个最大分数对应的 part id

最后在 `scene_details/scene_<id>.csv` 里输出：

- `graspability`
- `graspability_part_id`
- `graspability_parts`
- `score_ig`
- `score_ig_graspability`
- `selected`

`reason.txt` 的 downstream reason 里也会出现类似：

```text
G=0.800 best_part=18
```

也就是说，部件信息没有单独变成一个新模块，而是作为 perception 到 VLM 的辅助上下文，
帮助 VLM 判断“这个 top-layer 物体哪个露出部件最好抓”，最后取最高 part 分数作为
这个物体的 graspability。
