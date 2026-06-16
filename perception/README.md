# SmartGrasp Perception Pipeline

基于 SAM2 + VLM + LangSAM 的物体感知与遮挡关系图构建管线。

## 目录结构

```
SmartGrasp/
├── perception/
│   ├── _shared.py              # 共享工具（mask操作、IO、绘制、日志）
│   ├── background.py           # 深度背景检测（种子生成 + HSV颜色扩展）
│   ├── sam2auto.py             # SAM2 自动掩码生成 + 候选池 + 管道整合
│   ├── vlm_1_detection.py      # VLM 第1轮：从图像列出场景物体
│   ├── vlm_2_assemble.py       # VLM 第2轮：SAM2 碎片拼装为物体
│   ├── langsam.py              # LangSAM 文本引导分割 + 最佳掩码选择
│   ├── occlusion_map.py        # 遮挡图构建 + 掩码最终化 + 编号
│   ├── perception.py           # 入口（CLI + GT/VLM 双模式）
│   └── data_loader.py          # 数据加载（Parquet + NPZ）
├── run_perception.sh           # SLURM 提交脚本
├── perception_flowchart.md     # 管线流程图
└── data/                       # 场景数据（scene_*/）
```

### 模块依赖

```
_shared.py ◄── background.py ◄── sam2auto.py ◄── occlusion_map.py ◄── perception.py
    ◄── vlm_1_detection.py ◄── vlm_2_assemble.py ◄──┘
    ◄── langsam.py ◄────────────────────────────────┘
```

## 快速开始

### 环境

```bash
conda activate smartgrasp
```

### GT 模式（仅地面真值遮挡图）

不加载任何模型，直接使用数据集提供的实例掩码和深度图构建遮挡关系图：

```bash
# 单个场景
MODE=gt sbatch run_perception.sh --scene-id 184

# 多个场景
MODE=gt sbatch run_perception.sh --scene-ids 184 59 125
```

输出目录：`data/scene_{id}/gt/`

### VLM 模式（完整感知管线）

SAM2 → VLM 审阅 → LangSAM 精炼 → 遮挡图：

```bash
# 默认即为 VLM 模式
sbatch run_perception.sh --scene-id 184

# 批量
sbatch run_perception.sh --scene-ids 184 59 125
```

输出目录：`data/scene_{id}/perception/`

## 管线流程

参见 [perception_flowchart.md](perception_flowchart.md)

```
① 背景排除掩码（深度阈值 + HSV 颜色扩展）
    ↓
② SAM2 自动生成 → VLM 审阅 → LangSAM 精炼
    ②a SAM2 候选池（评分 + 过滤）
    ②b VLM 第1轮：列出场景物体
    ②c VLM 第2轮：SAM2 碎片拼装为物体
    ②d LangSAM 文本分割 + 最佳掩码选择
    ②e 未认领 SAM2 候选保留（深度连续性分组）
    ↓
③ 最终化：重叠消解 → 碎片过滤 → 重新编号
    ↓
④ 遮挡关系图构建（深度比较 + 接触检测）
```

## 输出文件

| 文件 | 说明 | GT | VLM |
|------|------|:---:|:---:|
| `occlusion_graph.json` | 遮挡关系图 | ✅ | ✅ |
| `occlusion_graph.png` | 遮挡图可视化 | ✅ | ✅ |
| `summary.json` | 场景摘要 | ✅ | ✅ |
| `scene_image.png` | 原始图像 | ✅ | ✅ |
| `depth.npy` | 深度图 | ✅ | ✅ |
| `mask/*.png` | 物体掩码 | - | ✅ |
| `mask/000_background_mask.png` | 背景排除掩码 | - | ✅ |
| `label_1_sam2auto.png` | SAM2 候选标注 | - | ✅ |
| `label_2_VLM_langsam.png` | VLM+LangSAM 结果 | - | ✅ |
| `label_3_final.png` | 最终掩码 | - | ✅ |
| `sam2_rgb_parts_sheet.png` | 候选切割图集 | - | ✅ |

## 主要参数

| 参数 | 默认值 | 说明 |
|------|------|------|
| `--scene-id` | - | 单个场景 ID |
| `--scene-ids` | - | 批量场景 ID |
| `--mode` | vlm | `gt` / `vlm` |
| `--epsilon` | 0.05 | 遮挡判定深度阈值 |
| `--kernel-size` | 5 | 接触检测膨胀核 |
| `--sam2-points-per-side` | 24 | SAM2 采样密度 |
| `--sam2-pred-iou-thresh` | 0.7 | SAM2 IoU 阈值 |
| `--sam2-stability-score-thresh` | 0.88 | SAM2 稳定性阈值 |
| `--preserve-unclaimed-sam2` | 24 | 保留未认领候选数 |

## 任务监控

```bash
# 确认场景完成
ls data/scene_{id}/gt/occlusion_graph.json        # GT 模式
ls data/scene_{id}/perception/occlusion_graph.json # VLM 模式

# 查看日志
tail -50 logs/perception-{jobid}.err
```
