# SmartGrasp Perception Pipeline

基于 SAM2 自动分割 + VLM 单轮审阅 + 深度遮挡判定的物体感知管线。

## 目录结构

```
SmartGrasp/
├── perception/
│   ├── _shared.py              # 共享工具（mask操作、IO、绘制、日志）
│   ├── background.py           # 深度背景检测（种子生成 + HSV颜色扩展）
│   ├── sam2auto.py             # SAM2 自动掩码生成 + 候选池 + 管道整合
│   ├── vlm.py                  # VLM 单轮审阅：识别物体并分配 SAM2 id
│   ├── occlusion_map.py        # 遮挡图构建 + 掩码最终化 + 编号
│   ├── perception.py           # 入口（CLI + GT/VLM 双模式）
│   └── data_loader.py          # 数据加载（Parquet + NPZ）
├── run_perception.sh           # 本地运行脚本
├── perception_flowchart.md     # 管线流程图
└── data/                       # 场景数据（scene_*/）
```

### 模块依赖

```
_shared.py ◄── background.py ◄── sam2auto.py ◄── occlusion_map.py ◄── perception.py
    ◄── vlm.py ◄────────────────────────────────────┘
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
MODE=gt bash run_perception.sh 184

# 多个场景
MODE=gt bash run_perception.sh 184 59 125
```

输出目录：`data/scene_{id}/gt/`

### VLM 模式（完整感知管线）

SAM2 → VLM 审阅 → 掩码组装 → 遮挡图：

```bash
# 默认即为 VLM 模式
bash run_perception.sh 184

# 批量
bash run_perception.sh 184 59 125
```

输出目录：`data/scene_{id}/perception/`

## 管线流程

参见 [perception_flowchart.md](perception_flowchart.md)

```
① 背景排除掩码（默认 GT；可选深度 + HSV）
    ↓
② SAM2 自动生成 → VLM 单轮审阅 → 掩码组装
    ②a RGB + 深度图双路 SAM2 候选池
    ②b VLM 输出 objects + visible_parts + sam2_ids
    ②c 按 sam2_ids 合并候选 mask
    ↓
③ 遮挡关系图构建（接触检测 + 深度窄带比较）
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
| `label_2_vlm.png` | VLM 组装结果 | - | ✅ |
| `vlm.json` | VLM 审阅结果 | - | ✅ |
| `sam2_rgb_parts_sheet.png` | 候选切割图集 | - | ✅ |

## 主要参数

| 参数 | 默认值 | 说明 |
|------|------|------|
| `--scene-id` | - | 单个场景 ID |
| `--scene-ids` | - | 批量场景 ID |
| `--mode` | vlm | `gt` / `vlm` |
| `--kernel-size` | 5 | 接触检测膨胀核 |
| `--sam2-points-per-side` | 24 | SAM2 采样密度 |
| `--sam2-pred-iou-thresh` | 0.85 | SAM2 IoU 阈值 |
| `--sam2-stability-score-thresh` | 0.95 | SAM2 稳定性阈值 |
| `--mask` | gt | 背景排除来源：`gt` / `depth` |

## 任务监控

```bash
# 确认场景完成
ls data/scene_{id}/gt/occlusion_graph.json        # GT 模式
ls data/scene_{id}/perception/occlusion_graph.json # VLM 模式

# 查看日志
tail -50 logs/*.log
```

## 场景难度分级

数据集按遮挡链长度分为 Easy / Medium / Hard，按同类物体歧义分为 without Ambiguity / with Ambiguity。

### Hard without Ambiguity（48 个场景）

```
827 1113 1312 1318 1459 1733 1784 1942 1996 2274
3125 3576 4992 5155 5447 5778 6732 6755 6760 6784
6801 6807 6835 6837 6870 6873 6941 6949 6971 7074
7084 7108 7146 7175 7228 7239 7271 7281 7291 7310
7321 7340 7346 7348 7368 7387 7404 7433
```

### Hard with Ambiguity（49 个场景）

```
59 242 691 815 823 1072 1094 1101 1109 1365
1383 1394 1419 1449 1556 1657 1703 1709 1711 1755
1842 1958 1961 2014 2030 2035 2096 2186 2310 2355
2357 2804 2839 3486 3724 3727 4015 4018 4109 4156
4232 4570 5062 5076 5110 5223 5359 5368 5405
```

批量运行示例：

```bash
# 所有 Hard without Ambiguity
bash run_perception.sh 827 1113 1312 1318 1459 1733 1784 1942 1996 2274 3125 3576 4992 5155 5447 5778 6732 6755 6760 6784 6801 6807 6835 6837 6870 6873 6941 6949 6971 7074 7084 7108 7146 7175 7228 7239 7271 7281 7291 7310 7321 7340 7346 7348 7368 7387 7404 7433
```
