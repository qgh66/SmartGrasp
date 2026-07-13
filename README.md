# SmartGrasp

## 快速开始

```bash
conda activate smartgrasp
cd /home/admin128/qiuguanhe/SmartGrasp
```

## 入口脚本

| 脚本 | 用途 | 包含模块 |
|------|------|---------|
| `perception/run_perception.sh` | 感知 + 默认 Reason | Perception → Reason |
| `run_pipeline.sh` | **★ 全流程** | Perception → Intent → Reason |
| `reason/run_reason.sh` | Reason 批量对比（多模型×多算法, closed-loop） | Reason only |
| `intent/run_intent.sh` | Intent 独立运行 | Intent only |

## 常用命令

```bash
# 全流程跑单个场景（推荐先试这个）
bash run_pipeline.sh 59

# 跑感知，并默认接着跑 Reason
bash perception/run_perception.sh 59

# 只跑感知
RUN_REASON_AFTER_PERCEPTION=0 bash perception/run_perception.sh 59

# 全量跑50个场景（耗时 2-4 小时）
bash run_pipeline.sh

# 指定多个场景
bash run_pipeline.sh 59 242 691

# 跳过 Intent，遍历所有物体
RUN_INTENT=0 bash run_pipeline.sh 59

# 指定物体 id，跳过 Intent
TARGET_ID=5 bash run_pipeline.sh 59

# 自定义自然语言指令（覆盖 summary.json annotation）
INSTRUCTION="拿左边的扳手" bash run_pipeline.sh 59

# 从 input/scene_59 读取 depth.npy 和 instruction.txt
bash run_pipeline.sh 59 --instruction=input
```

本地 RGB+depth 输入可以放在 `input/scene_<id>/`：

```text
input/scene_<id>/scene_image.png
input/scene_<id>/depth.npy
input/scene_<id>/summary.json       # 可选；默认从这里读 annotation
input/scene_<id>/input.txt          # 可选；存在时覆盖 annotation
```

`instruction.txt` 也会被兼容读取。只有这种 RGB+`depth.npy` 本地输入会读取
`input.txt`/`instruction.txt`；回退到 `.parquet` + `npz_file.zip` 时仍使用
parquet 里的 `annotation`。

## 模块说明

```
Parquet(RGB+标注) + NPZ(深度+GT掩码)
    │
    ▼
┌──────────────────────────────────────────┐
│ Perception (perception/run_perception.py)    │
│  SAM2 分割 → VLM 审阅 → 遮挡关系图       │
│  输出: data/scene_<id>/perception/       │
│        └─ summary.json  (下游统一入口)    │
└────────────────┬─────────────────────────┘
                 │ summary.json
    ┌────────────┴────────────┐
    ▼                         ▼
┌──────────────────┐  ┌──────────────────────┐
│ Intent           │  │ Reason               │
│ 自然语言→物体id   │  │ 遮挡图→分支→grasp评分 │
│ intent/run_intent.py │  │ reason/run_reason.py │
│                  │  │                      │
│ 输出: id.txt     │  │ 输出: results.csv    │
└────────┬─────────┘  └──────────┬───────────┘
         │ target_id              │
         └────────┬───────────────┘
                  │ grasp_id + branch
                  │  ⚠️ 尚未联通 Execution
                  ▼
┌──────────────────────────────────────────┐
│ Execution (PyBullet 抓取)                 │
│ execution/run_execution.py               │
│ 需要: object.name + scene_config         │
└──────────────────────────────────────────┘
```

## 输出路径

| 阶段 | 路径 | 核心文件 |
|------|------|---------|
| Perception | `data/scene_<id>/perception/` | `summary.json` |
| GT 参考 | `data/scene_<id>/gt/` | `summary.json` |
| Intent | `data/scene_<id>/intent/` | `intent_result.json` + `id.txt` |
| Reason | `data/scene_<id>/reason/` | `results.csv` |
| Reason (对比) | `runs_reason_compare/` | `results.csv` + 分析报告 |

## 依赖

- conda 环境: `smartgrasp`
- 数据: `data/train-*-of-*.parquet` + `data/npz_file.zip`
- GPU: SAM2 需要 CUDA
- 网络: VLM 审阅 / Intent / graspability 评分均需 API 访问
