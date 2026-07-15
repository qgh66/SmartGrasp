# SSR — Segmentation Success Rate 测试框架

基于 FreeGrasp 论文的 SSR 指标，对 SmartGrasp 的 perception 分割质量进行系统性评估。

## 文件说明

| 文件 | 用途 |
|------|------|
| `prepare.py` | 从 parquet 生成任务清单（`tasks.json` + `scene_lists/*.txt`） |
| `run_all.sh` | 以 scene 为单位：perception → intent → reason → organize |
| `evaluate_ssr.py` | 计算 SSR 指标（模型预测 mask vs GT mask 的 IoU） |
| `evaluate_ssr.sh` | SSR 计算的 shell 封装 |
| `results/` | SSR 结果输出目录 |

## 快速开始

```bash
conda activate smartgrasp

# 1. 生成任务清单
python ssr/prepare.py

# 2. 跑全部（6 类共 291 场景，非常耗时）
bash ssr/run_all.sh

# 3. 只跑某一类
bash ssr/run_all.sh easy

# 4. 只跑某一个 scene
bash ssr/run_all.sh easy 0
```

## 执行流程（逐 scene）

```
For each scene:
  [0/4] mkdir -p data/{category}/scene_X/{gt,perception,intent/split{0,1,2},reason/split{0,1,2}}
  [1/4] VLM perception  → data/scene_X/perception/
  [2/4] GT perception   → data/scene_X/gt/ → mv to target
  [3/4] intent + reason × 3 splits（不同 annotation）
         intent  → data/scene_X/intent_splitN/
         reason  → data/scene_X/reason_splitN/
  [4/4] mv perception + splits → data/{category}/scene_X/
```

## 目录结构（最终）

```
data/
├── easy/
│   └── scene_0/
│       ├── gt/                    # GT 感知，1 份
│       │   ├── summary.json
│       │   ├── occlusion_graph.json
│       │   └── mask/
│       ├── perception/            # VLM 感知，1 份
│       │   ├── summary.json
│       │   ├── mask/
│       │   └── ...
│       ├── intent/                # 意图解析，3 份（不同 annotation）
│       │   ├── split0/intent_result.json
│       │   ├── split1/intent_result.json
│       │   └── split2/intent_result.json
│       └── reason/                # 推理结果，3 份（不同 annotation）
│           ├── split0/results.csv
│           ├── split1/results.csv
│           └── split2/results.csv
├── easy-ambi/
├── medium/
├── medium-ambi/
├── hard/
└── hard-ambi/
```

## 6 类场景数量

| 类别 | 场景数 |
|------|--------|
| easy | 50 |
| easy-ambi | 48 |
| medium | 49 |
| medium-ambi | 49 |
| hard | 48 |
| hard-ambi | 47 |
| **总计** | **291** |

## 计算 SSR

```bash
# 单类
bash ssr/evaluate_ssr.sh easy

# 多类
bash ssr/evaluate_ssr.sh easy medium hard

# 全部 6 类
bash ssr/evaluate_ssr.sh --all

# 逐场景详细输出
bash ssr/evaluate_ssr.sh -v easy
```

结果写入 `ssr/results/{category}_ssr.json`：
```json
{
  "category": "easy",
  "iou_threshold": 0.5,
  "total": 80,
  "success": 65,
  "ssr": 0.8125,
  "details": [
    {
      "scene_id": 0,
      "split": "split0",
      "grasp_id": 4,
      "gt_object_ids": [3],
      "best_gt_id": 3,
      "iou": 0.9931,
      "success": true
    }
  ]
}
```

## SSR 定义

对每个 scene × split（3 个 annotation）：
1. 从 `reason/splitN/results.csv` 取 `grasp_id`（模型预测要抓的物体）
2. 找 perception mask：`perception/mask/{grasp_id:03d}_*.png`
3. 找 GT mask：`gt/mask/mask_{groundTruthObjId+1:03d}_gt.png`（groundTruthObjIds 是 0-based）
4. 算 IoU，≥ 0.5 视为分割成功
5. SSR = 成功数 / 总数

## 注意

- GT 和 perception 每个场景只跑一次（与 annotation split 无关）
- intent 和 reason 每个场景跑 3 次（3 个不同的 annotation split）
- `groundTruthObjIds` 是 0-based，GT mask 文件是 1-based，需要 +1
- 不动源码，所有脚本在 `ssr/` 目录内
- 环境变量已在 `run_all.sh` 中设置
