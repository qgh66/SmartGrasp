# SSR — Segmentation Success Rate 测试框架

基于 FreeGrasp 论文的 SSR 指标，对 SmartGrasp 的 perception 分割质量进行系统性评估。

## 文件说明

| 文件 | 用途 |
|------|------|
| `prepare.py` | 从 parquet 生成任务清单（`tasks.json` + `scene_lists/*.txt`） |
| `run_all.sh` | 以 scene 为单位：perception → intent → reason → organize |
| `run_all_reason.sh` | **仅重跑 intent + reason**，基于已有 perception 结果 |
| `evaluate_ssr.py` | 计算 SSR 指标（模型预测 mask vs GT mask 的 IoU） |
| `evaluate_ssr.sh` | SSR 计算的 shell 封装 |
| `results/` | SSR 结果输出目录 |

## 快速开始

```bash
conda activate smartgrasp

# 1. 生成任务清单（300 条，含多 query 场景）
python ssr/prepare.py

# 2. 跑全部（6 类共 300 场景，非常耗时）
bash ssr/run_all.sh

# 3. 只跑某一类
bash ssr/run_all.sh easy

# 4. 只跑某一个 scene
bash ssr/run_all.sh easy 0
# 跨类别混合，完全 OK
bash ssr/run_all.sh 1449 1670 1755 1860 976 1365 3576 6971 7029

# 5. 断点续跑（从指定 scene_id 开始）
bash ssr/run_all.sh --from 1556           # 全部类别，从 scene_1556 开始
bash ssr/run_all.sh hard-ambi --from 1556 # 指定类别，从 scene_1556 开始

# 6. 指定 query（多 query 场景的第二 query，目录名 scene_{id}_q{qid}）
bash ssr/run_all.sh --query 4 1449           # scene_1449_q4 in hard-ambi
bash ssr/run_all.sh --query 4 1449 --query 9 1755  # 多个 scene，不同 query
bash ssr/run_all.sh --query 4 1449 1670 --query 9 1755  # query 与默认混合，1670 跑默认
```

### 只重跑 intent + reason（不重跑 perception）

当 perception 已经跑完，只想换模型/算法重新跑 intent 和 reason 时使用：

```bash
# 全部 6 类
bash ssr/run_all_reason.sh --all

# 重跑某个类
bash ssr/run_all_reason.sh medium

# 从指定 scene_id 开始（断点续跑）
bash ssr/run_all_reason.sh medium --from 79
bash ssr/run_all_reason.sh --all --from 5000

# 只跑指定场景
bash ssr/run_all_reason.sh medium 79 206 348 6862
```

脚本顶部可切换模型和算法：
```bash
REASON_MODEL="gpt-4o"
INTENT_MODEL="gpt-4o"
REASON_ARGS="
  --model ${REASON_MODEL}
  --intent-model ${INTENT_MODEL}
  --ranking-score ig
"
```

与 `run_all.sh` 的区别：
- **跳过** perception（不重新跑 SAM2 + VLM review）
- 自动检测 perception 是否存在，不存在则跳过
- 自动清除旧的 `intent/` 和 `reason/` 子目录
- 直接从 `data/{category}/` 读取 perception，无需软链接


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
│   ├── scene_0/
│   │   ├── gt/                    # GT 感知，1 份
│   │   ├── perception/            # VLM 感知，1 份
│   │   ├── intent/                # 意图解析，3 份（不同 annotation）
│   │   │   ├── split0/intent_result.json
│   │   │   ├── split1/intent_result.json
│   │   │   └── split2/intent_result.json
│   │   └── reason/                # 推理结果，3 份（不同 annotation）
│   │       ├── split0/results.csv
│   │       ├── split1/results.csv
│   │       └── split2/results.csv
│   └── ...
├── easy-ambi/
│   ├── scene_976/
│   └── scene_976_q1/              # 第二 query（perception → ../scene_976/perception 软链接）
├── medium/
│   ├── scene_1449/
│   ├── scene_1670/
│   ├── scene_1755/
│   ├── scene_1860/
│   ├── scene_7029/
│   └── scene_7029_q7/             # 第二 query
├── medium-ambi/
│   └── scene_1860_q7/             # scene_1860 的第二 query
├── hard/
│   ├── scene_3576/
│   ├── scene_3576_q3/
│   ├── scene_6971/
│   └── scene_6971_q18/
└── hard-ambi/
    ├── scene_1365/
    ├── scene_1365_q3/
    ├── scene_1449_q4/
    ├── scene_1755_q9/
    └── ...
```

> 带 `_qN` 后缀的目录是同一场景的第二 query object，`perception/` 和 `gt/` 通过软链接复用主场景的数据。

## 6 类场景数量

| 类别 | 场景数 |
|------|--------|
| easy | 50 |
| easy-ambi | 50 |
| medium | 50 |
| medium-ambi | 50 |
| hard | 50 |
| hard-ambi | 50 |
| **总计** | **300** |

> 注：9 个场景有 2 个 query object（不同难度），第二个 query 以 `scene_{id}_q{qid}` 命名，自动复用第一个 query 的 perception 结果。

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
