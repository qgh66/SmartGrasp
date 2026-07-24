# SmartGrasp

基于视觉语言模型（VLM）的遮挡场景机器人抓取系统。给定 RGB-D 图像和自然语言指令，自动完成物体分割、遮挡关系推理、抓取目标选择。

---

## 目录

1. [环境配置](#1-环境配置)
2. [数据准备](#2-数据准备)
3. [文件与脚本说明](#3-文件与脚本说明)
4. [快速开始](#4-快速开始)
5. [参数说明](#5-参数说明)
6. [技术路线](#6-技术路线)
7. [评估指标](#7-评估指标)
8. [输出结构](#8-输出结构)

---

## 1. 环境配置

### 1.1 Conda 环境

```bash
# 创建环境（Python 3.12）
conda create -n smartgrasp python=3.12
conda activate smartgrasp

# 安装 PyTorch（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
pip install openai opencv-python scipy scikit-learn pillow matplotlib pandas networkx pyarrow python-dotenv
```

### 1.2 SAM2 模型

```bash
# 克隆 SAM2 仓库
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..

# 下载 SAM2 权重（约 2.4 GB）
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 1.3 API 配置

在项目根目录创建 `api_config.json`（已加入 `.gitignore`，不会提交到 git）：

```json
{
  "api_key": "sk-你的密钥",
  "base_url": "https://yunwu.ai/v1"
}
```

所有 Python 和 Shell 脚本自动从这个文件读取 API 密钥和地址，无需手动设置环境变量。

### 1.4 验证安装

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from sam2.build_sam import build_sam2; print('SAM2 OK')"
python -c "import json; json.load(open('api_config.json')); print('API config OK')"
```

---

## 2. 数据准备

源数据需放在 `data/` 目录下：

| 文件 | 说明 |
|------|------|
| `data/train-00000-of-00002.parquet` | RGB 图像 + 标注文本 |
| `data/train-00001-of-00002.parquet` | RGB 图像 + 标注文本 |
| `data/npz_file.zip` | 深度图 + GT 实例分割掩码 |

数据集按两个维度分类，共 **6 类 291 个场景**：

| | easy | medium | hard |
|---|---|---|---|
| **非歧义** | easy (50) | medium (49) | hard (48) |
| **歧义** | easy-ambi (48) | medium-ambi (49) | hard-ambi (47) |

- **非歧义**（Non-ambiguous）：指令直接描述物体，如 `"the lemon"` —— 场景中只有一个柠檬
- **歧义**（Ambiguous）：场景中有多个同类物体，需空间消歧，如 `"the orange nearest to the image bottom"`

每场景有 **3 条不同的标注**（split 0/1/2），用于评估鲁棒性——同场景不同措辞的指令是否都能正确识别。

---

## 3. 文件与脚本说明

### 3.1 项目结构

```
SmartGrasp/
├── api_config.json            # API 密钥和地址（不入 git）
├── run_pipeline.sh            # 全流程入口：Perception → Intent → Reason
│
├── perception/                # ① 感知模块：物体分割 + 遮挡关系
│   ├── run_perception.sh      #    批量启动脚本
│   ├── run_perception.py      #    主程序入口
│   ├── sam2auto.py            #    SAM2 自动分割 + mask 质量过滤
│   ├── vlm.py                 #    VLM 调用：物体识别 + mask 分配
│   ├── occlusion_map.py       #    遮挡关系图（ORG）构建
│   ├── background.py          #    背景掩码生成
│   └── data_loader.py         #    数据加载（parquet + npz）
│
├── reason/                    # ②③ 推理模块：指令解析 + 抓取决策
│   ├── run_reason.py          #    推理主程序
│   ├── schemas.py             #    数据结构定义（Summary 等）
│   ├── graspability.py        #    VLM graspability 评分
│   ├── closed_loop.py         #    Closed-loop 模拟器
│   ├── branch_judge/          #    分支分类
│   │   └── classifier.py      #        fully_visible / partially_occluded / invisible
│   ├── fully_visible/         #    完全可见分支
│   │   └── handler.py         #        直接抓目标
│   ├── partially_visible/     #    部分遮挡分支
│   │   └── prior.py           #        祖先语义先验 + 图几何先验 → 熵排序
│   ├── invisible/             #    不可见分支
│   │   └── geometry.py        #        信息增益最大化
│   ├── intent_handle/         #    指令解析
│   │   └── intent_handler.py  #        VLM 将自然语言 → 目标物体 ID
│   └── vlm/                   #    VLM 客户端
│       ├── client.py          #        API 调用封装
│       └── config.py          #        模型/URL/Temperature 配置
│
├── intent/                    # Intent 独立运行入口
│   └── run_intent.py
│
├── ssr/                       # SSR（分割成功率）评估框架
│   ├── run_all.sh             #    批量运行全部场景（Perception→Intent→Reason→Organize）
│   ├── run_all_reason.sh      #    只重跑 intent+reason（复用已有 perception）
│   ├── evaluate_ssr.py        #    计算 SSR 指标
│   ├── evaluate_ssr.sh        #    SSR 计算脚本封装
│   ├── prepare.py             #    生成任务清单（tasks.json）
│   └── results/               #    SSR 结果输出
│
├── rsr/                       # RSR（推理成功率）评估框架
│   ├── run_rsr.sh             #    独立批量运行（含 perception）
│   ├── evaluate_rsr.py        #    计算 RSR 指标
│   ├── evaluate_rsr.sh        #    RSR 计算脚本封装
│   └── prepare_inputs.py      #    生成输入数据
│
├── tests/                     # 测试
│   ├── depth_gradient_masks/  #    深度边缘检测测试用例（23个标注样本）
│   │   ├── judge_mask.py      #       判定算法
│   │   ├── visualize_large_gradient.py  # 可视化工具
│   │   └── scene_*_mask_*/    #       测试 fixture
│   └── test_internal_depth_topology.py
│
├── data/                      # 原始数据（parquet + npz）
├── data-0716-55-ig/           # 0716 实验快照（gpt-5.5 + IG）
├── data-0716-4o-ig/           # 0716 实验快照（gpt-4o + IG）
├── data-0721-55-ig/           # 0721 实验快照（gpt-5.5 + IG）
├── logs/                      # 运行日志
├── runs_detail/               # 推理详情（per-model 输出）
└── runs_reason_current/       # 当前推理结果
```

### 3.2 核心脚本速查

| 脚本 | 用途 |
|------|------|
| `run_pipeline.sh 59` | 跑单个场景全流程（最快验证） |
| `ssr/run_all.sh` | 批量 SSR 评估（全部 291 场景） |
| `ssr/run_all.sh hard` | 只跑 hard 类别 |
| `ssr/run_all.sh hard hard-ambi` | 跑多个类别 |
| `ssr/run_all.sh --from 1548` | 断点续跑（从 scene_1548 开始） |
| `ssr/run_all_reason.sh hard` | 只重跑 intent+reason（复用 perception） |
| `perception/run_perception.sh 59` | 单独跑感知 |
| `ssr/evaluate_ssr.sh --all` | 计算全部 SSR 指标 |
| `rsr/evaluate_rsr.sh hard` | 计算 hard RSR 指标 |

### 3.3 Python 关键文件

| 文件 | 核心类/函数 | 职责 |
|------|-----------|------|
| `perception/sam2auto.py` | `_internal_depth_edge_report` | 深度边缘过滤（绿红桥梁算法） |
| `perception/vlm.py` | `review_and_assign_sam2` | VLM 物体识别 + mask 分配 |
| `perception/occlusion_map.py` | `build_occlusion_graph` | 遮挡关系图构建 |
| `reason/run_reason.py` | `main` | 推理主流程 |
| `reason/intent_handle/intent_handler.py` | `ResponsesVLMClient.choose_target` | Intent VLM 调用 |
| `reason/vlm/client.py` | `OpenAIVisionClient` | Reason VLM graspability 评分 |
| `reason/vlm/config.py` | `VLM_MODEL`, `VLM_BASE_URL` | 从 api_config.json 读取的配置 |

---

## 4. 快速开始

### 4.1 跑一个场景（最快验证，约 2-3 分钟）

```bash
conda activate smartgrasp
bash run_pipeline.sh 59
```

输出在 `data/scene_59/` 下。检查 `data/scene_59/perception/summary.json` 和 `data/scene_59/reason/results.csv`。

### 4.2 批量跑（SSR 评估）

```bash
# 全部 291 场景（耗时数小时）
bash ssr/run_all.sh

# 只跑特定类别
bash ssr/run_all.sh hard

# 跑多个类别
bash ssr/run_all.sh hard hard-ambi medium

# 断点续跑（跳过已完成的场景）
bash ssr/run_all.sh --from 1548

# 指定类别的断点续跑
bash ssr/run_all.sh hard --from 827
```

### 4.3 只重跑推理（跳过 perception）

当 perception 已经跑完，只想换模型或参数重跑 intent + reason：

```bash
# 全部类别
bash ssr/run_all_reason.sh --all

# 指定类别
bash ssr/run_all_reason.sh hard

# 断点续跑
bash ssr/run_all_reason.sh hard --from 5000
```

### 4.4 单独跑 perception

```bash
# 单个场景
bash perception/run_perception.sh 59

# 多个场景
bash perception/run_perception.sh 59 242 691

# 只跑 perception，不自动接 reason
RUN_REASON_AFTER_PERCEPTION=0 bash perception/run_perception.sh 59

# 输出 SAM2 debug 信息
DEBUG=sam2 bash perception/run_perception.sh 59
```

### 4.5 跑 intent（指令 → 目标物体 ID）

```bash
python -m intent.run_intent --scene-id 59 --instruction "the lemon"
```

### 4.6 跑 reason（遮挡推理 → 抓取物体 ID）

```bash
python -m reason.run_reason \
  --root data --scene-id 59 \
  --target-source auto \
  --instruction "the lemon" \
  --scene-root data \
  --model gpt-5.5 --intent-model gpt-5.5 --ranking-score ig
```

### 4.7 计算指标

```bash
# SSR
bash ssr/evaluate_ssr.sh --all
bash ssr/evaluate_ssr.sh hard

# RSR
bash rsr/evaluate_rsr.sh hard
```

---

## 5. 参数说明

### 5.1 Perception 参数

| 参数 | 当前值 | 说明 |
|------|:---:|------|
| `--mode` | `vlm` | `vlm`: SAM2 + VLM 全流程；`gt`: 仅生成 GT 遮挡图 |
| `--review-model-id` | `gpt-5.5` | VLM 审阅模型 |
| `--review-timeout` | `300` | VLM API 超时（秒） |
| `--sam2-points-per-side` | `24` | SAM2 自动分割的网格点密度，越大 mask 越多 |
| `--sam2-pred-iou-thresh` | `0.68` | SAM2 预测 IoU 阈值，低于此值的 mask 丢弃 |
| `--sam2-stability-score-thresh` | `0.83` | SAM2 稳定性阈值，低于此值的 mask 丢弃 |
| `--sam2-crop-n-layers` | `0` | SAM2 多尺度裁剪层数，0 表示不用多尺度 |
| `--depth-sam2-crop-n-layers` | `1` | 深度图 SAM2 的裁剪层数（深度图特征少，用多尺度补偿） |
| `--depth-sam2-pred-iou-thresh` | `0.58` | 深度图 SAM2 IoU 阈值（比 RGB 更宽松） |
| `--depth-sam2-stability-score-thresh` | `0.73` | 深度图 SAM2 稳定性阈值 |
| `--kernel-size` | `11` | 遮挡检测膨胀核（像素），越大越容易检测到远距离遮挡 |
| `--min-contact-pixels` | `50` | 最小接触像素数，低于此值不认为有遮挡关系 |
| `--min-contact-ratio` | `0.002` | 最小接触比例（接触像素 / 较小物体面积） |
| `--max-contact-background-ratio` | `0.3` | 接触区域最大背景占比（30%），超过则不认为有效接触 |
| `--mask-clean-kernel` | `3` | mask 清理核大小，用于去除 mask 边缘毛刺 |
| `--proposal-min-area-ratio` | `0.006` | 候选 mask 最小面积比（相对于图像总面积） |
| `--proposal-max-area-ratio` | `0.11` | 候选 mask 最大面积比，过大可能是背景 |
| `--proposal-border-fraction-threshold` | `0.18` | 候选 mask 接触图像边界的比例上限 |
| `--debug` | 无 | 设为 `sam2` 输出调试信息（debug_sam2.json） |

### 5.2 深度边缘过滤参数（sam2auto.py 内置）

用于过滤 SAM2 产生的内部有深度裂缝的 mask：

| 参数 | 当前值 | 说明 |
|------|:---:|------|
| 梯度阈值 | `0.40` | 绝对深度梯度阈值（blur=3, Sobel=1） |
| 最小绿点距离 | `8` px | 两绿色连通域间最小欧氏距离 |
| 最大区域比 | `30` | 切开后两大区域面积比上限 |
| 腐蚀像素 | `2` px | 从 mask 边界向内收缩量 |

在 22 个标注样本上准确率 **77%（17/22）**。

### 5.3 Reason 参数

| 参数 | 当前值 | 说明 |
|------|:---:|------|
| `--model` | `gpt-5.5` | Reason VLM 模型 |
| `--intent-model` | `gpt-5.5` | Intent VLM 模型 |
| `--ranking-score` | `ig` | 排序算法：`ig`（信息增益）或 `theory`（理论概率） |
| `--prior-prompt` | `graspability` | VLM prompt 模式，当前固定用 graspability |

### 5.4 VLM 通用配置（reason/vlm/config.py）

| 参数 | 值 | 说明 |
|------|:---:|------|
| `VLM_TEMPERATURE` | `0.0` | 确定性输出，消除 VLM 随机性 |
| `VLM_TIMEOUT` | `600` | API 超时（秒） |
| `VLM_MAX_RETRIES` | `0` | 不自动重试，失败即报错 |

---

## 6. 技术路线

```
输入: RGB 图像 + 深度图 + 自然语言指令
         │
         ▼
┌──────────────────────────────────────────────┐
│ ① Perception — 物体分割 + 遮挡关系图           │
│                                              │
│  1. SAM2 自动分割（RGB 源 + Depth 源双路）     │
│  2. 面积/边界/背景过滤                         │
│  3. 内部深度边缘过滤（绿红桥梁算法）            │
│  4. 深度重叠解析（按 z 轴优先级合并）           │
│  5. VLM 单次调用：物体识别 + mask 分配          │
│  6. 遮挡关系图（ORG）：膨胀接触 + 深度中值判向  │
│                                              │
│  输出: masks + summary.json + occlusion_graph │
└────────────────────┬─────────────────────────┘
                     │ summary.json
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌──────────────────────┐
│ ② Intent        │    │ ③ Reason              │
│ VLM 指令→物体ID  │    │ 遮挡图→分支→排序评分   │
│                 │    │                      │
│ 支持歧义消解     │    │ - fully_visible:      │
│ 例："the pear   │    │   直接抓取目标          │
│  under the      │    │                      │
│  other pear"    │    │ - partially_occluded:  │
│                 │    │   祖先语义×图几何→熵排序 │
│ 输出: target_id │    │                      │
│                 │    │ - invisible:          │
│                 │    │   信息增益最大化        │
└────────┬────────┘    │                      │
         │             │ 输出: grasp_id        │
         └──────┬──────┤       + results.csv   │
                ▼      └──────────────────────┘
           grasp_object
```

### 深度边缘过滤（绿红桥梁算法）

用于过滤 SAM2 产生的内部有深度裂缝的 mask（如一个物体被错误分成两半）：

1. 计算 mask 内部的**绝对深度梯度**（GaussianBlur=3, Sobel=1）
2. 只从**外边界**向内腐蚀 2px → 标记边界高梯度像素为**绿色**，内部高梯度像素为**红色**
3. 判断是否存在**绿 → 红 → 绿** 的 8-方向连通路径
4. 路径两端绿色连通域距离 ≥ 8px，且切开后两大区域面积比 ≤ 30×
5. 满足条件 → 判定 mask 内部有深度裂缝 → **拒绝该 mask 候选**

---

## 7. 评估指标

### SSR（Segmentation Success Rate）

评估 Perception 分割质量。VLM 输出的物体 mask 与 GT 物体 mask 的 **IoU ≥ 0.5** 即判定为成功。

```bash
bash ssr/evaluate_ssr.sh --all     # 全部 6 类
bash ssr/evaluate_ssr.sh hard      # 单类
```

结果保存在 `ssr/results/{category}_ssr.json`。

### RSR（Reasoning Success Rate）

评估端到端推理质量。模型预测的 **grasp 位姿** 与 GT 的 IoU ≥ 0.5 为成功。

```bash
bash rsr/evaluate_rsr.sh hard
```

---

## 8. 输出结构

```
data/{category}/scene_{id}/
├── perception/                # ① Perception 输出（1 份，被 3 条标注共享）
│   ├── summary.json           #    物体列表 + 遮挡图摘要（下游统一入口）
│   ├── mask/                  #    每个物体的二值 mask (PNG)
│   │   ├── 000_background_mask.png
│   │   ├── 001_anchor_<物体描述>.png
│   │   ├── 002_anchor_<物体描述>.png
│   │   └── ...
│   ├── occlusion_graph.json   #    遮挡关系图（节点 + 边 + 接触比例 + 背景占比）
│   ├── occlusion_graph.png    #    遮挡图可视化
│   ├── vlm.json               #    VLM 原始 JSON 输出
│   ├── label_1_sam2auto.png   #    SAM2 自动分割标签图
│   ├── label_2_vlm.png        #    VLM 审阅后的标签图
│   ├── scene_image.png        #    场景 RGB 原图
│   ├── scene_depth.png        #    深度可视化
│   ├── depth.npy              #    深度数据 (float32)
│   ├── points.json            #    SAM2 采样点
│   ├── final_objects_sheet.png#    最终物体剪切表
│   └── debug_sam2.json        #    SAM2 候选详情（DEBUG=sam2 时输出）
├── gt/                        # GT 参考
│   ├── summary.json
│   └── mask/
├── intent/                    # ② Intent 输出（3 条不同标注）
│   ├── split0/intent_result.json
│   ├── split1/intent_result.json
│   └── split2/intent_result.json
└── reason/                    # ③ Reason 输出（3 条不同标注）
    ├── split0/summary.json
    ├── split1/summary.json
    └── split2/summary.json
```

---

## 依赖

- **Python 3.12** + Conda 环境 `smartgrasp`
- **GPU（CUDA）**：SAM2 模型需要，CPU 模式极慢不推荐
- **网络**：VLM 调用（perception / intent / reason 均需访问 API）
- 主要 PyPI 包：`torch`, `sam2`, `opencv-python`, `scipy`, `scikit-learn`, `pillow`, `matplotlib`, `pandas`, `networkx`, `pyarrow`, `openai`, `python-dotenv`
