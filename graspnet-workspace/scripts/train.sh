#!/usr/bin/env bash
# ============================================================
# GraspNet 训练脚本
# ============================================================
# 用法:
#   conda activate smartgrasp
#   cd /home/admin128/beilei/graspnet-workspace
#   bash scripts/train.sh
# ============================================================

set -e

ROOT=$(dirname $(dirname $(realpath $0)))
cd $ROOT

echo "=========================================="
echo " GraspNet Training"
echo " ROOT: $ROOT"
echo "=========================================="

# 参数配置
DATASET_ROOT="/path/to/graspnet_dataset"   # TODO: 修改为实际数据集路径
NUM_EPOCHS=20
BATCH_SIZE=4
NUM_WORKERS=4
LEARNING_RATE=0.001
LOG_DIR="logs/train_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Dataset: $DATASET_ROOT"
echo "Epochs:  $NUM_EPOCHS"
echo "Log:     $LOG_DIR"
echo ""

# 训练（调用 graspnet-baseline/train.py 的通用模式）
python -c "
import os, sys
sys.path.insert(0, '$ROOT')
sys.path.insert(0, os.path.join('$ROOT', 'models'))
sys.path.insert(0, os.path.join('$ROOT', 'pointnet2'))
sys.path.insert(0, os.path.join('$ROOT', 'utils'))
sys.path.insert(0, os.path.join('$ROOT', 'knn'))
sys.path.insert(0, os.path.join('$ROOT', 'dataset'))

import torch
from models.graspnet import GraspNet, pred_decode

print('✅ 导入成功，训练逻辑待补充')
"

echo ""
echo "✅ done"
