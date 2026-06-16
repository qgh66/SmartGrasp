#!/usr/bin/env bash
# ============================================================
# GraspNet + PyBullet 仿真测试脚本
# ============================================================
# 用法:
#   conda activate smartgrasp
#   cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace
#   bash scripts/demo_simulation.sh
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo " GraspNet + PyBullet Simulation Demo"
echo " ROOT: $ROOT"
echo "=========================================="

# 检查物体文件
OBJ_PATH="${OBJ_PATH:-/home/admin128/beilei/obj_phase3/002/textured.obj}"
if [ ! -f "$OBJ_PATH" ]; then
    echo "⚠️ 物体文件不存在: $OBJ_PATH"
    echo "  尝试使用其他物体..."
    OBJ_PATH=$(find /home/admin128/beilei/obj_phase3 -name "textured.obj" 2>/dev/null | head -1)
    if [ -z "$OBJ_PATH" ]; then
        echo "❌ 未找到任何物体文件！"
        exit 1
    fi
fi
echo "  Object: $OBJ_PATH"

# 检查 checkpoint
CKPT="${CKPT:-$ROOT/checkpoints/checkpoint-rs.tar}"
if [ ! -f "$CKPT" ]; then
    echo "⚠️ checkpoint 不存在: $CKPT"
    echo "  请将 checkpoint 放到该路径下"
    CKPT="/home/admin128/beilei/graspnet-baseline/checkpoints/checkpoint-rs.tar"
    if [ -f "$CKPT" ]; then
        echo "  使用源项目 checkpoint: $CKPT"
    else
        echo "❌ checkpoint 未找到！"
        exit 1
    fi
fi

echo ""
echo "Parameters:"
echo "  Object:     $OBJ_PATH"
echo "  Checkpoint: $CKPT"
echo "  Device:     cpu (默认)"
echo "  Top-K:      10"
echo "  Mode:       DIRECT (无GUI窗口，仅终端输出)"
echo ""

python -m simulation.run_sim \
    --obj_path "$OBJ_PATH" \
    --checkpoint_path "$CKPT" \
    --device cpu \
    --top_k 10 \
    --output results_simulation_demo.json \
    --random_orientation 2>&1 | tee simulation_demo.log

echo ""
echo "✅ 仿真完成，结果保存到 results_simulation_demo.json"
