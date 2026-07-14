#!/usr/bin/env bash
# ============================================================
# GraspNet + PyBullet 仿真测试脚本
# ============================================================
# 用法:
#   conda activate smartgrasp
#   cd /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace
#   bash scripts/demo_simulation.sh
# ============================================================

set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

echo "=========================================="
echo " GraspNet + PyBullet Simulation Demo"
echo " ROOT: $ROOT"
echo "=========================================="

# 检查物体文件
OBJ_PATH="${OBJ_PATH:-$ROOT/assets/objects/industrial_tools/ycb/050_medium_clamp/google_16k/textured.obj}"
if [ ! -f "$OBJ_PATH" ]; then
    echo "Object file not found: $OBJ_PATH"
    echo "  Trying another repository-local object..."
    OBJ_PATH=$(find "$ROOT/assets/objects" -name "textured.obj" -print -quit 2>/dev/null)
    if [ -z "$OBJ_PATH" ]; then
        echo "No repository-local textured.obj found."
        exit 1
    fi
fi
echo "  Object: $OBJ_PATH"

# 检查 checkpoint
CKPT="$ROOT/checkpoints/checkpoint-rs.tar"
if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT"
    echo "Place checkpoint-rs.tar under $ROOT/checkpoints/ or set CKPT."
    exit 1
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
