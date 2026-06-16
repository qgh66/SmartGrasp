#!/usr/bin/env bash
# GraspNet + PyBullet grasp simulation. Run from anywhere.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OBJ_PATH="${OBJ_PATH:-/home/admin128/beilei/obj_phase3/002/textured.obj}"
CKPT="${CKPT:-$ROOT/checkpoints/checkpoint-rs.tar}"
DEVICE="${DEVICE:-cpu}"
TOP_K="${TOP_K:-10}"
OUTPUT="${OUTPUT:-results_simulation_demo.json}"

if [ ! -f "$OBJ_PATH" ]; then
    FOUND_OBJ="$(find /home/admin128/beilei/obj_phase3 -name "textured.obj" 2>/dev/null | head -1 || true)"
    if [ -n "$FOUND_OBJ" ]; then
        OBJ_PATH="$FOUND_OBJ"
    else
        echo "Object mesh not found. Set OBJ_PATH=/path/to/textured.obj"
        exit 1
    fi
fi

if [ ! -f "$CKPT" ]; then
    FALLBACK_CKPT="/home/admin128/beilei/graspnet-baseline/checkpoints/checkpoint-rs.tar"
    if [ -f "$FALLBACK_CKPT" ]; then
        CKPT="$FALLBACK_CKPT"
    else
        echo "Checkpoint not found. Set CKPT=/path/to/checkpoint-rs.tar"
        exit 1
    fi
fi

echo "=========================================="
echo " GraspNet + PyBullet Grasp Simulation"
echo " ROOT:       $ROOT"
echo " Object:     $OBJ_PATH"
echo " Checkpoint: $CKPT"
echo " Device:     $DEVICE"
echo " Top-K:      $TOP_K"
echo " Output:     $OUTPUT"
echo "=========================================="

python -m simulation.run_sim \
    --obj_path "$OBJ_PATH" \
    --checkpoint_path "$CKPT" \
    --device "$DEVICE" \
    --top_k "$TOP_K" \
    --output "$OUTPUT" \
    --random_orientation
