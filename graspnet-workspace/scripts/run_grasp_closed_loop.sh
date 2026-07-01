#!/usr/bin/env bash
# GraspNet + PyBullet closed-loop grasp simulation for Dash replay.
# Run from anywhere.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-0}"

DEFAULT_PYTHON="/home/admin128/anaconda3/envs/smartgrasp/bin/python"
if [ -x "$DEFAULT_PYTHON" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

OBJ_PATH="${OBJ_PATH:-/home/admin128/beilei/obj_phase3/002/textured.obj}"
CKPT="${CKPT:-}"
DEVICE="${DEVICE:-cuda:0}"
TOP_K="${TOP_K:-5}"
OUTPUT="${OUTPUT:-results/grasp_closed_loop.json}"
ROBOT_MODEL="${ROBOT_MODEL:-jaka}"
DEMO_SNAP_TO_OBJECT="${DEMO_SNAP_TO_OBJECT:-1}"

if [ ! -f "$OBJ_PATH" ]; then
    FOUND_OBJ="$(find /home/admin128/beilei/obj_phase3 -name "textured.obj" 2>/dev/null | head -1 || true)"
    if [ -n "$FOUND_OBJ" ]; then
        OBJ_PATH="$FOUND_OBJ"
    else
        echo "Object mesh not found. Set OBJ_PATH=/path/to/textured.obj"
        exit 1
    fi
fi

if [ -z "$CKPT" ]; then
    for candidate in \
        "$ROOT/checkpoints/checkpoint-rs.tar" \
        "/home/admin128/beilei/graspnet-baseline/checkpoints/checkpoint-rs.tar"; do
        if [ -f "$candidate" ]; then
            CKPT="$candidate"
            break
        fi
    done
fi

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found. Set CKPT=/path/to/checkpoint-rs.tar"
    exit 1
fi

echo "=========================================="
echo " GraspNet + PyBullet Closed-loop Grasp"
echo " ROOT:       $ROOT"
echo " Object:     $OBJ_PATH"
echo " Checkpoint: $CKPT"
echo " Device:     $DEVICE"
echo " Top-K:      $TOP_K"
echo " Robot:      $ROBOT_MODEL"
echo " Demo snap:  $DEMO_SNAP_TO_OBJECT"
echo " Python:     $PYTHON_BIN"
echo " Output:     $OUTPUT"
echo "=========================================="

SNAP_ARGS=()
if [ "$DEMO_SNAP_TO_OBJECT" = "0" ]; then
    SNAP_ARGS+=(--no-demo-snap-to-object)
fi

"$PYTHON_BIN" scripts/demo_closed_loop.py \
    --obj "$OBJ_PATH" \
    --ckpt "$CKPT" \
    --top_k "$TOP_K" \
    --device "$DEVICE" \
    --robot-model "$ROBOT_MODEL" \
    --output "$OUTPUT" \
    "${SNAP_ARGS[@]}"

VIZ="${OUTPUT%.json}_viz_data.pkl"
echo ""
echo "Done."
echo "Result: $ROOT/$OUTPUT"
echo "Viz:    $ROOT/$VIZ"
echo ""
echo "Open replay:"
echo "  RESULTS=$OUTPUT VIZ_DATA=$VIZ PORT=8051 bash scripts/run_reveal_push_gui.sh"
