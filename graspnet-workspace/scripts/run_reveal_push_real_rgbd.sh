#!/usr/bin/env bash
# Reveal push simulation from real aligned RGB-D + mask.
# Required env vars:
#   RGB=/path/to/rgb.jpg
#   DEPTH=/path/to/aligned_depth.npy
#   MASK=/path/to/object_mask.png
#   INTRINSICS=/path/to/camera_intrinsics.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${RGB:?Set RGB=/path/to/aligned_rgb.jpg}"
: "${DEPTH:?Set DEPTH=/path/to/aligned_depth.npy}"
: "${MASK:?Set MASK=/path/to/object_mask.png}"
: "${INTRINSICS:?Set INTRINSICS=/path/to/camera_intrinsics.json}"

DEPTH_SCALE="${DEPTH_SCALE:-1000}"
MASS="${MASS:-0.05}"
FRICTION="${FRICTION:-0.7}"
DISTANCE="${DISTANCE:-0.05}"
ROBOT_MODEL="${ROBOT_MODEL:-jaka}"
OUTPUT="${OUTPUT:-results/reveal_push_real_rgbd.json}"

echo "=========================================="
echo " Reveal Push Simulation - Real RGB-D"
echo " ROOT:        $ROOT"
echo " RGB:         $RGB"
echo " Depth:       $DEPTH"
echo " Mask:        $MASK"
echo " Intrinsics:  $INTRINSICS"
echo " Robot model: $ROBOT_MODEL"
echo " Output:      $OUTPUT"
echo "=========================================="

python scripts/demo_reveal_push.py \
    --robot-model "$ROBOT_MODEL" \
    --rgb "$RGB" \
    --depth "$DEPTH" \
    --mask "$MASK" \
    --intrinsics "$INTRINSICS" \
    --depth-scale "$DEPTH_SCALE" \
    --mass "$MASS" \
    --friction "$FRICTION" \
    --distance "$DISTANCE" \
    --output "$OUTPUT"

VIZ="${OUTPUT%.json}_viz_data.pkl"
echo ""
echo "Done."
echo "Result: $ROOT/$OUTPUT"
echo "Viz:    $ROOT/$VIZ"
echo ""
echo "Open replay:"
echo "  RESULTS=$OUTPUT VIZ_DATA=$VIZ bash scripts/run_reveal_push_gui.sh"
