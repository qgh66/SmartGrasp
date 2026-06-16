#!/usr/bin/env bash
# Reveal push simulation with the original simple box gripper.
# Run from anywhere.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DISTANCE="${DISTANCE:-0.05}"
OUTPUT="${OUTPUT:-results/reveal_push_simple.json}"
PORT="${PORT:-8051}"

echo "=========================================="
echo " Reveal Push Simulation - Simple Gripper"
echo " ROOT:     $ROOT"
echo " Distance: $DISTANCE"
echo " Output:   $OUTPUT"
echo "=========================================="

python scripts/demo_reveal_push.py \
    --robot-model simple \
    --distance "$DISTANCE" \
    --output "$OUTPUT"

VIZ="${OUTPUT%.json}_viz_data.pkl"
echo ""
echo "Done."
echo "Result: $ROOT/$OUTPUT"
echo "Viz:    $ROOT/$VIZ"
echo ""
echo "Open replay:"
echo "  RESULTS=$OUTPUT VIZ_DATA=$VIZ PORT=$PORT bash scripts/run_reveal_push_gui.sh"
