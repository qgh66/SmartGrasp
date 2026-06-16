#!/usr/bin/env bash
# Dash replay for saved Reveal push results. Run from anywhere.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8051}"
RESULTS="${RESULTS:-results/reveal_push_jaka.json}"
VIZ_DATA="${VIZ_DATA:-${RESULTS%.json}_viz_data.pkl}"

if [ ! -f "$RESULTS" ]; then
    echo "Results file not found: $RESULTS"
    echo "Generate it first, for example:"
    echo "  bash scripts/run_reveal_push_jaka.sh"
    exit 1
fi

if [ ! -f "$VIZ_DATA" ]; then
    echo "Viz data file not found: $VIZ_DATA"
    exit 1
fi

echo "=========================================="
echo " Reveal Push Dash Replay"
echo " ROOT:    $ROOT"
echo " Host:    $HOST"
echo " Port:    $PORT"
echo " Results: $RESULTS"
echo " Viz:     $VIZ_DATA"
echo "=========================================="

python gui/app.py \
    --host "$HOST" \
    --port "$PORT" \
    --results "$RESULTS" \
    --viz-data "$VIZ_DATA"
