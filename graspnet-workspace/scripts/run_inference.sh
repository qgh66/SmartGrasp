#!/usr/bin/env bash
# Pure GraspNet inference demo. Run from anywhere.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo " GraspNet Pure Inference Demo"
echo " ROOT: $ROOT"
echo "=========================================="

python scripts/demo_inference.py
