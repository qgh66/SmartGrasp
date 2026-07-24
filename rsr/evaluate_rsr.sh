#!/usr/bin/env bash
# 计算 RSR（Reasoning Success Rate）— 二值 IoU ≥ 0.5
# 用法: bash rsr/evaluate_rsr.sh [--all] [-v] <category...>

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$HOME/miniconda3/envs/smartgrasp/bin/python"
[[ -x "$PYTHON" ]] || { echo "❌ Python not found" >&2; exit 1; }

ALL_CATEGORIES=(easy easy-ambi medium medium-ambi hard hard-ambi)
CATEGORIES=()
VERBOSE=""

for arg in "$@"; do
    case "$arg" in
        --all)   CATEGORIES=("${ALL_CATEGORIES[@]}") ;;
        -v|--verbose) VERBOSE="--verbose" ;;
        *)       CATEGORIES+=("$arg") ;;
    esac
done

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
    echo "用法: bash rsr/evaluate_rsr.sh [--all] [-v] <category...>"
    echo "类别: ${ALL_CATEGORIES[*]}"
    exit 1
fi

exec "$PYTHON" "$ROOT_DIR/rsr/evaluate_rsr.py" $VERBOSE "${CATEGORIES[@]}"
