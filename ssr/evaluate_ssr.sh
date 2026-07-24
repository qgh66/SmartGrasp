#!/usr/bin/env bash
# 计算 SSR（Segmentation Success Rate）
# 用法:
#   bash ssr/evaluate_ssr.sh easy          → 单个类别
#   bash ssr/evaluate_ssr.sh easy medium   → 多个类别
#   bash ssr/evaluate_ssr.sh --all         → 全部 6 类
#   bash ssr/evaluate_ssr.sh -v easy       → 详细输出（逐场景）

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
    echo "用法: bash ssr/evaluate_ssr.sh [--all] [-v] <category...>"
    echo "类别: ${ALL_CATEGORIES[*]}"
    exit 1
fi

exec "$PYTHON" "$ROOT_DIR/ssr/evaluate_ssr.py" $VERBOSE "${CATEGORIES[@]}"
