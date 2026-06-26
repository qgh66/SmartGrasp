#!/usr/bin/env bash
# ============================================================================
# SmartGrasp Perception Pipeline — macOS 本地运行（无需 SLURM）
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

# ---- 日志文件（自动序号） ----
LOG_NUM=$(find logs -maxdepth 1 -name '*.log' 2>/dev/null | wc -l | tr -d ' ')
LOG_NUM=$((LOG_NUM + 1))
LOG_FILE="logs/$(printf "%03d" "$LOG_NUM")_scene_${1:-${SCENE_ID:-unknown}}.log"

# ---- Python 解释器 ----
# 优先用 smartgrasp-mac conda 环境的 python
PYTHON=""
for candidate in \
    "$HOME/anaconda3/envs/smartgrasp-mac/bin/python" \
    "/opt/anaconda3/envs/smartgrasp-mac/bin/python" \
    "$(which python3 2>/dev/null)"; do
    if [[ -x "$candidate" ]]; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "❌ 找不到 Python，请先激活 conda 环境: conda activate smartgrasp-mac" >&2
    exit 1
fi

# ---- 环境变量 ----
export SMARTGRASP_DATA_DIR="${SMARTGRASP_DATA_DIR:-$ROOT_DIR/data}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export MPLCONFIGDIR="/tmp/smartgrasp-matplotlib-${USER:-user}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export KMP_DUPLICATE_LIB_OK=TRUE

# ---- API 密钥 ----
# 优先用你 shell 里已 export 的，否则用默认值
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-zOddAlSLsleWkmOYXR1iDjtWwC7a745r2fKjU5wEvbGOTwMO}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://www.highland-api.top/v1}"

# ---- 运行参数 ----
# 支持单个场景: bash run_perception.sh 184
# 支持多个场景: bash run_perception.sh 184 59 125
# 也支持环境变量: SCENE_ID=184 bash run_perception.sh
if [[ $# -ge 2 ]]; then
    # 多个场景 → --scene-ids
    SCENE_ARGS=(--scene-ids "$@")
    SCENE_DISPLAY="$*"
elif [[ $# -eq 1 ]]; then
    SCENE_ARGS=(--scene-id "$1")
    SCENE_DISPLAY="$1"
else
    SCENE_ARGS=(--scene-id "${SCENE_ID:-527}")
    SCENE_DISPLAY="${SCENE_ID:-527}"
fi
MODE="${MODE:-vlm}"

# ---- SAM2 参数 ----
# 直接改这里即可；depth 留空表示继承对应的 RGB SAM2 参数。
SAM2_POINTS_PER_SIDE="${SAM2_POINTS_PER_SIDE:-24}"
SAM2_CROP_N_LAYERS="${SAM2_CROP_N_LAYERS:-0}"
SAM2_PRED_IOU_THRESH="${SAM2_PRED_IOU_THRESH:-0.68}"
SAM2_STABILITY_SCORE_THRESH="${SAM2_STABILITY_SCORE_THRESH:-0.83}"
DEPTH_SAM2_POINTS_PER_SIDE="${DEPTH_SAM2_POINTS_PER_SIDE:-}"
DEPTH_SAM2_CROP_N_LAYERS="${DEPTH_SAM2_CROP_N_LAYERS:-0}"
DEPTH_SAM2_PRED_IOU_THRESH="${DEPTH_SAM2_PRED_IOU_THRESH:-0.58}"
DEPTH_SAM2_STABILITY_SCORE_THRESH="${DEPTH_SAM2_STABILITY_SCORE_THRESH:-0.73}"

if [[ -n "${DEPTH_SAM2_POINTS_PER_SIDE:-}" ]]; then
    SCENE_ARGS+=(--depth-sam2-points-per-side "$DEPTH_SAM2_POINTS_PER_SIDE")
fi
if [[ -n "${DEPTH_SAM2_CROP_N_LAYERS:-}" ]]; then
    SCENE_ARGS+=(--depth-sam2-crop-n-layers "$DEPTH_SAM2_CROP_N_LAYERS")
fi
if [[ -n "${DEPTH_SAM2_PRED_IOU_THRESH:-}" ]]; then
    SCENE_ARGS+=(--depth-sam2-pred-iou-thresh "$DEPTH_SAM2_PRED_IOU_THRESH")
fi
if [[ -n "${DEPTH_SAM2_STABILITY_SCORE_THRESH:-}" ]]; then
    SCENE_ARGS+=(--depth-sam2-stability-score-thresh "$DEPTH_SAM2_STABILITY_SCORE_THRESH")
fi

# ---- 终端提示 ----
echo "[$(printf "%03d" "$LOG_NUM")] scene ${SCENE_DISPLAY} → ${LOG_FILE}"

# ---- 运行（全部输出写入 log） ----
{
    echo "========================================="
    echo " SmartGrasp Perception (macOS)"
    echo "========================================="
    echo "  Python:    $PYTHON"
    echo "  Scene(s):  $SCENE_DISPLAY"
    echo "  Mode:      $MODE"
    echo "  Data dir:  $SMARTGRASP_DATA_DIR"
    echo "  API URL:   $OPENAI_BASE_URL"
    echo "  API Key:   ${OPENAI_API_KEY:+✓ 已设置}"
    echo "========================================="
    echo ""

    START_TIME=$(date +%s)

    "$PYTHON" -u perception/perception.py \
        "${SCENE_ARGS[@]}" \
        --mode "$MODE" \
        --review-model-id "${REVIEW_MODEL_ID:-gpt-5.5}" \
        --review-api-key-env OPENAI_API_KEY \
        --review-base-url "$OPENAI_BASE_URL" \
        --review-timeout "${REVIEW_TIMEOUT:-300}" \
        --epsilon "${EPSILON:-0.05}" \
        --kernel-size "${KERNEL_SIZE:-5}" \
        --min-contact-pixels "${MIN_CONTACT_PIXELS:-50}" \
        --min-contact-ratio "${MIN_CONTACT_RATIO:-0.002}" \
        --mask-clean-kernel "${MASK_CLEAN_KERNEL:-3}" \
        --proposal-min-area-ratio "${PROPOSAL_MIN_AREA_RATIO:-0.006}" \
        --proposal-max-area-ratio "${PROPOSAL_MAX_AREA_RATIO:-0.11}" \
        --proposal-border-fraction-threshold "${PROPOSAL_BORDER_FRACTION_THRESHOLD:-0.18}" \
        --sam2-points-per-side "$SAM2_POINTS_PER_SIDE" \
        --sam2-crop-n-layers "$SAM2_CROP_N_LAYERS" \
        --sam2-pred-iou-thresh "$SAM2_PRED_IOU_THRESH" \
        --sam2-stability-score-thresh "$SAM2_STABILITY_SCORE_THRESH" \
        --preserve-unclaimed-sam2 "${PRESERVE_UNCLAIMED_SAM2:-18}" \
        ${DEVICE:+--device "$DEVICE"} \
        ${DEBUG:+--debug "$DEBUG"} \
        ${SAVE_CANDIDATES:+--save-candidates}

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "========================================="
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo " ✅ PASS  (${DURATION}s)"
    else
        echo " ❌ FAIL  (${DURATION}s, exit code=$EXIT_CODE)"
    fi
    echo "========================================="
} &> "$LOG_FILE"

# ---- 终端显示结果 ----
if grep -q "✅ PASS" "$LOG_FILE" 2>/dev/null; then
    echo "    ✅ PASS  — $LOG_FILE"
else
    echo "    ❌ FAIL  — $LOG_FILE"
fi
