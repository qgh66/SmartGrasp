#!/usr/bin/env bash
# ============================================================================
# SmartGrasp Perception Pipeline — Linux + 批量 + API 错误自动恢复
# ============================================================================
# 用法：
#   bash perception/run_perception.sh              → 跑全部场景
#   bash perception/run_perception.sh 59           → 跑单个场景
#   bash perception/run_perception.sh 59 242 691   → 跑指定多个场景
# 默认跑 perception + reason。如需只跑 perception:
#   RUN_REASON_AFTER_PERCEPTION=0 bash perception/run_perception.sh 59
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

# ---- 全量场景列表 ----
ALL_SCENES=(
  59 242 691 815 823 1072 1094 1101 1109 1365 1383 1394 1419 1449 1556 1657 1703
  1709 1711 1755 1842 1958 1961 2014 2030 2035 2096 2186 2310 2355 2357 2804 2839
  3486 3724 3727 4015 4018 4109 4156 4232 4570 5062 5076 5110 5223 5359 5368 5405
)

# ---- Python 解释器 ----
PYTHON=""
for candidate in \
    "$HOME/anaconda3/envs/smartgrasp/bin/python" \
    "$HOME/miniconda3/envs/smartgrasp/bin/python" \
    "/opt/anaconda3/envs/smartgrasp/bin/python" \
    "$(which python3 2>/dev/null)"; do
    if [[ -x "$candidate" ]]; then
        PYTHON="$candidate"
        break
    fi
done
[[ -z "$PYTHON" ]] && { echo "❌ 找不到 Python，请先: conda activate smartgrasp" >&2; exit 1; }

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
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(python3 -c "import json;print(json.load(open('$ROOT_DIR/api_config.json'))['api_key'])")}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$(python3 -c "import json;print(json.load(open('$ROOT_DIR/api_config.json'))['base_url'])")}"

# ---- 参数 ----
MODE="${MODE:-vlm}"
SAM2_PPS="${SAM2_POINTS_PER_SIDE:-24}"
SAM2_CNL="${SAM2_CROP_N_LAYERS:-0}"
SAM2_PIT="${SAM2_PRED_IOU_THRESH:-0.68}"
SAM2_SST="${SAM2_STABILITY_SCORE_THRESH:-0.83}"
D_PPS="${DEPTH_SAM2_POINTS_PER_SIDE:-}"
D_CNL="${DEPTH_SAM2_CROP_N_LAYERS:-1}"
D_PIT="${DEPTH_SAM2_PRED_IOU_THRESH:-0.58}"
D_SST="${DEPTH_SAM2_STABILITY_SCORE_THRESH:-0.73}"

# ---- reason-after-perception 配置：当次场景，单步非 closed-loop ----
RUN_REASON_AFTER_PERCEPTION="${RUN_REASON_AFTER_PERCEPTION:-1}"
REASON_PYTHON="${REASON_PYTHON:-$PYTHON}"
REASON_DATA_ROOT="${REASON_DATA_ROOT:-data}"
REASON_OUT_ROOT="${REASON_OUT_ROOT:-runs_reason_current}"
REASON_MODEL="${REASON_MODEL:-${REVIEW_MODEL_ID:-gpt-5.5}}"
REASON_PRIOR_PROMPT="${REASON_PRIOR_PROMPT:-graspability}"
REASON_RANKING_SCORE="${REASON_RANKING_SCORE:-ig_graspability}"
REASON_TARGET_SOURCE="${REASON_TARGET_SOURCE:-auto}"

check_source_inputs() {
    local ok=0
    compgen -G "$SMARTGRASP_DATA_DIR/*.parquet" >/dev/null || {
        echo "  missing source: $SMARTGRASP_DATA_DIR/*.parquet"
        ok=1
    }
    [[ -f "$SMARTGRASP_DATA_DIR/npz_file.zip" ]] || {
        echo "  missing source: $SMARTGRASP_DATA_DIR/npz_file.zip"
        ok=1
    }
    return "$ok"
}

check_scene_outputs() {
    local ok=0
    local scene_id
    for scene_id in "$@"; do
        local perception_dir="$ROOT_DIR/data/scene_${scene_id}/perception"
        local required=(
            "$perception_dir/summary.json"
            "$perception_dir/scene_image.png"
            "$perception_dir/depth.npy"
        )
        local path
        for path in "${required[@]}"; do
            [[ -f "$path" ]] || {
                echo "  missing output: $path"
                ok=1
            }
        done
    done
    return "$ok"
}

run_reason_for_scenes() {
    local scenes=("$@")
    if [[ "$RUN_REASON_AFTER_PERCEPTION" != "1" ]]; then
        echo ""
        echo "Reason after perception: skipped (RUN_REASON_AFTER_PERCEPTION=$RUN_REASON_AFTER_PERCEPTION)"
        return 0
    fi

    local -a scene_args
    if [[ ${#scenes[@]} -eq 1 ]]; then
        scene_args=(--scene-id "${scenes[0]}")
    else
        scene_args=(--scene-ids "${scenes[@]}")
    fi

    local -a target_args=(--target-source "$REASON_TARGET_SOURCE")
    if [[ -n "${REASON_TARGET_ID:-}" ]]; then
        target_args=(--target-source id --target-id "$REASON_TARGET_ID")
    fi
    if [[ -n "${REASON_INSTRUCTION:-}" ]]; then
        target_args+=(--instruction "$REASON_INSTRUCTION")
    fi

    echo ""
    echo "========================================="
    echo " SmartGrasp Reason (current perception run)"
    echo "========================================="
    echo "  Python:          $REASON_PYTHON"
    echo "  Scenes:          ${scenes[*]}"
    echo "  Data root:       $REASON_DATA_ROOT"
    echo "  Out root:        $REASON_OUT_ROOT"
    echo "  Model:           $REASON_MODEL"
    echo "  Prior prompt:    $REASON_PRIOR_PROMPT"
    echo "  Ranking score:   $REASON_RANKING_SCORE"
    echo "  Target source:   ${target_args[*]}"
    echo "  Closed loop:     disabled"
    echo "========================================="

    "$REASON_PYTHON" -u -m reason.run_reason \
        --root "$REASON_DATA_ROOT" \
        "${scene_args[@]}" \
        "${target_args[@]}" \
        --model "$REASON_MODEL" \
        --prior-prompt "$REASON_PRIOR_PROMPT" \
        --ranking-score "$REASON_RANKING_SCORE" \
        --out-root "$REASON_OUT_ROOT"
}

# ---- 解析场景 ----
if [[ $# -eq 0 ]] || [[ "$1" == "--all" ]]; then
    SCENES=("${ALL_SCENES[@]}")
    shift 2>/dev/null || true
else
    SCENES=("$@")
fi

# ---- 核心函数：跑一次 ----
run_once() {
    local log_file="$1"; shift
    local scenes=("$@")
    local -a scene_args
    if [[ ${#scenes[@]} -eq 1 ]]; then
        scene_args=(--scene-id "${scenes[0]}")
    else
        scene_args=(--scene-ids "${scenes[@]}")
    fi

    local extra=()
    [[ -n "${D_PPS:-}" ]] && extra+=(--depth-sam2-points-per-side "$D_PPS")
    [[ -n "${D_CNL:-}" ]] && extra+=(--depth-sam2-crop-n-layers "$D_CNL")
    [[ -n "${D_PIT:-}" ]] && extra+=(--depth-sam2-pred-iou-thresh "$D_PIT")
    [[ -n "${D_SST:-}" ]] && extra+=(--depth-sam2-stability-score-thresh "$D_SST")
    local debug_args=()
    [[ "${DEBUG:-}" == "sam2" ]] && debug_args=(--debug "$DEBUG")

    {
        echo "========================================="
        echo " SmartGrasp Perception (Linux)"
        echo "========================================="
        echo "  Python:    $PYTHON"
        echo "  Scenes:    ${scenes[*]}"
        echo "  Mode:      $MODE"
        echo "  Data dir:  $SMARTGRASP_DATA_DIR"
        echo "  API URL:   $OPENAI_BASE_URL"
        [[ -n "${OPENAI_API_KEY:-}" ]] && echo "  API Key:   ✓ 已设置"
        echo "========================================="
        echo ""

        START_TIME=$(date +%s)
        if "$PYTHON" -u -m perception.run_perception \
            "${scene_args[@]}" --mode "$MODE" \
            --review-model-id "${REVIEW_MODEL_ID:-gpt-5.5}" \
            --review-api-key-env OPENAI_API_KEY \
            --review-base-url "$OPENAI_BASE_URL" \
            --review-timeout "${REVIEW_TIMEOUT:-300}" \
            --kernel-size "${KERNEL_SIZE:-11}" \
            --min-contact-pixels "${MIN_CONTACT_PIXELS:-50}" \
            --min-contact-ratio "${MIN_CONTACT_RATIO:-0.002}" \
            --depth-gap-threshold "${DEPTH_GAP_THRESHOLD:-0.5}" \
            --mask-clean-kernel "${MASK_CLEAN_KERNEL:-3}" \
            --proposal-min-area-ratio "${PROPOSAL_MIN_AREA_RATIO:-0.006}" \
            --proposal-max-area-ratio "${PROPOSAL_MAX_AREA_RATIO:-0.11}" \
            --proposal-border-fraction-threshold "${PROPOSAL_BORDER_FRACTION_THRESHOLD:-0.18}" \
            --sam2-points-per-side "$SAM2_PPS" \
            --sam2-crop-n-layers "$SAM2_CNL" \
            --sam2-pred-iou-thresh "$SAM2_PIT" \
            --sam2-stability-score-thresh "$SAM2_SST" \
            ${extra[@]+"${extra[@]}"} \
            ${DEVICE:+--device "$DEVICE"} \
            ${debug_args[@]+"${debug_args[@]}"} \
            ${SAVE_CANDIDATES:+--save-candidates}; then
            EXIT_CODE=0
        else
            EXIT_CODE=$?
        fi
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo ""
            echo "Output check priority 1: perception/summary.json, perception/scene_image.png, perception/depth.npy"
            if ! check_scene_outputs "${scenes[@]}"; then
                EXIT_CODE=1
            fi
        fi
        if [[ $EXIT_CODE -eq 0 ]]; then
            run_reason_for_scenes "${scenes[@]}" || EXIT_CODE=$?
        fi
        if [[ $EXIT_CODE -ne 0 ]]; then
            echo ""
            echo "Input search priority 1: input/scene_<id>/{summary.json,scene_image.png,depth.npy,input.txt}"
            echo "Input search priority 2: data/scene_<id>/perception/summary.json, scene_image.png, depth.npy"
            if check_scene_outputs "${scenes[@]}"; then
                echo "  generated perception inputs found; fallback source check skipped"
            else
                echo ""
                echo "Input search priority 3: *.parquet, npz_file.zip"
                check_source_inputs || true
            fi
        fi
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo "========================================="
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo " generation check complete (${DURATION}s)"
        else
            echo " ❌ FAIL  (${DURATION}s, exit code=$EXIT_CODE)"
        fi
        echo "========================================="
        return "$EXIT_CODE"
    } &> "$log_file"
}

# ---- 执行 ----
if [[ "${PIPELINE_MODE:-0}" == "1" ]]; then
    # 被 run_pipeline.sh 调用，不单独写 log，输出到 stdout 由 pipeline 统一收集
    run_once "/dev/stdout" "${SCENES[@]}" || exit $?
elif [[ ${#SCENES[@]} -eq 1 ]]; then
    # 单场景
    LOG_NUM=$(find logs -maxdepth 1 -name '*.log' 2>/dev/null | wc -l | tr -d ' ')
    LOG_NUM=$((LOG_NUM + 1))
    LOG_FILE="logs/$(printf "%03d" "$LOG_NUM")_scene_${SCENES[0]}.log"
    echo "[$(printf "%03d" "$LOG_NUM")] scene ${SCENES[0]} → ${LOG_FILE}"
    if run_once "$LOG_FILE" "${SCENES[@]}"; then
        echo "    generation check complete — $LOG_FILE"
    else
        echo "    ❌ FAIL  — $LOG_FILE"
        exit 1
    fi
else
    # 批量 + 自动恢复
    MAX_RETRIES=10
    CURRENT_SCENES=("${SCENES[@]}")
    echo "批量运行 ${#CURRENT_SCENES[@]} 个场景（API error 自动恢复已启用）"

    for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
        LOG_NUM=$(find logs -maxdepth 1 -name '*.log' 2>/dev/null | wc -l | tr -d ' ')
        LOG_NUM=$((LOG_NUM + 1))
        LOG_FILE="logs/$(printf "%03d" "$LOG_NUM")_scenes_${CURRENT_SCENES[0]}-${CURRENT_SCENES[-1]}.log"
        echo "[$(printf "%03d" "$LOG_NUM")] batch ${#CURRENT_SCENES[@]} scenes → ${LOG_FILE}"

        RUN_OK=0
        if run_once "$LOG_FILE" "${CURRENT_SCENES[@]}"; then
            RUN_OK=1
        fi

        # 检查 API error → 裁剪剩余场景重试
        if grep -qi "API error" "$LOG_FILE" 2>/dev/null; then
            FAILED_SCENE=$(grep -oP '\[scene_\K\d+' "$LOG_FILE" | tail -1)
            echo "    ⚠️  API error at scene $FAILED_SCENE (attempt $attempt), 恢复中..."
            NEW_SCENES=()
            FOUND=0
            for s in "${ALL_SCENES[@]}"; do
                [[ "$s" == "$FAILED_SCENE" ]] && FOUND=1
                [[ $FOUND -eq 1 ]] && NEW_SCENES+=("$s")
            done
            if [[ ${#NEW_SCENES[@]} -gt 0 ]]; then
                CURRENT_SCENES=("${NEW_SCENES[@]}")
            fi
            sleep 5
            continue
        fi

        # 没有 API error，检查结果
        if [[ $RUN_OK -eq 1 ]]; then
            echo "    generation check complete — $LOG_FILE"
            echo "=== DONE ==="
            exit 0
        else
            echo "    ❌ FAIL  — $LOG_FILE"
            exit 1
        fi
    done
    echo "❌ 超过最大重试 ($MAX_RETRIES)"
    exit 1
fi
