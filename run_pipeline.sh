#!/usr/bin/env bash
# ============================================================================
# SmartGrasp Full Pipeline — Perception → Intent → Reason
# ============================================================================
# 用法：
#   bash run_pipeline.sh              → 跑全部场景
#   bash run_pipeline.sh 59           → 跑单个场景
#   bash run_pipeline.sh 59 242 691   → 跑指定多个场景
#   bash run_pipeline.sh 59 --instruction=input
#                                     → 从 input/scene_59 读取 RGB/depth 和指令
#
# 执行顺序：Perception（SAM2+VLM+遮挡图）→ Intent（VLM意图解析）→ Reason（分支分类+graspability评分）
#
# 环境变量覆盖：
#   RUN_INTENT=0              → 跳过 Intent，Reason 遍历所有物体
#   TARGET_ID=5               → 跳过 Intent，Reason 只跑指定 id
#   INSTRUCTION="拿左边扳手"   → 自定义 Intent 指令（覆盖 annotation）
#   --instruction=input       → 不传字面指令；使用 input/scene_<id> 中的 input.txt/instruction.txt
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

# ---- 全量场景列表（与 perception/run_perception.sh 一致）----
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
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-D3Kd8gupG4HqUgTMsawHZBPmlEolExOmFHgkUkPt6TKuhllT}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://www.highland-api.top/v1}"

# ---- Pipeline 开关 ----
RUN_INTENT="${RUN_INTENT:-1}"
PIPELINE_VERBOSE="${PIPELINE_VERBOSE:-0}"

# ---- Reason 配置 ----
REASON_DATA_ROOT="${REASON_DATA_ROOT:-data}"
REASON_MODEL="${REASON_MODEL:-gpt-5.5}"
REASON_PRIOR_PROMPT="${REASON_PRIOR_PROMPT:-graspability}"
REASON_RANKING_SCORE="${REASON_RANKING_SCORE:-ig_graspability}"

# ---- 解析场景和指令参数 ----
USE_ALL=0
INPUT_INSTRUCTION_MODE="${INPUT_INSTRUCTION_MODE:-0}"
SCENES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            USE_ALL=1
            shift
            ;;
        --instruction=input)
            INPUT_INSTRUCTION_MODE=1
            shift
            ;;
        --instruction=*)
            INSTRUCTION="${1#--instruction=}"
            INPUT_INSTRUCTION_MODE=0
            shift
            ;;
        --instruction)
            if [[ $# -lt 2 ]]; then
                echo "❌ --instruction requires a value" >&2
                exit 2
            fi
            if [[ "$2" == "input" ]]; then
                INPUT_INSTRUCTION_MODE=1
            else
                INSTRUCTION="$2"
                INPUT_INSTRUCTION_MODE=0
            fi
            shift 2
            ;;
        --*)
            echo "❌ Unknown option: $1" >&2
            exit 2
            ;;
        *)
            SCENES+=("$1")
            shift
            ;;
    esac
done

if [[ ${#SCENES[@]} -eq 0 ]] || [[ "$USE_ALL" == "1" ]]; then
    SCENES=("${ALL_SCENES[@]}")
fi

# ===================================================================
# 主流程：所有输出写入 log，终端只显示最终成功/失败
# ===================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"

PIPELINE_EXIT=0
{
    echo "========================================="
    echo " SmartGrasp Full Pipeline"
    echo "========================================="
    echo "  Time:     $(date)"
    echo "  Scenes:   ${SCENES[*]}"
    echo "  Input instruction mode: ${INPUT_INSTRUCTION_MODE}"
    echo "========================================="
    echo ""

    # ── Stage 1: Perception ──
    echo "━━━ Stage 1/2: Perception ━━━"
    P_START=$(date +%s)

    PERCEPTION_DETAIL_LOG="$(mktemp "/tmp/smartgrasp_pipeline_perception_${TIMESTAMP}.XXXXXX.log")"
    if RUN_REASON_AFTER_PERCEPTION=0 PIPELINE_MODE=1 bash perception/run_perception.sh "${SCENES[@]}" > "$PERCEPTION_DETAIL_LOG" 2>&1; then
        PERCEPTION_EXIT=0
    else
        PERCEPTION_EXIT=$?
    fi

    P_ELAPSED=$(($(date +%s) - P_START))
    if [[ $PERCEPTION_EXIT -ne 0 ]]; then
        echo "Perception: FAIL (${P_ELAPSED}s, exit=${PERCEPTION_EXIT})"
        echo ""
        echo "---- Perception output ----"
        cat "$PERCEPTION_DETAIL_LOG"
        rm -f "$PERCEPTION_DETAIL_LOG"
        exit $PERCEPTION_EXIT
    fi
    echo "Perception: OK (${P_ELAPSED}s)"
    if [[ "$PIPELINE_VERBOSE" == "1" ]]; then
        echo ""
        echo "---- Perception output ----"
        grep -v "Reason after perception: skipped" "$PERCEPTION_DETAIL_LOG" || true
    fi
    rm -f "$PERCEPTION_DETAIL_LOG"
    echo ""

    # ── Stage 2: Intent + Reason（逐场景计时）──
    echo "━━━ Stage 2/2: Intent + Reason ━━━"

    # 构建 target-source 参数
    TARGET_ARGS=()
    if [[ -n "${TARGET_ID:-}" ]]; then
        TARGET_ARGS=(--target-source id --target-id "$TARGET_ID")
    elif [[ "$RUN_INTENT" == "1" ]]; then
        TARGET_ARGS=(--target-source auto)
        if [[ "$INPUT_INSTRUCTION_MODE" != "1" && -n "${INSTRUCTION:-}" ]]; then
            TARGET_ARGS+=(--instruction "$INSTRUCTION")
        fi
    else
        TARGET_ARGS=(--target-source all)
    fi

    R_TOTAL=0
    SCENE_COUNT=0
    for scene_id in "${SCENES[@]}"; do
        SCENE_COUNT=$((SCENE_COUNT + 1))
        scene_reason_dir="$REASON_DATA_ROOT/scene_${scene_id}/reason"

        RS=$(date +%s)
        REASON_DETAIL_LOG="$(mktemp "/tmp/smartgrasp_pipeline_reason_${TIMESTAMP}_${scene_id}.XXXXXX.log")"
        if "$PYTHON" -u -m reason.run_reason \
            --root "$REASON_DATA_ROOT" \
            --scene-id "$scene_id" \
            "${TARGET_ARGS[@]}" \
            --model "$REASON_MODEL" \
            --prior-prompt "$REASON_PRIOR_PROMPT" \
            --ranking-score "$REASON_RANKING_SCORE" \
            --out-root "$scene_reason_dir" \
            --scene-root "$REASON_DATA_ROOT" \
            --quiet > "$REASON_DETAIL_LOG" 2>&1; then
            REASON_EXIT=0
        else
            REASON_EXIT=$?
        fi

        RE=$(($(date +%s) - RS))
        R_TOTAL=$((R_TOTAL + RE))
        if [[ $REASON_EXIT -ne 0 ]]; then
            echo "Reason: FAIL scene=${scene_id} (${RE}s, exit=${REASON_EXIT})"
            echo ""
            echo "---- Reason output (scene ${scene_id}) ----"
            cat "$REASON_DETAIL_LOG"
            rm -f "$REASON_DETAIL_LOG"
            exit $REASON_EXIT
        fi

        printf "Reason: OK scene=%s (%ss) -> %s\n" "$scene_id" "$RE" "$scene_reason_dir"
        if [[ "$PIPELINE_VERBOSE" == "1" ]]; then
            grep -E '\[TIMING\]|\[ERROR\]|\[scene-out\]' "$REASON_DETAIL_LOG" || true
        fi
        rm -f "$REASON_DETAIL_LOG"
    done
    echo ""
    echo "Intent+Reason: OK (${R_TOTAL}s, ${SCENE_COUNT} scenes)"
    echo ""

    # ── Summary ──
    TOTAL=$((P_ELAPSED + R_TOTAL))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Pipeline Complete"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Perception:    ${P_ELAPSED}s"
    echo "  Intent+Reason: ${R_TOTAL}s"
    echo "  Total:         ${TOTAL}s"

} > "$LOG_FILE" 2>&1 || PIPELINE_EXIT=$?

# ── 终端只输出最终结果 ──
if [[ $PIPELINE_EXIT -eq 0 ]]; then
    echo "✅ SUCCESS — $LOG_FILE"
else
    echo "❌ FAIL (exit=$PIPELINE_EXIT) — $LOG_FILE"
    exit $PIPELINE_EXIT
fi
